# Monitor the training of pytorch modules.
import torch
import numpy as np
import math
from typing import Union


def format_module_name(name: str):
    if name == "" or name == "_orig_mod":
        return "[root module]"
    for s in ["_forward_module.", "_orig_mod.", "_fsdp_wrapped_module."]:
        name = name.replace(s, "")
    return name
    

class MonitoredModule:
    """A torch.nn.Module can subclass this class to monitor custom metrics during training.

    During the foward pass, the module can obtain the training monitor via get_training_monitor(). 

    TrainingMonitor automatically calls set_training_monitor on all modules that subclass MonitoredModule.
    """
    #################################################################
    # Setup
    #################################################################
    def __init__(self):
        self.training_monitor = None
        self.is_reference_module = False

    def set_training_monitor(self, monitor =None, is_reference_module =False):
        self.training_monitor = monitor
        self.is_reference_module = is_reference_module

    def get_training_monitor(self):
        return self.training_monitor

    @property
    def is_monitoring(self):
        return self.training_monitor is not None and self.training_monitor.is_monitoring()


class TrainingMonitor:
    """Monitor the training of a pytorch modules.

    Supports a reference module and micro-batches. 

    The reference module is another copy of the module, and we compare the activations and parameters of the monitored module with the reference module. 
    The reference module must take a forward pass with the same input as the monitored module BEFORE the monitored module takes a forward pass.

    Mirco-batches are supported by allowing abitrarily many forward passes before after_forward is called. After_forward aggregates the statistics of the mini-batches. 
    If there is a reference module, the respective micro-batch of the refernce module must take place before the micro-batch of the monitored module.

    We use a simple convention for the keys under which different metrics are logged:
    
     - Module activations are logged under "{module_name}.activation". For example, "{module_name}.activation.l2norm" is the l2 norm of the activations.
     - Parameters logged under "{parameter_name}". For example, "{parameter_name}.l2norm" is the l2 norm of the parameters.
     - Gradients are logged under "{parameter_name}.gradient". For example, "{parameter_name}.gradient.l2norm" is the l2 norm of the gradients.
     - The difference between the module and the reference module is indicated by ".diff". For example, "{module_name}.activation.diff.l2norm" is the l2 norm of the difference between the activations of the module and the reference module.
     - Module-specific metrics are logged similarly, for example "{module_name}.head_1.keys.activation.l2norm" is the l2 norm of the activations of the keys of the first attention head of the module.
    """

    #################################################################
    # Setup
    #################################################################
    def __init__(self, 
                 module = None,
                 reference_module = None,
                 monitor_interval = 20,
                 step_start = 0,
                 monitor = True): 
        """Init the training monitor."""
        self.module = None
        self.module_hooks = {}
        self.module_names = {}      # a mapping from modules to their names

        self.reference_module = None
        self.reference_module_hooks = {}
        self.reference_module_names = {}        # a mapping from modules to their names
        self.reference_module_parameters = {}   # a mapping from parameter names to the parameters
        self.reference_module_activations = {}  # a mapping from module names to their activations

        self.verbose = False
        self.monitor_interval = monitor_interval
        self.step = step_start
        self.monitor = monitor
        self.monitor_step = False # do we monitor the current gradient step?
        self.log_dict = {} # a dict to log the parameters, activations, etc. of all gradient steps. maps the step number to the log dict of that step.

        if module is not None:
            self.set_module(module)

        if reference_module is not None:
            self.set_reference_module(reference_module)


    def set_module(self, module):
        """Set the module that we want to monitor. This function will register forward hooks on the module to monitor the activations. It will also remove any previously set module and reference module."""
        # remove any previous reference module
        self.remove_reference_module()
        
        # remove any previous module
        if self.module is not None:
            self.module = None
            for hook in self.module_hooks.values():
                hook.remove()
            self.module_hooks = {}
            self.module_names = {}
            # if the module implements the MonitoredModule interface, remove the training monitor
            for _, m in module.named_modules():
                if isinstance(m, MonitoredModule):
                    m.set_training_monitor(None, False)

        # setup for the new module
        self.module = module
        for name, m in module.named_modules():
            self.module_names[m] = format_module_name(name)
            self.module_hooks[m] = m.register_forward_hook(self.get_forwad_hook(self.module_names[m]))
            if self.verbose:
                print("Info: Registered forward hook for module ", name)
            # if the module implements the MonitoredModule interface, set the training monitor
            if isinstance(m, MonitoredModule):
                m.set_training_monitor(self, False)


    def set_reference_module(self, module):
        """Set the reference module. The training monitor compares the activations and parameters of the monitored module with the reference module. 
        The reference module must take a forward pass with the same input as the monitored module BEFORE the monitored module takes a forward pass."""
        # remove any previous reference module
        self.remove_reference_module()

        # register the modules via their names and set forward hooks
        self.reference_module = module
        for name, m in module.named_modules():
            self.reference_module_names[m] = format_module_name(name)
            self.reference_module_hooks[m] = m.register_forward_hook(self.get_reference_forwad_hook())

            # if the module implements the MonitoredModule interface, set the training monitor
            if isinstance(m, MonitoredModule):
                m.set_training_monitor(self, True)

        # register the parameters of the reference module
        for name, param in module.named_parameters():
            self.reference_module_parameters[format_module_name(name)] = param

        # assert that the reference module names contains the same keys as the monitored module names
        assert set(self.module_names.values()) == set(self.reference_module_names.values()), "The reference module must have the same structure as the monitored module (there are modules with different names)."


    def remove_reference_module(self):
        """Remove the reference module."""
        if not self.has_reference_module():
            return
        # notify the MonitoredModules
        for _, m in self.reference_module.named_modules():
            if isinstance(m, MonitoredModule):
                m.set_training_monitor(None, False)
        # remove the reference module
        self.reference_module = None
        for hook in self.reference_module_hooks.values():
            hook.remove()
        self.reference_module_hooks = {}
        self.reference_module_names = {}
        self.reference_module_parameters = {}
        self.reference_module_activations = {}


    def has_reference_module(self): 
        return self.reference_module is not None
    
    def set_verbose(self, verbose):
        self.verbose = verbose

    #################################################################
    # Set the current step, get log dict
    #################################################################
    def set_step(self, step):
        """Notify the monitor that a new step has started. This function should be called before the forward pass. Returns True if the step should be monitored, False otherwise."""
        # clean-up the previous step
        self.reference_module_activations = {}

        # do we monitor this step?
        self.step = step
        self.monitor_step = step % self.monitor_interval == 1
        if step <= 20:  # monitor the first 20 steps
            self.monitor_step = True
        if not self.monitor_step and step <= 100: # more frequent monitoring for the first 100 steps
            self.monitor_step = step % 20 == 1

        if not self.monitor: # global toggle to turn monitoring off
            self.monitor_step = False

        # if we monitor this step, create a new entry in the log dict
        if self.monitor_step: 
            self.log_dict[step] = {}
            
        return self.monitor_step
    

    def is_monitoring(self):
        return self.monitor_step
    

    def get_step_metrics(self):
        """Return the log dict of the current step"""
        if self.monitor_step:
            return self.log_dict[self.step]
        return {}
    

    def get_all_metrics(self):
        """Return the full log dict with all steps that have been logged so far."""
        return self.log_dict


    def condensed_log_dict(self):
        """Take the log_dict which has the form
        {
            step1: {"key1": value1, "key2": value2},
            step2: {"key1": value1, "key2": value2},
        }

        and return a new dict of the form
        {
            "key1": {step1: value1, step2: value2},
            "key2": {step1: value1, step2: value2},
        }
        """
        new_dict = {}
        for key, value in self.log_dict.items():
            for name, val in value.items():
                if name not in new_dict:
                    new_dict[name] = {}
                new_dict[name][key] = val
        return new_dict


    def save_hdf5(self, filename, condensed=True):
        """Save the log dict as hdf5."""
        import h5py

        log_dict = self.log_dict
        if condensed:
            log_dict = self.condensed_log_dict()

        with h5py.File(filename, 'w') as f:
            for parameter, value_dict in log_dict.items():
                # Create a group for each parameter
                group = f.create_group(parameter)
                print(parameter)

                # Save keys and values separately, converting to list first
                # This avoids numpy array conversion issues
                keys = list(value_dict.keys())
                values = list(value_dict.values())

                keys = np.array(keys, dtype=np.float64)
                values = np.array(values, dtype=np.float64)

                print(keys.shape, values.shape)
                group.create_dataset('keys', data=keys)
                group.create_dataset('values', data=values)


    #################################################################
    # Basic monitoring functions
    # These functions can also be called by the monitored module
    # to log custom metrics.
    #################################################################
    def monitor_scalar(self, key, value):
        """Log a scalar value."""
        if self.monitor_step:
            self.log_dict[self.step][key] = value
        else:
            if self.verbose:
                print("Warning: logging a value while not monitoring the current step.")


    #################################################################
    # Activations are logged with forward hooks
    #################################################################
    def mointor_activations(self, 
                            module :Union[str, torch.nn.Module], 
                            activations :torch.Tensor,
                            is_reference :bool = False):
        """Monitor the activations of a module."""
        if not self.is_monitoring():
            if self.verbose:
                print("Warning: Attempted to monitor activations of module ", module, " but not monitoring the current step.")
            return

        # assert that module_name is a string
        module_name = self._module_name(module, is_reference)

        # if called with the activations of the reference module, store them in the reference_module_activations dict
        if is_reference:
            # raise a warning if no reference module is set
            if not self.has_reference_module():
                print(f"Warning: Attempted to monitor activations of the reference module, but no reference module is set (for module {module_name}).")
                return
            # raise a warning if the reference module has already stored activations for this module
            if module_name in self.reference_module_activations:
                print(f"Warning: Attempted to monitor activations of the reference module for module {module_name}, but activations are already stored.")
                return
            # store the activations
            self.reference_module_activations[module_name] = activations.detach() # detach from the graph but keep on the device
            # we are done
            return
        
        # log the l2norm of the activations    
        # to support mini batches, we first keep all the norms in a list and average after the entire batch is done
        log_entry = f"{module_name}.activation.l2norm"
        self.log_dict[self.step].setdefault(log_entry, [])
        norm = torch.linalg.vector_norm(activations, ord=2, dim=-1, keepdim=False, out=None)  # shape [B, S]
        norm = norm.detach().cpu()
        norm = norm.flatten() # shape [B*S]
        self.log_dict[self.step][log_entry].extend(norm.tolist())

        # if there is a reference module, log the difference in l2 norm between the activations of the monitored module and the reference module
        if self.has_reference_module():
            if module_name in self.reference_module_activations:
                log_entry = f"{module_name}.activation.diff.l2norm"
                if not log_entry in self.log_dict[self.step]:
                    self.log_dict[self.step][log_entry] = []
                ref_activation = self.reference_module_activations[module_name]
                diff_norm = torch.linalg.vector_norm(activations - ref_activation, ord=2, dim=-1, keepdim=False, out=None)
                diff_norm = diff_norm.detach().cpu()
                diff_norm = diff_norm.flatten() # shape [B*S]
                self.log_dict[self.step][log_entry].extend(diff_norm.tolist())
            else:
                if self.verbose:
                    print("Warning: No reference module activations found for module ", module_name)

        if self.verbose:
            print("Info: Monitored activations of module ", module_name, " with shape ", activations.shape)


    def get_forwad_hook(self, module_name : str):
        def hook(module, input, output):
            self.mointor_activations(module_name, output, is_reference=False)
        return hook
    

    def get_reference_forwad_hook(self):
        def hook(module, input, output):
            self.mointor_activations(module, output, is_reference=True)
        return hook
    

    def clear_reference_activations(self):
        """Clear the reference module activations. To be called after a mini-batch is done."""
        self.reference_module_activations = {}
    

    def after_forward(self):
        """This function is called after all mini-batches of a gradient step are done. It aggregates the batch statistics. """
        if self.is_monitoring():
            for k, v in list(self.log_dict[self.step].items()): # we iterate over a copy to modify the original dict
                if k.endswith(".l2norm") and type(v) == list:
                    mean = np.mean(v)
                    std = np.std(v)  
                    self.log_dict[self.step][k] = mean
                    self.log_dict[self.step][k + ".std"] = std

            # print total number of keys
            if self.verbose:
                print("Training Monitor: Total number of logging keys in the current step:", len(self.log_dict[self.step]))

                # size of the dict in mb
                import sys
                size = sys.getsizeof(self.log_dict) / 1024 / 1024
                print("Training Monitor: Size of log_dict in MB:", size)


    #################################################################
    # Monitor a scaled dot product attention operation
    # This function has to be called by the monitored module
    # We currently monitor the activations
    #################################################################
    def monitor_scaled_dot_product_attention(self,
                                             module :Union[str, torch.nn.Module],  # the module that calls the attention function
                                             query :torch.Tensor, 
                                             key :torch.Tensor, 
                                             value :torch.Tensor,
                                             output :torch.Tensor = None, # the return value of torch.nn.functional.scaled_dot_product_attention (optional)
                                             attn_mask=None, 
                                             dropout_p=0.0,
                                             is_causal=False, 
                                             scale=None, 
                                             enable_gqa=False,
                                             is_reference :bool = False):
        """Monitor a scaled dot product attention operation. 
        Follows the signature of the pytorch function torch.nn.functional.scaled_dot_product_attention.
        """
        module_name = self._module_name(module, is_reference)

        if self.verbose:
            print("Info: Monitoring scaled dot product attention for module ", module_name, " with query shape ", query.shape, " key shape ", key.shape, " value shape ", value.shape)

        # the monitoring here is VERY inefficient, as we have to recompute the attention weights
        # first, we detach all tensors from the graph
        query = query.detach()
        key = key.detach()
        value = value.detach()
        if output is not None:
            output = output.detach()
        if attn_mask is not None:
            attn_mask = attn_mask.detach()

        # now we follow the reference implementation, see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device) 
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        # monitoring for multi-head attention with n heads
        if S == L and query.size(-3) == key.size(-3) and query.size(-1) == value.size(-1):
            n_head = query.size(-3)
            for i_head in range(n_head):
                q = query[..., i_head, :, :]        # [B, S, D]
                k = key[..., i_head, :, :]
                v = value[..., i_head, :, :]
                if output is not None:
                    o = output[..., i_head, :, :]

                self.mointor_activations(f"{module_name}.head_{i_head}.query", q, is_reference=is_reference)
                self.mointor_activations(f"{module_name}.head_{i_head}.key", k, is_reference=is_reference)
                self.mointor_activations(f"{module_name}.head_{i_head}.value", v, is_reference=is_reference)
                if output is not None:
                    self.mointor_activations(f"{module_name}.head_{i_head}.output", o, is_reference=is_reference)
        else:
            if self.verbose:
                print("Warning: monitor_scaled_dot_product_attention assumes that S == L and that the key query and value tensor have the same dimension.")

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        # return attn_weight @ value [This is provided in output if the user wants us to monitor the output]


    #################################################################
    # Logging of gradients
    #################################################################
    def monitor_gradients(self, before_clip=False):
        if self.is_monitoring():
            for name, param in self.module.named_parameters():
                # check if param has a gradient
                if param.grad is not None:
                    # the l2 norm of the gradient
                    log_entry = f"{format_module_name(name)}.gradient.l2norm"
                    if before_clip:
                        log_entry += ".before_clip"
                    norm = param.grad.norm(p='fro').item()
                    self.monitor_scalar(log_entry, norm)
                    if self.verbose:
                        print("Info: Parameter ", name, " has gradient of shape ", param.grad.shape, " with l2 norm ", norm, "(logged as ", log_entry, ")")
                else:
                    if self.verbose:
                        print("Warning: Parameter without gradient:", name)
    

    #################################################################
    # Logging of parameters
    #################################################################
    def monitor_parameters(self):
        if self.is_monitoring():
            for name, param in self.module.named_parameters():
                # the l2 norm of the parameter
                log_entry = f"{format_module_name(name)}.l2norm"
                norm = param.norm(p='fro').item()
                self.monitor_scalar(log_entry, norm)

                # the difference in l2 norm to the reference module
                if self.reference_module is not None:
                    key = format_module_name(name)
                    ref_param = self.reference_module_parameters[key]
                    log_entry = f"{format_module_name(name)}.diff.l2norm"
                    diff = (param - ref_param).norm(p='fro').item()
                    self.monitor_scalar(log_entry, diff)

                if self.verbose:
                    print("Info: Parameter ", name, " has shape ", param.shape, " with l2 norm ", norm, "(logged as ", log_entry, ")") 
        


    #################################################################
    # Internal helper functions
    #################################################################
    def _module_name(self, module :Union[str, torch.nn.Module], is_reference :bool) -> str:
        """Look up the name of a torch.nn.Module in the appropriate dict."""
        module_name = module
        if isinstance(module_name, str): # when the module name is already provided as a string
            return module_name
        # handle normal and reference modules
        if is_reference:
            module_name = self.reference_module_names[module]
        else:
            if not module in self.module_names:
                print("Warning: Module ", module, " not found in the module names dict.")
                return "[unknown module]"
            module_name = self.module_names[module]
        assert isinstance(module_name, str)
        return module_name