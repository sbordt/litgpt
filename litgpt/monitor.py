# Monitor the training of pytorch modules.
#
# We use a simple convention for the keys under which different metrics are logged:
#
# - Module activations are logged under "{module_name}.activation". For example, "{module_name}.activation.l2norm" is the l2 norm of the activations.
# - Parameters logged under "{parameter_name}". For example, "{parameter_name}.l2norm" is the l2 norm of the parameters.
# - Gradients are logged under "{parameter_name}.gradient". For example, "{parameter_name}.gradient.l2norm" is the l2 norm of the gradients.
# - The difference between the module and the reference module is indicated by ".diff". For example, "{module_name}.activation.diff.l2norm" is the l2 norm of the difference between the activations of the module and the reference module.
# - Module-specific metrics are logged similarly, for example "{module_name}.head_1.keys.activation.l2norm" is the l2 norm of the activations of the keys of the first attention head of the module.
#
#



import torch
import numpy as np
import math
from typing import Union


def format_module_name(name):
    if name == "" or name == "_orig_mod":
        return "[root module]"
    return name.removeprefix("_forward_module.").removeprefix("_orig_mod.")
    


class MonitoredModule:
    """This class provides a convient interface to the TrainingMonitor for subclasses. 
    
    A torch.nn.Module can subclass this class to monitor custom metrics during training.   

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

    def is_monitoring(self):
        return self.training_monitor is not None and self.training_monitor.is_monitoring()


    #################################################################
    # Interface with the training monitor
    # To be called by the monitored module
    #################################################################
    
    def monitor_scaled_dot_product_attention(self, *args, **kwargs):
        """See the corresponding function in the TrainingMonitor class."""
        if not self.is_monitoring():
            return
        # add the module itself as the first argument
        args = [self] + list(args)
        # add is_reference to the kwargs
        kwargs["is_reference"] = self.is_reference_module
        self.training_monitor.monitor_scaled_dot_product_attention(*args, **kwargs)



class TrainingMonitor:

    #################################################################
    # Setup
    #################################################################
    def __init__(self, 
                 module = None,
                 reference_module = None,
                 monitor_interval = 20,
                 step_start = 0): 
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
        self.monitor_step = False
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
            self.module_hooks[m] = m.register_forward_hook(self.get_forwad_hook())
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
        if step <= 20:
            self.monitor_step = True

        # if we monitor this step, create a new entry in the log dict
        # also monitor the parameters
        if self.monitor_step: 
            self.log_dict[step] = {}
            self.monitor_parameters()

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


    def mointor_activations(self, 
                            module :Union[str, torch.nn.Module], 
                            activations :torch.Tensor,
                            is_reference :bool = False):
        """Monitor the activations of the module named module_name.
        Currently we log the l2-norm of the activations.
        """
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
            return
        
        # log the l2norm of the activations    
        log_entry = f"{module_name}.activation.l2norm"
        self.log_dict[self.step].setdefault(log_entry, [])
        norm = torch.linalg.vector_norm(activations, ord=2, dim=-1, keepdim=False, out=None)  # shape [B, S]
        norm = norm.detach().cpu()
        norm = norm.flatten() # shape [B*S]
        self.log_dict[self.step][log_entry].extend(norm.tolist())

        if self.verbose:
            print("Info: Monitored activations of module ", module_name, " with shape ", activations.shape)

        # if there is a reference module that has stored the corresponding activations, log the difference in l2 norm
        # TODO



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
    # Logging of module activations with forward hooks
    #################################################################


    def get_reference_forwad_hook(self):
        def hook(module, input, output):
            if self.is_monitoring():
                if module in self.reference_module_names:
                    module_name = self._module_name(module, True)
                    self.reference_module_activations[module_name] = output.detach() # detach from the graph but keep on the device
                else:
                    if self.verbose:
                        print("Warning: Unknown module with output shape:", output.shape, " in reference module.")
            
        return hook


    def get_forwad_hook(self):
        def hook(module, input, output):
            if self.is_monitoring():
                if module in self.module_names:
                    module_name = self._module_name(module, False)

                    # log the l2 norm of the activations
                    # to support mini batches, we first keep all the norms in a list and average after the entire batch is done
                    log_entry = f"{module_name}.activation.l2norm"
                    self.log_dict[self.step].setdefault(log_entry, [])
                    norm = torch.linalg.vector_norm(output, ord=2, dim=-1, keepdim=False, out=None)  # [B, S]
                    norm = norm.detach().cpu()
                    norm = norm.flatten() # [B*S]
                    self.log_dict[self.step][log_entry].extend(norm.tolist())

                    # if there is a reference module, log the different in l2 norm between the activations of the monitored module and the reference module
                    if self.reference_module is not None:
                        if module_name in self.reference_module_activations:
                            log_entry = f"{module_name}.activation.diff.l2norm"
                            if not log_entry in self.log_dict[self.step]:
                                self.log_dict[self.step][log_entry] = []
                            ref_activation = self.reference_module_activations[module_name]
                            diff = torch.linalg.vector_norm(output - ref_activation, ord=2, dim=-1, keepdim=False, out=None)
                            diff = diff.detach().cpu()
                            diff = diff.flatten()
                            self.log_dict[self.step][log_entry] = diff.tolist()
                        else:
                            if self.verbose:
                                print("Warning: No activation found for module ", module_name)
                        

                else:
                    if self.verbose:
                        print("Warning: Unknown module with output shape:", output.shape)
            
        return hook
    

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
                print("Info: Total number of keys in log_dict[step]:", len(self.log_dict[self.step]))

                # size of the dict in mb
                import sys
                size = sys.getsizeof(self.log_dict[self.step]) / 1024 / 1024
                print("Info: Size of log_dict[step] in MB:", size)






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

    def _module_name(self, module :Union[str, torch.nn.Module], is_reference :bool):
        """Look up the name of a torch.nn.Module in the appropriate dict."""
        module_name = module
        if isinstance(module_name, str): # when the module name is already provided as a string
            return module_name
        # handle normal and reference modules
        if is_reference:
            module_name = self.reference_module_names[module]
        else:
            module_name = self.module_names[module]
        assert module_name is not None
        assert isinstance(module_name, str)
        return module_name