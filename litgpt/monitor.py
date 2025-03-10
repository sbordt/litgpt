# Monitor the training of pytorch modules.
import torch
import numpy as np
import math
from typing import Union, List


def format_module_name(name: str):
    if name == "" or name == "_orig_mod":
        return "[root module]"
    for s in ["_forward_module.", "_orig_mod.", "_fsdp_wrapped_module."]:
        name = name.replace(s, "")
    return name
    

#################################################################
# Different Metrics that we can use to monitor activations,
# parameters, and gradients.
#
# Activation tensors will be passed in full shape, while
# parameters and gradients will be passed as flattened tensors.
#
# To work with both activation tensors and flattened tensors,
# the metrics should be computed along the last
# dimension of the tensor.
#################################################################
def l1_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Compute L1 norm along last dimension."""
    return torch.linalg.vector_norm(tensor, ord=1, dim=-1)

def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm along last dimension."""
    return torch.linalg.vector_norm(tensor, ord=2, dim=-1)

def mean(tensor: torch.Tensor) -> torch.Tensor:
    """Compute mean along last dimension."""
    return torch.mean(tensor, dim=-1)

def std(tensor: torch.Tensor) -> torch.Tensor:
    """Compute standard deviation along last dimension."""
    return torch.std(tensor, dim=-1)

def max_value(tensor: torch.Tensor) -> torch.Tensor:
    """Compute maximum value along last dimension."""
    return torch.max(tensor, dim=-1).values

def min_value(tensor: torch.Tensor) -> torch.Tensor:
    """Compute minimum value along last dimension."""
    return torch.min(tensor, dim=-1).values

def sparsity(tensor: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """Compute sparsity (fraction of near-zero values) along last dimension."""
    zeros = (torch.abs(tensor) < threshold).float()
    return torch.mean(zeros, dim=-1)


#################################################################
# Modules can subclass MonitoredModule to implement
# custom monitoring behavior.
#################################################################
class MonitoredModule:
    """A torch.nn.Module can subclass this class to log custom metrics during the forward pass.
    This is used to monitor the attention operation.

    During the foward pass, the module can obtain the training monitor via get_training_monitor(). 

    TrainingMonitor automatically calls set_training_monitor on all modules that subclass MonitoredModule.
    """
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


#################################################################
# The training mointor class
#################################################################
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
                 monitor = True,
                 activation_metrics=None,
                 parameter_metrics=None,
                 gradient_metrics=None): 
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

        self.activation_metrics = activation_metrics if activation_metrics is not None else {"l2norm": l2_norm}
        self.parameter_metrics = parameter_metrics if parameter_metrics is not None else {"l2norm": l2_norm}
        self.gradient_metrics = gradient_metrics if gradient_metrics is not None else {"l2norm": l2_norm}

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
            self.module_hooks[m] = m.register_forward_hook(self._get_activation_forwad_hook(self.module_names[m]))
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
            self.reference_module_hooks[m] = m.register_forward_hook(self._get_reference_activation_forwad_hook())

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


    #################################################################
    # HDF5 saving and loading
    #################################################################
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

                # Save keys and values separately, converting to list first
                # This avoids numpy array conversion issues
                keys = list(value_dict.keys())
                values = list(value_dict.values())

                keys = np.array(keys, dtype=np.float64)
                values = np.array(values, dtype=np.float64)

                group.create_dataset('keys', data=keys)
                group.create_dataset('values', data=values)

    
    @classmethod
    def read_hdf5_entry_keys(cls, filename):
        """Read the names of the entries in a hdf5 file."""
        import h5py

        with h5py.File(filename, 'r') as f:
            return list(f.keys())


    @classmethod 
    def read_hdf5_entry(cls, filename, entry_key):
        """
        Read a single entry from HDF5 file by its outer key.
        
        Args:
            filename (str): Input HDF5 filename
            entry_key (str): The outer key to load
        
        Returns:
            dict: Single inner dictionary corresponding to entry_key
            None: If entry_key doesn't exist
        """
        import h5py

        with h5py.File(filename, 'r') as f:
            # Convert key to string for HDF5 lookup
            key_str = str(entry_key)
            
            # Check if key exists
            if key_str not in f:
                return None
                
            # Read just this group
            inner_dict = {}
            for key in f[key_str]:
                value = f[key_str][key][()]
                # Convert numpy types back to Python native types
                if isinstance(value, np.generic):
                    value = value.item()
                inner_dict[key] = value
                
            return inner_dict['keys'], inner_dict['values']


    @classmethod
    def load_hdf5(cls, filename):
        log_dict = {}
        for entry_key in TrainingMonitor.read_hdf5_entry_keys(filename):
            keys, values = TrainingMonitor.read_hdf5_entry(filename, entry_key)
            log_dict[entry_key] = {k: v for (k, v) in zip(list(keys), list(values))}
        return log_dict 


    #################################################################
    # Monitoring of scalar values
    #################################################################
    def monitor_scalar(self, key, value):
        """Monitor a scalar value such as the loss or the learning rate."""
        if not self.is_monitoring():
            return
        
        if key in self.log_dict[self.step]:
            print("Warning: Monitoring ", key, " that has already been set in the current step.")

        self.log_dict[self.step][key] = value


    def monitor_scalars(self, monitor_dict: dict):
        """Monitor a dictionary of scalar values."""
        for key, value in monitor_dict.items():
            self.monitor_scalar(key, value)


    #################################################################
    # Activations are logged with forward hooks
    #################################################################
    def mointor_activations(self, 
                            module :Union[str, torch.nn.Module], 
                            activations :torch.Tensor,
                            is_reference :bool = False):
        """Monitor the activations of a module.

           This function is automatically called by the forward hooks that are registered on all modules of a monitored model.
        
           In addition, this function can be used to monitor activations that are not the output of a module.
           This is what monitor_scaled_dot_product_attention does.
        """
        if not self.is_monitoring():
            if self.verbose:
                print("Warning: Attempted to monitor activations of module ", module, " but not monitoring the current step.")
            return

        # assert that module_name is a string
        module_name = self._module_name(module, is_reference)

        # detach activations from the graph but keep on the device
        activations = activations.detach()

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
            self.reference_module_activations[module_name] = activations 
            # we are done
            return

        # Compute and log each metric
        for metric_name, metric_fn in self.activation_metrics.items():
            # Compute the metric
            result = metric_fn(activations)
            
            # Create log entry
            log_entry = f"{module_name}.activation.{metric_name}"
            self.log_dict[self.step].setdefault(log_entry, [])
            
            # Detach, move to CPU, and flatten
            result_cpu = result.detach().cpu().flatten()
            self.log_dict[self.step][log_entry].extend(result_cpu.tolist())
            
            if self.verbose:
                print(f"Info: Monitored {metric_name} of activations for module {module_name}")

        # if there is a reference module, log the difference in l2 norm between the activations of the monitored module and the reference module
        if self.has_reference_module():
            if module_name in self.reference_module_activations:
                log_entry = f"{module_name}.activation.diff.l2norm"
                if not log_entry in self.log_dict[self.step]:
                    self.log_dict[self.step][log_entry] = []
                ref_activation = self.reference_module_activations[module_name]
                diff_norm = torch.linalg.vector_norm(activations - ref_activation, ord=2, dim=-1, keepdim=False, out=None)
                diff_norm = diff_norm.detach().cpu()
                diff_norm = diff_norm.flatten()
                self.log_dict[self.step][log_entry].extend(diff_norm.tolist())
            else:
                if self.verbose:
                    print("Warning: No reference module activations found for module ", module_name)

        if self.verbose:
            print("Info: Monitored activations of module ", module_name, " with shape ", activations.shape)


    def _get_activation_forwad_hook(self, module_name : str):
        def hook(module, input, output):
            self.mointor_activations(module_name, output, is_reference=False)
        return hook
    

    def _get_reference_activation_forwad_hook(self):
        def hook(module, input, output):
            self.mointor_activations(module, output, is_reference=True)
        return hook
    

    def clear_reference_activations(self):
        """Clear the reference module activations. To be called after a mini-batch is done."""
        self.reference_module_activations = {}
    

    def after_forward(self):
        """This function is called after all mini-batches of a gradient step are done. It aggregates the batch statistics. """
        if self.is_monitoring():
            # aggregate metrics
            suffixes = list(self.activation_metrics.keys())

            for k, v in list(self.log_dict[self.step].items()): # we iterate over a copy to modify the original dict
                if type(v) == list and any(k.endswith(f".{suffix}") for suffix in suffixes):
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
    # Monitor the activations inside a scaled dot product attention operation
    # This function has to be called by the monitored module
    #################################################################
    def monitor_scaled_dot_product_attention(self,
                                             module :Union[str, torch.nn.Module],  # the module that calls the attention function
                                             query :torch.Tensor, 
                                             key :torch.Tensor, 
                                             value :torch.Tensor,
                                             attn_mask=None, 
                                             dropout_p=0.0,
                                             is_causal=False, 
                                             scale=None, 
                                             enable_gqa=False,
                                             activation :torch.Tensor = None, # the return value of torch.nn.functional.scaled_dot_product_attention (optional)
                                             is_reference :bool = False): 
        """Monitor a scaled dot product attention operation. 
        Follows the signature of the pytorch function torch.nn.functional.scaled_dot_product_attention.
        """
        module_name = self._module_name(module, is_reference)

        if self.verbose:
            print("Info: Monitoring scaled dot product attention for module ", module_name, " with query shape ", query.shape, " key shape ", key.shape, " value shape ", value.shape)

        # the monitoring here is VERY inefficient, as we have to recompute the attention weights
        # first, we detach all tensors from the computational graph
        query = query.detach()
        key = key.detach()
        value = value.detach()
        if activation is not None:
            activation = activation.detach()
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
                if activation is not None:
                    o = activation[..., i_head, :, :]

                self.mointor_activations(f"{module_name}.head_{i_head}.query", q, is_reference=is_reference)
                self.mointor_activations(f"{module_name}.head_{i_head}.key", k, is_reference=is_reference)
                self.mointor_activations(f"{module_name}.head_{i_head}.value", v, is_reference=is_reference)
                if activation is not None:
                    self.mointor_activations(f"{module_name}.head_{i_head}.activation", o, is_reference=is_reference)
        else:
            if self.verbose:
                print("Warning: monitor_scaled_dot_product_attention assumes that S == L and that the key query and value tensor have the same dimension.")

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        # return attn_weight @ value [This is provided in the activation argument]


    #################################################################
    # Logging of gradients
    #################################################################
    def monitor_gradients(self, before_clip=False):
        if not self.is_monitoring():
            return
        
        for name, param in self.module.named_parameters():
            if param.grad is None:
                if self.verbose:
                    print("Warning: Found a parameter where the gradient is None:", name)
                continue

            # log the different metrics (most likely the frobenius norm of the gradients)
            for metric_name, metric_fn in self.gradient_metrics.items():
                # we apply the metrics to the flattened gradient tensor
                result = metric_fn(param.grad.detach().flatten())

                # if result is a tensor, apply item()
                if isinstance(result, torch.Tensor):
                    result = result.item()

                # Create log entry
                log_entry = f"{format_module_name(name)}.gradient.{metric_name}"
                if before_clip:
                    log_entry += ".before_clip"

                self.monitor_scalar(log_entry, result)

                if self.verbose:
                    print("Info: Parameter ", name, " has gradient of shape ", param.grad.shape, " with ", metric_name, " ", result, "(logged as ", log_entry, ")")


    #################################################################
    # Logging of parameters
    #################################################################
    def monitor_parameters(self):
        if not self.is_monitoring():
            return
        
        for name, param in self.module.named_parameters():

            # log the different metrics (most likely the frobenius norm of the parameters)
            for metric_name, metric_fn in self.parameter_metrics.items():
                # we apply the metrics to the flattened parameter tensor
                result = metric_fn(param.flatten())

                # if result is a tensor, apply item()
                if isinstance(result, torch.Tensor):
                    result = result.item()

                # Create log entry
                log_entry = f"{format_module_name(name)}.{metric_name}"
                self.monitor_scalar(log_entry, result)

                if self.verbose:
                    print("Info: Parameter ", name, " has shape ", param.shape, " with ", metric_name, " ", result, "(logged as ", log_entry, ")")
                    
            # the difference in l2 norm to the reference module
            if self.reference_module is not None:
                key = format_module_name(name)
                ref_param = self.reference_module_parameters[key]
                log_entry = f"{format_module_name(name)}.diff.l2norm"
                diff = (param - ref_param).norm(p='fro').item()
                self.monitor_scalar(log_entry, diff)


    #################################################################
    # Mointor the inner product betwenn the module input
    # and a custom vector.
    #################################################################
    def monitor_activation_updates(self, comparison_model: torch.nn.Module):
        """Mointor the change in activations due to a change in parameters.

        During a forward pass of a new input through the model,
        we pass the input of every module also into the corresponding module of the comparison model.
        This means that we perform a forward operation in the comparison model
        with the **intermediate** activations of the new model. 
        
        The advantage of this approach over the reference module is that the difference in activations does not 
        accumulate from layer to layer. This is because every module in the comparison model
        gets exactly the same input as the correpsonding module in the model.
        Of course, the comparison model can be the reference model (but it can also be an independent copy
        of the model from the previous gradient step).

        The disadvantage of this approach is that I don't know how to make it work with FSDP.
        So this only works on a single GPU.
         
        This is likely the best way to perform a muP coordinate check during the first couple of gradient steps.
        """
        pass
         

    def _get_activation_update_forwad_hook(self, module_name : str):
        def hook(module, input, output):
            self.mointor_activations(module_name, output, is_reference=False)
        return hook
    

    #################################################################
    # Merge another log dict from a distributed traing run.
    #     
    # With FSDP, we monitor activations and their 
    # differences on each gpu separately, then merge them into rank0
    # after training.
    #################################################################
    def merge_log_dicts(self, other_log_dicts: List[dict]):
        """Note: For the math to be valid, we need to merge all distributed log dicts in one step."""
        from copy import deepcopy
        new_log_dict = deepcopy(self.log_dict)

        for step, step_logs in new_log_dict.items():
            for key, value in step_logs.items():
                # means
                if key.endswith("activation.l2norm") or \
                   key.endswith("activation.diff.l2norm"):
                    means = [value]
                    for other_log_dict in other_log_dicts:
                         means.append(other_log_dict[step][key])
                    mean = np.mean(means)
                    new_log_dict[step][key] = mean
                # standard deviations
                elif key.endswith("activation.l2norm.std") or \
                     key.endswith("activation.diff.l2norm.std"):
                    # gather the stds
                    stds = [value]
                    for other_log_dict in other_log_dicts:
                        stds.append(other_log_dict[step][key])
                    # now gather the means
                    means_key = key.removesuffix(".std")
                    means = [step_logs[means_key]]
                    for other_log_dict in other_log_dicts:
                         means.append(other_log_dict[step][means_key])
                    mean = np.mean(means)
                    # compute σ² = [(σ₁² + (μ₁ - μ)²) + (σ₂² + (μ₂ - μ)²) + ...]/n
                    std = np.mean([s**2 + (m - mean)**2 for s, m in zip(stds, means)])**0.5
                    new_log_dict[step][key] = std

        self.log_dict = new_log_dict # successfull merge


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