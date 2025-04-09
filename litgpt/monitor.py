# Monitor the training of pytorch modules.
import torch
import numpy as np
import math
from typing import Union, List
from numbers import Number
import logging
import sys
from contextlib import contextmanager
import re


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
class MonitorMixin:
    """A torch.nn.Module can subclass this class to log custom metrics during the forward pass.
    This is used to monitor the attention operation.

    During the foward pass, the module can obtain the training monitor via get_training_monitor(). 

    TrainingMonitor automatically calls set_training_monitor on all modules that subclass MonitoredModule.
    """
    def __init__(self):
        self.training_monitor = None

        self.is_reference_module = False 
        """Whether the module is a part of the reference module."""

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
class ModuleMonitor:
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
                 activation_difference_metrics=None,
                 parameter_metrics=None,                    # metric that are applied to all parameters
                 parameter_metrics_spec=None,               # metrics that are applied to specific parameters based on a regular expression or a custom strategy
                 parameter_difference_metrics_spec=None,    # metrics that are applied to specific parameter differences based on a regular expression or a custom strategy
                 gradient_metrics=None,
                 logger=None,
                 cpu_offload=False): 
        """Init the training monitor."""
        self.module = None
        self.module_hooks = {}
        self.activation_difference_hooks = {}   # used by monitor_activation_differences
        self.module_names = {}                  # a mapping from modules to their names
        self.cpu_offload = cpu_offload          # whether to offload activations to the CPU

        self.reference_module = None
        self.reference_module_hooks = {}
        self.ignore_reference_module_activations = False
        self.reference_module_names = {}        # a mapping from modules to their names
        self.reference_module_parameters = {}   # a mapping from parameter names to the parameters
        self.reference_module_activations = {}  # a mapping from module names to their activations

        self.monitor_interval = monitor_interval
        self.step = step_start
        self.monitor = monitor
        self.monitor_step = False # do we monitor the current gradient step?
        self.log_dict = {} # a dict to log the parameters, activations, etc. of all gradient steps. maps the step number to the log dict of that step.

        self.activation_metrics = activation_metrics if activation_metrics is not None else {}
        self.activation_difference_metrics = activation_difference_metrics if activation_difference_metrics is not None else {}
        self.parameter_metrics = parameter_metrics if parameter_metrics is not None else {}
        self.parameter_difference_metrics_spec = parameter_difference_metrics_spec if parameter_difference_metrics_spec is not None else {}
        self.gradient_metrics = gradient_metrics if gradient_metrics is not None else {}

        # compile regular expressions
        self.parameter_metrics_spec = {}
        for regex, metrics in parameter_metrics_spec.items():
            compiled_re = re.compile(regex)
            self.parameter_metrics_spec[compiled_re] = metrics

        # Initialize logger
        if logger is None:
            self.logger = logging.getLogger("ModuleMonitor")
            self.logger.addHandler(logging.NullHandler())  # Silent default logger
        else:
            self.logger = logger

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
            for hook in self.module_hooks.values(): # remove all hooks
                hook.remove()
            self.module_hooks = {}
            for hook in self.activation_difference_hooks.values():
                hook.remove()
            self.activation_difference_hooks = {}
            self.module_names = {}
            # if the module implements the MonitoredModule interface, remove the training monitor
            for _, m in module.named_modules():
                if isinstance(m, MonitorMixin):
                    m.set_training_monitor(None, False)

        # setup for the new module
        self.module = module
        for name, m in module.named_modules():
            self.module_names[m] = format_module_name(name)
            self.module_hooks[m] = m.register_forward_hook(self._get_activation_forwad_hook(self.module_names[m]))
            self.logger.debug(f"Step {self.step}: Registered forward hook for module %s", name)
            # if the module implements the MonitoredModule interface, set the training monitor
            if isinstance(m, MonitorMixin):
                m.set_training_monitor(self, False)


    def set_reference_module(self, module):
        """Set the reference module. The training monitor compares the activations and parameters of the monitored module with the reference module. 
        The reference module must take a forward pass with the same input as the monitored module BEFORE the monitored module takes a forward pass."""
        # remove any previous reference module
        self.remove_reference_module()

        self.reference_module = module
        if module is None:
            return
        
        # setup for the new reference module
        # register the modules via their names and set forward hooks
        for name, m in module.named_modules():
            self.reference_module_names[m] = format_module_name(name)
            self.reference_module_hooks[m] = m.register_forward_hook(self._get_reference_activation_forwad_hook())

            # if the module implements the MonitoredModule interface, set the training monitor
            if isinstance(m, MonitorMixin):
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
            if isinstance(m, MonitorMixin):
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


    #################################################################
    # Set the current step, get log dict
    #################################################################
    def start_step(self, step: int):
        """Notify the monitor that a new step has started. This function should be called before the forward pass. Returns True if the step should be monitored, False otherwise."""
        # clean-up any previous step (if not done already)
        self.reference_module_activations = {}        

        # do we monitor this step?
        self.monitor_step = self.is_step_monitored(step)
        self.step = step

        # check that there is a module to monitor
        if self.monitor_step and self.module is None:
            raise ValueError("No module to monitor. Please set the module first.")

        # if we monitor this step, create a new entry in the log dict
        if self.monitor_step: 
            self.log_dict[step] = {}
            
        return self.monitor_step
    

    def is_step_monitored(self, step: int):
        """Will we monitor the given step? Most useful to check if the next step will be monitored in case we need to do some setting up before the step."""
        if not self.monitor: # global toggle to turn off monitoring
            return False
        monitor_step = step % self.monitor_interval == 1
        if step <= 20:  # we always monitor the first 20 steps
            monitor_step = True
        if not monitor_step and step <= 100: # more frequent monitoring for the first 100 steps
            monitor_step = step % 20 == 1
        return monitor_step
    

    def is_monitoring(self):
        return self.monitor_step
    

    @contextmanager
    def no_monitoring(self):
        """
        Context manager to temporarily disable all monitoring. Use this to compute the validation loss and perform other forward operations that should not be monitored.
        """
        original_state = self.monitor_step
        self.monitor_step = False
        try:
            yield
        finally:
            self.monitor_step = original_state
    

    def get_step_metrics(self):
        """Return the log dict of the current step"""
        if self.monitor_step:
            return self.log_dict[self.step]
        return {}
    

    def get_all_metrics(self):
        """Return the full log dict with all steps that have been logged so far."""
        return self.log_dict
    

    def load_metrics(self, log_dict):
        """Load a log dict."""
        # assert that the current log dict is empty
        if len(self.log_dict) > 0:
            raise ValueError("The current log dict is not empty. Please clear it first.")
        self.log_dict = log_dict


    #################################################################
    # Basic logging functions
    #################################################################
    def log_scalar(self, key: str, value, force=False):
        """Monitor a scalar value such as the loss or the learning rate.
        
        If force is set to True, the value is logged even if we are not monitoring the current step.
        This is useful to monitor values like the final validation loss.

        A scalar value is first logged internally as an individual value to the given key.
        If the same key is logged again, loggend values are stored in a list and subsquently aggregated by after_forward.
        """
        if not self.is_monitoring() and not force:
            return
        
        if force and self.step not in self.log_dict:
            self.log_dict[self.step] = {}

        # if value is a tensor, it has to be a scalar value so we call item()
        if isinstance(value, torch.Tensor):
            value = value.item()

        # verify that value is a number
        if not isinstance(value, Number):
            raise ValueError("Value must be a number.")
        
        if key in self.log_dict[self.step]: # log different values of the same key in a list
            if not isinstance(self.log_dict[self.step][key], list):
                self.log_dict[self.step][key] = [self.log_dict[self.step][key]]
            self.log_dict[self.step][key].append(value)
        else:
            self.log_dict[self.step][key] = value


    def log_tensor(self, key: str, tensor: torch.Tensor):
        """Monitor a torch.Tensor.

        Tensors are stored in a list and aggregated by after_forward.
        
        Args:
            key (str): The key under which to log the tensor metrics.
            tensor (torch.Tensor): The tensor to monitor.
            force (bool): Whether to log even if not monitoring the current step.
        """
        if not self.is_monitoring():
            return

        # detach and flatten the tensor
        tensor = tensor.detach().flatten()

        # tensors are stored in a list before aggregation
        if key in self.log_dict[self.step]:
            self.log_dict[self.step][key].append(tensor)
        else:
            self.log_dict[self.step][key] = [tensor]


    def log_scalars(self, monitor_dict: dict, force=False):
        """Monitor a dictionary of scalar values."""
        for key, value in monitor_dict.items():
            self.log_scalar(key, value, force=force)


    #################################################################
    # Activations are logged with forward hooks
    #################################################################
    def monitor_activations(self, 
                            module :Union[str, torch.nn.Module], 
                            activations :torch.Tensor,
                            is_reference :bool = False):
        """Monitor the activations of a module.

           This function is automatically called by the forward hooks that are registered on all modules of a monitored model.
        
           In addition, this function can be used to monitor activations that are not the output of a module.
           This is what monitor_scaled_dot_product_attention does.
        """
        if not self.is_monitoring():
            return

        # assert that module_name is a string
        module_name = self._module_name(module, is_reference)

        # detach activations from the graph but keep on the device
        activations = activations.detach()

        # if called with the activations of the reference module, store them in the reference_module_activations dict
        if is_reference:
            if self.ignore_reference_module_activations:
                return

            # raise a warning if no reference module is set
            if not self.has_reference_module():
                self.logger.warning(f"Step {self.step}: Attempted to monitor activations of the reference module, but no reference module is set (for module %s).", module_name)
                return
            # raise a warning if the reference module has already stored activations for this module
            if module_name in self.reference_module_activations:
                self.logger.warning(f"Step {self.step}: Attempted to monitor activations of the reference module for module %s, but activations are already stored.", module_name)
                return
            # optionally, offoad the activations to the CPU. this is extremely expensive but saves GPU memory.
            if self.cpu_offload:
                activations = activations.cpu()
            # store the activations
            self.reference_module_activations[module_name] = activations 
            # we are done
            return

        # monitor the different activation metrics
        for metric_name, metric_fn in self.activation_metrics.items():
            # compute the metric
            result = metric_fn(activations)

            # monitor the metric
            log_entry = f"activation/{module_name}/{metric_name}"
            self.log_tensor(log_entry, result)

        # metrics based on the activations and the reference activations (most likely L2 norm)
        if self.has_reference_module():
            if module_name in self.reference_module_activations:
                ref_activations = self.reference_module_activations[module_name]
                if self.cpu_offload: # activations need to be on the same device
                    activations = activations.cpu()

                for metric_name, metric_fn in self.activation_difference_metrics.items():
                    # compute the metric
                    result = metric_fn(activations, ref_activations)

                    log_entry = f"activation_difference/{module_name}/{metric_name}"
                    self.log_tensor(log_entry, result)
            else:
                self.logger.warning(f"Step {self.step}: No reference module activations found for module %s", module_name)

        self.logger.debug(f"Step {self.step}: Monitored activations of module %s with shape %s", module_name, activations.shape)


    def _get_activation_forwad_hook(self, module_name : str):
        def hook(module, input, output):
            self.monitor_activations(module_name, output, is_reference=False)
        return hook
    

    def _get_reference_activation_forwad_hook(self):
        def hook(module, input, output):
            self.monitor_activations(module, output, is_reference=True)
        return hook
    

    def clear_reference_activations(self):
        """Clear the reference module activations. To be called after a mini-batch is done."""
        self.reference_module_activations = {}
    

    def aggregate_step(self):
        """This function is called after all mini-batches of a gradient step are done. It aggregates the batch statistics. """
        if self.is_monitoring():
            # aggregate metrics
            #suffixes = list(self.activation_metrics.keys())

            for k, v in list(self.log_dict[self.step].items()): # iterate over a copy because we modify the original dict
                if type(v) == list: #  and any(k.endswith(f".{suffix}") for suffix in suffixes)
                    if isinstance(v[0], torch.Tensor): # list of tensors (activations)
                        v = torch.cat(v)
                        self.log_dict[self.step][k] = torch.mean(v).item()
                        self.log_dict[self.step][k + ".std"] = torch.std(v).item()
                    else: # list of scalars
                        self.log_dict[self.step][k] = np.mean(v)
                        self.log_dict[self.step][k + ".std"] = np.std(v)

            # print total number of keys
            size_mb = sys.getsizeof(self.log_dict) / 1024 / 1024
            self.logger.info(
                f"Logged {len(self.log_dict[self.step])} keys at step {self.step}. Total size of log data: {size_mb:.2f} MB"
            )

    #######################################################################################
    # Monitoring of activation differences with  requires additional forward passes.
    #######################################################################################
    def monitor_activation_differences(self, comparison_model: torch.nn.Module):
        """Mointor the difference in activations between the monitored module and a comparison module. This can be used to perform a muP coordinate check.

        When this function is called by the user, the training monitor performs additional computations to compare the intermediate activations of the monitored module and the comparison module.

        During a forward pass of the monitored module, we pass the input of every module also into the corresponding module of the comparison model and monitor the difference of the outputs.
        This means that we perform a forward operation in the comparison model with the **intermediate** input activations of the monitored module.
        The comparison module must have the same architecture as the monitored module. It can, but does not have to be the reference module.

        The comparison model must be set before the forward pass of the monitored module.
        
        Note: This function only works if the monitored module and the comparison module are on the same (single) device. This does NOT work with FSDP.
        """
        if self.module is None:
            raise ValueError("No module to monitor. Please set the module first.")

        # validate that we can find all modules in the comparison model
        modules = dict(self.module.named_modules())
        comparison_modules = dict(comparison_model.named_modules())

        for name in modules.keys(): 
            if not name in comparison_modules:
                raise ValueError(f"Module {name} not found in the comparison model.")
            
        # remove any previous activation_difference_hooks
        for hook in self.activation_difference_hooks.values():
            hook.remove()
        self.activation_difference_hooks = {}
            
        # register hooks to perform the additional forward passes
        for name in modules.keys():
            mod = modules[name]
            comp_mod = comparison_modules[name]
            self.activation_difference_hooks[mod] = mod.register_forward_hook(self._get_activation_differences_forwad_hook(self.module_names[mod], comp_mod))
         

    def _get_activation_differences_forwad_hook(self, module_name : str, comp_mod: torch.nn.Module):
        def hook(module, input, output):
            if not self.is_monitoring():
                return

            # every part of the input that is a tensor needs to be detached from the computational graph
            input = (i.detach() if isinstance(i, torch.Tensor) else i for i in input)

            # perform an additional forward pass in the comparison module
            with torch.no_grad():
                self.ignore_reference_module_activations = True      # temporarily ignore reference module activation hooks (relevant if the comparison module is the reference module)
                comp_output = comp_mod(*input)
                self.ignore_reference_module_activations = False

            # compute difference metrics
            for metric_name, metric_fn in self.activation_difference_metrics.items():
                # compute the metric
                result = metric_fn(output, comp_output.detach())

                # monitor the metric
                log_entry = f"advanced_activation_difference/{module_name}/{metric_name}"
                self.log_tensor(log_entry, result)

            self.logger.debug(f"Step {self.step}: Monitored advanced activation differences of module %s with shape %s", module_name, output.shape)

        return hook


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
        if not self.is_monitoring():
            return

        module_name = self._module_name(module, is_reference)

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

                self.monitor_activations(f"{module_name}.head_{i_head}.query", q, is_reference=is_reference)
                self.monitor_activations(f"{module_name}.head_{i_head}.key", k, is_reference=is_reference)
                self.monitor_activations(f"{module_name}.head_{i_head}.value", v, is_reference=is_reference)
                if activation is not None:
                    self.monitor_activations(f"{module_name}.head_{i_head}.activation", o, is_reference=is_reference)
        else:
            self.logger.warning(f"Step {self.step}: monitor_scaled_dot_product_attention assumes that S == L and that the key query and value tensor have the same dimension.")

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        # monitor entropy of the attention weights
        if S == L and query.size(-3) == key.size(-3) and query.size(-1) == value.size(-1):
            n_head = query.size(-3)
            for i_head in range(n_head):
                w = attn_weight[..., i_head, :, :]
                entropy = -torch.sum(w * torch.log(w + 1e-8), dim=-1)
                self.log_tensor(f"attention_entropy/{module_name}.head_{i_head}", entropy)
                self.logger.debug(f"Step {self.step}: Monitored attention entropy for head %s with shape %s", i_head, entropy.shape)

        self.logger.debug(f"Step {self.step}: Monitored scaled dot product attention for module %s with query shape %s, key shape %s, value shape %s", module_name, query.shape, key.shape, value.shape)
        # return attn_weight @ value [This is provided in the activation argument]


    #################################################################
    # Logging of gradients
    #################################################################
    def monitor_gradients(self, before_clip=False):
        if not self.is_monitoring():
            return
        
        for name, param in self.module.named_parameters():
            if param.grad is None:
                self.logger.warning(f"Step {self.step}: Found a parameter where the gradient is None: %s", name)
                continue

            # log the different metrics (most likely the frobenius norm of the gradients)
            for metric_name, metric_fn in self.gradient_metrics.items():
                # we apply the metrics to the flattened gradient tensor
                result = metric_fn(param.grad.detach().flatten())

                # if result is a tensor, apply item()
                if isinstance(result, torch.Tensor):
                    result = result.item()

                # Create log entry
                log_entry = f"gradient/{format_module_name(name)}/{metric_name}"
                if before_clip:
                    log_entry = log_entry.replace("gradient/", "gradient_before_clip/", 1)

                self.log_scalar(log_entry, result)
                self.logger.debug(f"Step {self.step}: Monitored gradient of parameter %s with shape %s with %s %s (logged as %s)", name, param.grad.shape, metric_name, result, log_entry)


    #################################################################
    # Logging of parameters
    #################################################################
    def _monitor_parameter(self, name, param, metric_fn, metric_name):
        """Low-level function that logs a metric for a parameter."""
        # apply the metrics to the flattened parameter tensor
        result = metric_fn(param.flatten())

        # if result is a tensor, apply item()
        if isinstance(result, torch.Tensor):
            result = result.item()

        # Create log entry
        log_entry = f"parameter/{format_module_name(name)}/{metric_name}"
        self.log_scalar(log_entry, result)
        self.logger.debug(f"Step {self.step}: Monitored parameter %s with shape %s with %s %s (logged as %s)", name, param.shape, metric_name, result, log_entry)


    def monitor_parameters(self):
        """Monitor the parameters of the monitored module. This is usually called after a gradient step is done."""
        if not self.is_monitoring():
            return
        
        for name, param in self.module.named_parameters():
            # log metrics that apply to all parameters
            for metric_name, metric_fn in self.parameter_metrics.items():
                self._monitor_parameter(name, param, metric_fn, metric_name)
                
            # log metrics that specify via a regular expression that they only apply to specific named parameters
            for compiled_re, metrics_dict in self.parameter_metrics_spec.items():
                if compiled_re.match(name):
                    for metric_name, metric_fn in metrics_dict.items():
                        self.monitor_parameter(name, param, metric_fn, metric_name)

            # the difference in l2 norm to the reference module
            if self.reference_module is not None:
                key = format_module_name(name)
                ref_param = self.reference_module_parameters[key]
                log_entry = f"parameter_difference/{format_module_name(name)}/l2norm"
                diff = (param - ref_param).norm(p='fro').item()
                self.log_scalar(log_entry, diff)
    

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

        for step, step_logs in self.log_dict.items():
            for key, value in step_logs.items():
                if not (key.startswith("activation") or not key.startswith("activation_difference")):
                    continue

                # means
                if key.endswith("l2norm"):
                    means = [value]
                    for other_log_dict in other_log_dicts:
                         means.append(other_log_dict[step][key])
                    mean = np.mean(means)
                    new_log_dict[step][key] = mean
                # standard deviations
                elif key.endswith("l2norm.std"):
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
                try:
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
                except Exception as e:
                    self.logger.warning(f"Step {self.step}: Could not save parameter %s with keys %s and values %s: %s", parameter, keys, values, e)

    
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
        for entry_key in ModuleMonitor.read_hdf5_entry_keys(filename):
            keys, values = ModuleMonitor.read_hdf5_entry(filename, entry_key)
            log_dict[entry_key] = {k: v for (k, v) in zip(list(keys), list(values))}
        return log_dict 


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
                self.logger.warning(f"Step {self.step}: Module %s not found in the module names dict.", module)
                return "[unknown module]"
            module_name = self.module_names[module]
        assert isinstance(module_name, str)
        return module_name