"""Litgpt training with Maximal Update Parametrization (μP).

   muP means that:
    - the initial weights of the output layer (lm_head) are set to zero
    - the input embedding forward pass output is multiplied by a tunable parameter (input_alpha)
    - the output unembedding forward pass output is multiplied by a tunable parameter (output_alpha)
    - 


   Inspired by https://github.com/EleutherAI/nanoGPT-mup/

   Note that this module does not (yet) support all classes of models. 
"""

from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.cli import instantiate_class
import lightning as L

from litgpt.utils import reset_parameters

from litgpt.config import Config

import math
from dataclasses import dataclass
from functools import partial

import torch.nn as nn
import inspect


@dataclass
class MuPArgs:
    """Arguments for training with Maximal Update Parametrization (μP).
    """

    enabled: bool = False
    """Whether to use muP. If False then all other mup variables are ignored"""

    width_multiplier: float = 1.0
    """mup_width_multiplier = width / base_width where base_width is typically 256"""

    input_alpha: float = 1.0
    """Tunable multiplier applied to input embedding forward pass output"""

    output_alpha: float = 1.0
    """Tunable multiplier applied to output unembedding forward pass output"""

    with_attention_head_1_over_n: bool = False
    """Whether to use 1/n scaling for attention heads."""


def _add_mup_args_to_config(config: Config, mup_args: MuPArgs) -> Config:
    """Add the MuP arguments to the config"""
    config.mup_args = mup_args
    return config


def has_mup_enabled(config: Config) -> bool:
    """Check if MuP is enabled in the config"""
    # check if the config has the mup_args attribute and if it is enabled
    return hasattr(config, 'mup_args') and config.mup_args.enabled


def scale_width(config : Config, width : int):
    """Scale the model width."""
    if not width % config.n_embd == 0:
        raise ValueError("the provided width must be a multiple of the original width")
    if not width % config.head_size == 0:
        raise ValueError("n_embd must be divisible by the head_size")
    config.n_embd = width
    config.n_head = config.n_embd // config.head_size
    config.intermediate_size = 4 * config.n_embd
    return config


def apply_mup(config : Config, 
              width : int,
              input_alpha : float = 1.0,
              output_alpha : float = 1.0):
    """Train a model with muP (Maximal Update Parametrization).

    This function scales the model width to the specified value and adds the muP arguments to the config.

    Note: This function does not modify the model itself. It only adjusts the config.
    """
    if has_mup_enabled(config):
        raise ValueError("Model has already been scaled with muP. Aborting because this could inidcate an error.")
    # currently we only support the pythia models
    # would need to check the details of other models to see if anything needs to be adjusted for muP
    if not "pythia-" in config.name:
        raise ValueError("Currently only pythia models are supported for muP")
    # adjust model parameters
    width_multiplier = width / config.n_embd
    config = scale_width(config, width)
    # add mup args to config
    mup_args = MuPArgs(True, width_multiplier, input_alpha, output_alpha)
    config = _add_mup_args_to_config(config, mup_args)
    return config


def initialize_standard_weights(fabric: L.Fabric, model, n_layer: int, n_embd: int) -> None:
    r"""Standard weight initialization. This is the GPT-NeoX weight initialization from pretrain.py.
    """
    from litgpt.model import LLaMAMLP, CausalSelfAttention

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def initialize_mup_weights(fabric: L.Fabric, model, n_layer: int, n_embd: int) -> None:
    r"""muP weight initialization. We set the initial weights of the output layer (lm_head) to zero.
    
    Otherwise this is equal to the standard weight initialization.
    """
    from litgpt.model import LLaMAMLP, CausalSelfAttention

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # we initialize the embedding layer with a constant standard deviation
    for mod in model.modules():
        if isinstance(mod, nn.Embedding):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / 256)) # this part is equivalent to the standard initialization for width=256

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    print("Performing muP weight initialization...")
    print(f"Model type: {type(model)}")
    print(f"Model has config attribute: {hasattr(model, 'config')}")
    print(f"Model has mup_args attribute: {hasattr(model.config, 'mup_args')}")

    # set the output layer weights to zero
    if has_mup_enabled(model.config):
        model.lm_head.reset_parameters = partial(init_weights, model.lm_head, std=0.0)
    else:
        print("WARNING: MuP is not enabled. Ignoring MuP weight initialization.")

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def instantiate_torch_mup_optimizer(optimizer: dict, model, **kwargs):
    # Special care taken where some optimizers do not have some parameters referenced in some of the code, for example "fused" in the pretrain.py script:
    #   bnb.optim.AdamW8bit
    #   grokadamw.GrokAdamW
    #   torch.optim.RMSprop
    # TODO this currently only supports adam! add support for sgd and others later
    optimizer = dict(optimizer)
    class_module, class_name = optimizer["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    optimizer_cls = getattr(module, class_name)

    valid_params = set(inspect.signature(optimizer_cls).parameters)
    kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}
    optimizer["init_args"].update(kwargs)

    if not has_mup_enabled(model.config):
        print("WARNING: MuP is not enabled. Ignoring MuP optimizer instantiation.")
        return instantiate_class(model.parameters(), optimizer)

    # define parameter groups for mup
    weight_decay = optimizer["init_args"].get("weight_decay", 0.0)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    mup_params = []
    other_params = []
    for n, p in param_dict.items():
        if n.endswith('attn.weight') or n.endswith('fc.weight') or n.endswith('proj.weight'):   # note that we do not scale the learning rate of the output layer (lm_head). we scale the input to the output layer (lm_head) in the forward pass instead.
            mup_params.append(p)
        else:
            other_params.append(p)
    optim_groups = [
        # we set a custom parameter lr_scale for each group
        # the training code is responsible for using this scale factor after
        # scheduling the overall leraning rate for the current step
        {"params": mup_params, "lr_scale": 1/model.config.mup_args.width_multiplier, "weight_decay": weight_decay * model.config.mup_args.width_multiplier},        # adujust weight decay for mup params: we want to keep the same effective weight decay for all parameters.
        {"params": other_params, "lr_scale": 1, "weight_decay": weight_decay},
    ]
    print(f"Number of parameters with muP learning rate scaling: {sum(p.numel() for p in mup_params if p.requires_grad)}")

    return instantiate_class(optim_groups, optimizer)


def count_parameters(parameters):
    return sum(p.numel() for p in parameters)


def print_parameter_info(model):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    print("Total number of parameters:", count_parameters(param_dict.values()))
    print("Total number of trainable parameters:", count_parameters(param_dict.values()))

    



