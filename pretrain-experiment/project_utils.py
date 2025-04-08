from litgpt.args import TrainArgs
from litgpt.model import GPT, Config

from pathlib import Path

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

def print_config_wrt_scaling(config : Config):
    print("d_model: ", config.n_embd)
    print("n_head: ", config.n_head)
    print("d_head: ", config.head_size)
    print("n_layer: ", config.n_layer)
    print("MLP intermediate size: ", config.intermediate_size)


def scale_config(config : Config, width : int):
    """Scale the width of the transformer"""
    config.n_embd = width
    # assert that width is a multiple of the original width
    assert width % config.n_embd == 0, "width must be divisible by n_embd"
    assert config.n_embd % config.head_size == 0, "n_embd must be divisible by head_size"
    config.n_head = config.n_embd // config.head_size
    config.intermediate_size = 4 * config.n_embd
    return config


def litgpt_model(model_name: str, width_scaling: int = None):
    model_config = Config.from_name(model_name)
    if width_scaling is not None:
        model_config = scale_config(model_config, width_scaling)
        print_config_wrt_scaling(model_config)
    return GPT(model_config)


def custom_model(model_name, width_scaling):
    model_config = Config.from_name(model_name)
    if width_scaling is not None:
        model_config = scale_config(model_config, width_scaling)
    # turn off bias terms
    model_config.bias = False
    # rms normn instead of layer norm
    model_config.norm_class_name = "RMSNorm"
    model_config.__post_init__()
    print(model_config)
    return GPT(model_config)


def wandb_args_to_dict(args_list):
    """convert the command line arguments of a run logged in wandb to a nicely formatted dictionary"""
    args_dict = {}
    i = 0
    
    while i < len(args_list):
        # Check if we have a flag (starts with --)
        if args_list[i].startswith('--'):
            key = args_list[i][2:]  # Remove the -- prefix
            
            # Check if there's a next item and it's not a flag
            if i + 1 < len(args_list) and not args_list[i + 1].startswith('--'):
                args_dict[key] = args_list[i + 1]
                i += 2  # Skip both the key and value
            else:
                # Flag without a value, set to True or some default
                args_dict[key] = True
                i += 1
        else:
            # Skip non-flag items that aren't associated with a flag
            i += 1
    
    return args_dict