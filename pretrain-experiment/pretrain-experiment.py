# run a pre-training experiment, optionally monitoring the training process

#import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from litgpt.model import GPT, Config
from litgpt.monitor import ModuleMonitor
from litgpt.pretrain import setup
from litgpt.args import TrainArgs, EvalArgs

from litgpt.mup import scale_width, apply_mup, initialize_mup_weights, initialize_standard_weights, print_parameter_info

from pathlib import Path
import argparse
import os
import pickle as pkl
from datetime import datetime
import time

from project_utils import count_model_parameters, print_config_wrt_scaling

from DclmData import DclmData

import math

import logging

import torch
torch.set_float32_matmul_precision('high')  

# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float, step: int) -> float:
    """For some reason, litgpt.pretrain specifies the learning rate in terms of the iteration, that is a micro batch.

    For our use I have additionally added the gradient step as an argument to the function signature.
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()

    # experiment name, logging and filesystem
    parser.add_argument("--run_name", type=str, default=None, help="run name (not required)")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None, help="name of the experiment (required)")
    parser.add_argument("--output_dir", type=str, default="/mnt/lustre/work/luxburg/shared_data/moritz_sebastian_2025/")
    parser.add_argument("--data_dir", type=str, default="/mnt/lustre/work/luxburg/shared_data/dclm-baseline-1.0-tokenized")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", default=False, help="resume training from the most recent checkpoint. the checkpoint needs to exist.")
    parser.add_argument("--auto-cancel", action="store_true", default=False, help="enable auto-cancel for the job. this will cancel the job if the validation loss is ever larger than 11.")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level")
    # monitoring parameters
    parser.add_argument("--monitor", action="store_true", default=True)
    parser.add_argument("--no-monitor", action="store_false", dest="monitor", help="global toggle to turn off all monitoring")
    parser.add_argument("--monitor_interval", type=int, default=100)
    parser.add_argument("--reference_model", type=str, default="init", help="compare activations to a reference model. 'init': compare to the model at initialization, 'previous_step': compare to the model at the previous gradient step")
    parser.add_argument("--activation_differences", action="store_true", default=False)
    parser.add_argument("--mup_coordinate_check", action="store_true", default=False)
    parser.add_argument("--monitor_cpu_offload", action="store_true", default=False, help="enable CPU offloading for the reference model")
    # parameters of the pre-training run
    parser.add_argument("--model", type=str, default="pythia-14m", help="model to train")
    parser.add_argument("--norm_class_name", type=str, default="LayerNorm", help="Set to LayerNorm or RMSNorm")
    parser.add_argument("--qk_norm", action="store_true", default=False, help="enable qk normalization")
    parser.add_argument("--layernorm_no_elementwise_affine", action="store_true", default=False, help="disable elementwise affine for LayerNorm")
    parser.add_argument("--width", type=int, default=128, help="width scaling")
    parser.add_argument("--max_tokens", type=int, default=1400000000) # 6.4B is 2x Chinchilla for 160m model
    parser.add_argument("--warmup_steps", type=float, default=700)
    parser.add_argument("--stop_step", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="AdamW", help="AdamW or SGD")
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument('--mup', action='store_true', help='Enable MUP (Maximal Update Parameterization)')
    parser.add_argument("--mup_input_alpha", type=float, default=1)
    parser.add_argument("--mup_output_alpha", type=float, default=1)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--tie_embeddings", action="store_true", default=False)
    parser.add_argument("--mse_loss", action="store_true", default=False, help="Use MSE loss instead of cross entropy")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    # pytorch options
    parser.add_argument("--use-pytorch-profiler", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False, help="enable model compilation with torch.compile")
    parser.add_argument("--no-compile", action="store_false", dest="compile", help="disable model compilation with torch.compile")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    SLURM_PROCID = int(os.environ.get('SLURM_PROCID', 0)) # Get SLURM process ID (equivalent to rank in distributed training)
    WORLD_SIZE = int(os.environ.get('SLURM_NTASKS', 1)) # Get total number of tasks
    if args.experiment_name is None:
        raise ValueError("Experiment name must be provided")
    if WORLD_SIZE > 1 and args.run_id is None:
        raise ValueError("Run-id must be provided for multi-GPU training")
    if args.resume and args.run_id is None:
        raise ValueError("Run-id must be provided for resuming training")
    if not args.optimizer in ["AdamW", "SGD"]:
        raise ValueError("Optimizer must be AdamW or SGD")
    
    # setup a shared directory for all experiments with the same name
    experiment_name = args.experiment_name
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    if SLURM_PROCID == 0:
        os.makedirs(experiment_dir, exist_ok=True)

    # create a unique named directory for the current run
    run_name = f"pretrain-{args.model}"
    if args.run_name is not None:
        run_name = args.run_name
    run_id = datetime.now().strftime("%Y%m%d%H%M%S") # use the current timestamp (day, hour, minute, second) as the run id
    if args.run_id is not None:
        run_id = args.run_id
    run_name += f"-id={run_id}"
    run_dir = os.path.join(experiment_dir, run_name)
    if args.resume:
        if not os.path.exists(run_dir):
            raise ValueError(f"Run drectory {run_dir} does not exist, cannot resume training")
    else:
        if SLURM_PROCID == 0: # otherwise rank0 creates the directory
            os.makedirs(run_dir, exist_ok=False)
    logging.info(f"Run directory: {run_dir}")

    # create the model config
    model_config = Config.from_name(args.model)
    model_config.norm_class_name = args.norm_class_name
    model_config.rmsnorm_elementwise_affine = False # disable elementwise affine for RMSNorm
    model_config.qk_norm = args.qk_norm
    model_config.layernorm_no_elementwise_affine = args.layernorm_no_elementwise_affine
    model_config.__post_init__() # required as we re-set the norm class name

    if args.mup:
        if SLURM_PROCID == 0:
            print("Setting up model config for training with muP...")
        model_config = scale_width(model_config, 256) # set 256 to be the base width and scale the mup width multiplier with that
        model_config = apply_mup(model_config, args.width, args.mup_input_alpha, args.mup_output_alpha)
    else:
        model_config = scale_width(model_config, args.width) # scale the width

    if SLURM_PROCID == 0:
        print(f"SLURM_PROCID: {SLURM_PROCID}, SLURM_NTASKS: {WORLD_SIZE}")
        print_config_wrt_scaling(model_config)
        print("Number of model parameters: ", count_model_parameters(GPT(model_config))) 

    # weight initialization for mup or sp
    initialize_weights_fn = initialize_mup_weights if args.mup else initialize_standard_weights

    # optimizer configuration
    optimizer_args = {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": args.lr,
                "betas": (0.9, 0.95),
                "weight_decay": 0.1,
                "eps": 1e-12,
            }
        }
    if args.optimizer == "SGD":
        optimizer_args = {
            "class_path": "torch.optim.SGD",
            "init_args": {
                "lr": args.lr,
                "weight_decay": 0.0,
            }
        }

    # dataset
    data = DclmData(data_path=Path(args.data_dir), seed=args.data_seed)

    # define the metrics that we use for monitoring. currently this is the same as the default, but here you see how to define custom metrics :)
    def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
        """we compute the metric along the last dimension, which is the embedding dimension. then we just return the remaining tensor of shape BxS (input shape is BxSxE)"""
        return torch.linalg.vector_norm(tensor, ord=2, dim=-1)
    def matrix_opnorm(tensor: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_norm(tensor, ord=2)

    activation_metrics = {
        "l2norm": l2_norm,
    }
    activation_difference_metrics = {
        "l2norm": lambda x, y: l2_norm(x - y),
    }
    parameter_metrics_spec = {
        r".*": {"l2norm": lambda param: l2_norm(param.flatten())},                      # l2 norm for all parameters
        r".*(norm_.*|ln_f).*": {"opnorm": lambda param: param.abs().max(dim=-1).values},        # operator norm for normalization layers (the maximum parameter value)
        r".*(mlp\.(fc|proj)\.weight|lm_head).*" : {"opnorm": matrix_opnorm},            # operator norm for linear layers   
    }
    parameter_difference_metrics_spec = {
        # frobenius norm of the weight updates
        r".*" : {"l2norm": lambda param, ref_param: l2_norm((param-ref_param).flatten())},

        # operator norm of the weight updates
        r".*(norm_.*|ln_f).*": {"opnorm": lambda param, ref_param: (param-ref_param).abs().max(dim=-1).values},   
        r".*(mlp\.(fc|proj)\.weight|lm_head).*" : {"opnorm": lambda param, ref_param: matrix_opnorm(param-ref_param)},                        
    }
    gradient_metrics = {
        "l2norm": l2_norm,
    }


    # create the training monitor
    training_monitor = ModuleMonitor(monitor_interval=args.monitor_interval, 
                                       monitor=args.monitor,
                                       logger=logging.getLogger("ModuleMonitor"),
                                       activation_metrics=activation_metrics,
                                       activation_difference_metrics=activation_difference_metrics,
                                       parameter_metrics_spec=parameter_metrics_spec,
                                        parameter_difference_metrics_spec=parameter_difference_metrics_spec,
                                       gradient_metrics=gradient_metrics,
                                       cpu_offload=args.monitor_cpu_offload)


    # setup training from scratch with the given configuration
    setup(None, 
          model_config=model_config,   # here we pass the model config, not the raw model
          out_dir=run_dir,
        precision=args.precision, 
        data = data,
        tokenizer_dir=None,
        resume=args.resume,
        auto_cancel=args.auto_cancel,
        train = TrainArgs(
            save_interval=args.save_interval,
            log_interval=1,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            max_tokens=args.max_tokens,
            max_seq_length=args.max_seq_length,
            max_norm=args.clip_grad_norm,
            min_lr=0.1*args.lr, # reduce the learning rate to 10% of the initial value
            lr_warmup_steps=700,
            tie_embeddings=args.tie_embeddings,
        ),
        optimizer = optimizer_args,
        eval = EvalArgs(
            interval=1000,
            max_iters=100000, # use the entire validation set
        ),
        mse_loss=args.mse_loss,
        devices = "auto",
        logger_name = "wandb",
        seed = args.seed,
        training_monitor = training_monitor,
        reference_model_type = args.reference_model,
        with_activation_differences = args.activation_differences,
        with_mup_coordinate_check = args.mup_coordinate_check,
        with_compile = args.compile,
        stop_after_step= args.stop_step,
        use_pytorch_profiler = args.use_pytorch_profiler,
        initialize_weights_fn = initialize_weights_fn,
        logger_kwargs = {
            "name": run_name,
            "project": experiment_name,
        },
        get_lr_fn = get_lr,
    )

    # rank0 merges the log dicts from all other ranks
    if WORLD_SIZE > 1 and SLURM_PROCID == 0:
        time.sleep(300) # give all ranks the time to pickle their log dicts
        log_dicts = []
        for i in range(1, WORLD_SIZE):
            with open(os.path.join(run_dir, f"final/log_dict_rank{i}.pkl"), "rb") as f:
                log_dicts.append(pkl.load(f))
        training_monitor.merge_log_dicts(log_dicts)
    
    # save the full log dict to HDF5 and pickle
    training_monitor.save_hdf5(os.path.join(run_dir, "log_dict.hdf5"))
    with open(os.path.join(run_dir, "log_dict.pkl"), "wb") as f:
        pkl.dump(training_monitor.get_all_metrics(), f)

    logging.shutdown()