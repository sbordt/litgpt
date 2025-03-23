# run a pre-training experiment, optionally monitoring the training process

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

import torch
torch.set_float32_matmul_precision('high')  

import logging


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
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None, help="name of the experiment (required)")
    parser.add_argument("--output_dir", type=str, default="/mnt/lustre/work/luxburg/shared_data/moritz_sebastian_2025/")
    parser.add_argument("--data_dir", type=str, default="/mnt/lustre/work/luxburg/shared_data/dclm-baseline-1.0-tokenized-preview")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", default=False, help="resume training from the most recent checkpoint. the checkpoint needs to exist.")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level")
    # monitoring parameters
    parser.add_argument("--monitor", action="store_true", default=True)
    parser.add_argument("--no-monitor", action="store_false", dest="monitor", help="global toggle to turn off all monitoring")
    parser.add_argument("--monitor_interval", type=int, default=100)
    parser.add_argument("--reference-model", action="store_true", default=True, 
                    help="compare activations to the reference model at initialization")
    parser.add_argument("--no-reference-model", action="store_false", dest="reference_model",
                    help="disable comparison of activations to the reference model")
    parser.add_argument("--advanced_activation_differences", action="store_true", default=False)
    # parameters of the pre-training run
    parser.add_argument("--model", type=str, default="pythia-14m", help="model to train")
    parser.add_argument("--width", type=int, default=128, help="width scaling")
    parser.add_argument("--max_tokens", type=int, default=6400000000) # 6.4B is 2x Chinchilla for 160m model
    parser.add_argument("--warmup_steps", type=float, default=700)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument('--mup', action='store_true', help='Enable MUP (Maximal Update Parameterization)')
    parser.add_argument("--mup_input_alpha", type=float, default=1)
    parser.add_argument("--mup_output_alpha", type=float, default=1)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--tie_embeddings", action="store_true", default=False)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    # pytorch options
    parser.add_argument("--use-pytorch-profiler", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False, 
                    help="enable model compilation with torch.compile")
    parser.add_argument("--no-compile", action="store_false", dest="compile",
                    help="disable model compilation with torch.compile")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    SLURM_PROCID = int(os.environ.get('SLURM_PROCID', 0)) # Get SLURM process ID (equivalent to rank in distributed training)
    WORLD_SIZE = int(os.environ.get('SLURM_NTASKS', 1)) # Get total number of tasks
    if args.experiment_name is None:
        raise ValueError("Experiment name must be provided")
    if WORLD_SIZE > 1 and args.timestamp is None:
        raise ValueError("Timestamp must be provided for multi-GPU training")
    
    # setup a shared directory for all experiments with the same name
    experiment_name = args.experiment_name
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    if SLURM_PROCID == 0:
        os.makedirs(experiment_dir, exist_ok=True)

    # create a unique named directory for the current run
    run_name = f"pretrain-{args.model}"
    if args.run_name is not None:
        run_name = args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # append the current timestamp (day, hour, minute, second) to the run name
    if args.timestamp is not None:
        timestamp = args.timestamp
    run_name += f"-timestamp={timestamp}"
    run_dir = os.path.join(experiment_dir, run_name)
    if args.resume:
        if not os.path.exists(experiment_dir):
            raise ValueError(f"Directory {experiment_dir} does not exist, cannot resume training")
    else:
        if SLURM_PROCID == 0: # otherwise rank0 creates the directory
            os.makedirs(run_dir, exist_ok=False)

    # create the model config with appropriate width scaling
    model_config = Config.from_name(args.model)
    if args.mup:
        model_config = apply_mup(model_config, args.width, args.mup_input_alpha, args.mup_output_alpha) # add mup hyperparameters
    else:
        model_config = scale_width(model_config, args.width) # only scale the width

    if SLURM_PROCID == 0:
        print(f"SLURM_PROCID: {SLURM_PROCID}, SLURM_NTASKS: {WORLD_SIZE}")
        print_config_wrt_scaling(model_config)
        print("Number of model parameters: ", count_model_parameters(GPT(model_config))) 

    # weight initialization for mup or sp
    initialize_weights_fn = initialize_mup_weights if args.mup else initialize_standard_weights

    # dataset
    data = DclmData(data_path=Path(args.data_dir), seed=args.data_seed)

    # define the metrics that we use for monitoring. currently this is the same as the default, but here you see how to define custom metrics :)
    def l2_norm(tensor: torch.Tensor) -> torch.Tensor:
        """we compute the metric along the last dimension, which is the embedding dimension. then we just return the remaining tensor of shape BxS (input shape is BxSxE)"""
        return torch.linalg.vector_norm(tensor, ord=2, dim=-1)

    activation_metrics = {
        "l2norm": l2_norm,
    }
    parameter_metrics = {
        "l2norm": l2_norm,
    }
    gradient_metrics = {
        "l2norm": l2_norm,
    }
    activation_difference_metrics = {
        "l2norm": lambda x, y: l2_norm(x - y),
    }

    # create the training monitor
    training_monitor = ModuleMonitor(monitor_interval=args.monitor_interval, 
                                       monitor=args.monitor,
                                       logger=logging.getLogger("ModuleMonitor"),
                                       activation_metrics=activation_metrics,
                                       parameter_metrics=parameter_metrics,
                                       gradient_metrics=gradient_metrics,
                                       activation_difference_metrics=activation_difference_metrics)

    # setup training from scratch with the given configuration
    setup(None, 
          model_config=model_config,   # here we pass the model config, not the raw model
          out_dir=run_dir,
        precision=args.precision, 
        data = data,
        tokenizer_dir=None,
        resume=args.resume,
        train = TrainArgs(
            save_interval=args.save_interval,
            log_interval=1,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            max_tokens=args.max_tokens,
            max_seq_length=args.max_seq_length,
            max_norm=1.0,
            min_lr=0.1*args.lr, # reduce the learning rate to 10% of the initial value
            lr_warmup_steps=700,
            tie_embeddings=args.tie_embeddings,
        ),
        optimizer = {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": args.lr,
                "betas": (0.9, 0.95),
                "weight_decay": 0.1,
            }
        },
        eval = EvalArgs(
            interval=1000,
            max_iters=100000, # use the entire validation set
        ),
        devices = "auto",
        logger_name = "wandb",
        seed = args.seed,
        training_monitor = training_monitor,
        with_reference_model = args.reference_model,
        with_advanced_activation_differences = args.advanced_activation_differences,
        with_compile = args.compile,
        use_pytorch_profiler = args.use_pytorch_profiler,
        initialize_weights_fn = initialize_weights_fn,
        logger_kwargs = {
            "name": run_name,
            "project": experiment_name,
        },
        get_lr_fn = get_lr,
    )

    # pickle the log dicts of inidividual ranks
    if WORLD_SIZE > 1:
        with open(os.path.join(run_dir, f"log_dict_rank{SLURM_PROCID}.pkl"), "wb") as f:
            pkl.dump(training_monitor.get_all_metrics(), f)

    # rank0 merges the log dicts from all other ranks
    if WORLD_SIZE > 1 and SLURM_PROCID == 0:
        time.sleep(300) # give all ranks the time to pickle their log dicts
        log_dicts = []
        for i in range(1, WORLD_SIZE):
            with open(os.path.join(run_dir, f"log_dict_rank{i}.pkl"), "rb") as f:
                log_dicts.append(pkl.load(f))
        training_monitor.merge_log_dicts(log_dicts)
    
    # save the full log dict to HDF5 and pickle
    training_monitor.save_hdf5(os.path.join(run_dir, "log_dict.hdf5"))
    with open(os.path.join(run_dir, "log_dict.pkl"), "wb") as f:
        pkl.dump(training_monitor.get_all_metrics(), f)

    # if this worked out, we can remove the individual log dicts
    if WORLD_SIZE > 1 and SLURM_PROCID == 0:
        for i in range(1, WORLD_SIZE):
            os.remove(os.path.join(run_dir, f"log_dict_rank{i}.pkl"))

    logging.shutdown()