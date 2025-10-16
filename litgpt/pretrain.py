# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

#
# we take this default litgpt pre-trainign script and modify it to monitor the training process of the model
#
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # when resuming from checkpoint, fall back to eager.
torch._dynamo.config.cache_size_limit = 25   # allow to monitor up to 25 steps with a compiled model withoug falling back to eager

import math
import pprint
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
from contextlib import nullcontext
import copy
import gc

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal

from torch.distributed.fsdp import FullyShardedDataParallel
from torch.profiler import profile, ProfilerActivity, record_function

from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.data import DataModule, TinyLlama
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    extend_checkpoint_dir,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
)

from litgpt.monitor import ModuleMonitor
from litgpt.mup import has_mup_enabled, instantiate_torch_mup_optimizer

import pickle


def check_modules_equal(module1: nn.Module, module2: nn.Module, 
                        rtol: float = 1e-5, atol: float = 1e-8,
                        check_device: bool = False) -> bool:
    """
    Check if two PyTorch modules have the same architecture and parameters.
    
    Args:
        module1: First PyTorch module
        module2: Second PyTorch module
        rtol: Relative tolerance for parameter comparison (default: 1e-5)
        atol: Absolute tolerance for parameter comparison (default: 1e-8)
        check_device: Whether to check if parameters are on the same device (default: False)
    
    Returns:
        True if modules are identical, False otherwise
    """
    import logging

    # Configure logger
    logger = logging.getLogger(__name__)
    
    # Check module class type
    if type(module1) != type(module2):
        logger.error(f"Module types differ: {type(module1).__name__} vs {type(module2).__name__}")
        return False
    
    # Check string representation (gives a good overview of architecture)
    if str(module1) != str(module2):
        logger.error(f"Module architecture differs:\nModule1:\n{str(module1)}\nModule2:\n{str(module2)}")
        return False
    
    # Get state dictionaries
    state_dict1 = module1.state_dict()
    state_dict2 = module2.state_dict()
    
    # Check if they have the same keys (parameter and buffer names)
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            logger.error(f"Missing in module2: {missing_in_2}")
        if missing_in_1:
            logger.error(f"Missing in module1: {missing_in_1}")
        return False
    
    # Check each parameter and buffer
    for key in keys1:
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        # Check shape
        if param1.shape != param2.shape:
            logger.error(f"Shape mismatch for '{key}': {param1.shape} vs {param2.shape}")
            return False
        
        # Check dtype
        if param1.dtype != param2.dtype:
            logger.error(f"Dtype mismatch for '{key}': {param1.dtype} vs {param2.dtype}")
            return False
        
        # Check device (optional)
        if check_device and param1.device != param2.device:
            logger.error(f"Device mismatch for '{key}': {param1.device} vs {param2.device}")
            return False
        
        # Check values
        if not torch.allclose(param1, param2, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(param1 - param2.to(param1.device))).item()
            logger.error(f"Value mismatch for '{key}': max difference = {max_diff}")
            return False
    
    # Check module attributes (like dropout rate, etc.)
    # This is done by checking named_modules
    modules1 = dict(module1.named_modules())
    modules2 = dict(module2.named_modules())
    
    if set(modules1.keys()) != set(modules2.keys()):
        logger.error("Submodule structure differs")
        return False
    
    # Check specific attributes for common layer types
    for name in modules1.keys():
        m1 = modules1[name]
        m2 = modules2[name]
        
        if type(m1) != type(m2):
            logger.error(f"Submodule type mismatch at '{name}': {type(m1).__name__} vs {type(m2).__name__}")
            return False
        
        # Check common attributes based on module type
        if isinstance(m1, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m1.in_features != m2.in_features or m1.out_features != m2.out_features:
                logger.error(f"Feature size mismatch at '{name}'")
                return False
            if m1.bias is None != m2.bias is None:
                logger.error(f"Bias existence mismatch at '{name}'")
                return False
                
        elif isinstance(m1, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            attrs_to_check = ['kernel_size', 'stride', 'padding', 'dilation', 'groups']
            for attr in attrs_to_check:
                if getattr(m1, attr) != getattr(m2, attr):
                    logger.error(f"Conv attribute '{attr}' mismatch at '{name}'")
                    return False
                    
        elif isinstance(m1, nn.Dropout):
            if m1.p != m2.p:
                logger.error(f"Dropout rate mismatch at '{name}': {m1.p} vs {m2.p}")
                return False
                
        elif isinstance(m1, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m1.num_features != m2.num_features:
                logger.error(f"BatchNorm features mismatch at '{name}'")
                return False
            if m1.eps != m2.eps or m1.momentum != m2.momentum:
                logger.error(f"BatchNorm parameters mismatch at '{name}'")
                return False
    
    logger.info("Modules are identical")
    return True


def setup(
    model_name: str,
    model_config: Optional[Config] = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Literal["auto"], Path] = False,
    auto_cancel: bool = False,                          # whether to cancel the training if the validation loss starts to diverge                   
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    mse_loss: bool = False,                             # replace cross entropy loss with MSE loss (applies the softmax to the logits in both cases!)
    optimizer: Union[str, Dict] = "AdamW",
    devices: Union[int, str] = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "wandb",
    seed: int = 42,
    training_monitor :ModuleMonitor = None,             # montior object to log activations, gradients and parameters during training
    reference_model_type: str = None,                   # "init" or "previous_step"        
    with_activation_differences = False,                # whether to monitor the activation differences between the model and the reference model
    with_mup_coordinate_check = False,                  # wheter the perform a muP coordinate check (performs additional forward passes)
    with_compile: bool = True,                          # whether to compile the model
    stop_after_step: int = None,                        # stop training after this number of steps (does not influence the training process, learning rate, etc.)
    initialize_weights_fn: Optional[callable] = None,   # specify a custom function to initialize the model weights
    initialize_optimizer_fn: Optional[callable] = None, # specify a custom function to initialize the optimizer
    get_lr_fn: Optional[callable] = None,               # specify a custom learning rate schedule
    use_pytorch_profiler: bool = False,
    logger_kwargs: Optional[Dict] = None,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``. Overrides the `model_name` if specified.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.

        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    if with_mup_coordinate_check:   # mup coordinate check uses the computations from the forward pass of the reference model
        with_activation_differences = True

    if model_config is None:
        # Support both model_name options: meta-llama/Meta-Llama-3-8B & Meta-Llama-3-8B
        try:
            model_config = Config.from_name(model_name)
        except ValueError:
            print(f"Model name {model_name} is not supported.\n")
            available_models = "\n".join(sorted(name_to_config))
            print(f"Available values:\n{available_models}")
            quit()

    hparams = capture_hparams()
    data = TinyLlama() if data is None else data

    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    # check if "project" is in logger_kwargs and pass it as "name" to choose_logger
    project = f"pretrain-{config.name}"
    if logger_kwargs and "project" in logger_kwargs:
        project = logger_kwargs["project"]
        del logger_kwargs["project"]
        
    logger = choose_logger(
        logger_name, out_dir, project=project, resume=False, log_interval=train.log_interval, entity="mup_limitations", **(logger_kwargs or {})  # need to check how to setup wandb resuming
    )

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=[logger]
    )

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric,
        devices,
        seed,
        initial_checkpoint_dir,
        resume,
        auto_cancel,
        config,
        data,
        out_dir,
        tokenizer_dir,
        tokenizer,
        train,
        eval,
        optimizer,
        mse_loss,
        training_monitor,
        reference_model_type,
        with_activation_differences,
        with_mup_coordinate_check,
        with_compile,
        stop_after_step,
        initialize_weights_fn,
        initialize_optimizer_fn,
        get_lr_fn,
        use_pytorch_profiler,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Literal["auto"], Path],
    auto_cancel: bool,
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    mse_loss: bool,
    training_monitor :ModuleMonitor = None,
    reference_model_type: str = None,
    with_activation_differences: bool = False,
    with_mup_coordinate_check: bool = False,
    with_compile: bool = True,
    stop_after_step: Optional[int] = None,
    initialize_weights_fn: Optional[callable] = None,
    initialize_optimizer_fn: Optional[callable] = None,
    get_lr_fn: Optional[callable] = None,
    use_pytorch_profiler: bool = False,
) -> None:
    if initialize_weights_fn is None:
        initialize_weights_fn = initialize_weights

    validate_args(train, eval, initial_checkpoint_dir, resume)
    assert reference_model_type in ["init", "previous_step"], f"Invalid reference model type: {reference_model_type}"
    with_reference_model = with_activation_differences or with_mup_coordinate_check
    
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
        reference_model = GPT(config) if with_reference_model else None

    initialize_weights_fn(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    if reference_model is not None:
        reference_model.load_state_dict(model.state_dict()) # initialize the reference model to the model at time step zero
        check_modules_equal(model, reference_model) # sanity check

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    # register hooks to monitor the training process (according to torch.compile docs, this should happen before compilation)
    training_monitor.set_module(model)

    if with_activation_differences:
        training_monitor.set_reference_module(reference_model)

    if with_mup_coordinate_check:
        if fabric.world_size != 1:
            raise ValueError("muP coordinate check is only supported for single-GPU training")
        training_monitor.setup_mup_coordinate_check()

    # torch.compile the model and setup for distributed training
    if with_compile:
        model = torch.compile(model)
    #model = fabric.setup(model)
    model.to(fabric.device)

    if reference_model is not None:
        # reference_model = fabric.setup(reference_model)
        reference_model.to(fabric.device)
        check_modules_equal(model, reference_model) # again sanity check

    # lightning performs re-initialization of model weights with FSDP.
    # we need to re-load the weights of the reference model
    # After both models are FSDP wrapped and setup
    if reference_model is not None and isinstance(fabric.strategy, FSDPStrategy):
        with FullyShardedDataParallel.summon_full_params(model):
            with FullyShardedDataParallel.summon_full_params(reference_model):
                reference_model.load_state_dict(model.state_dict())

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    if initialize_optimizer_fn is not None:
        optimizer = initialize_optimizer_fn(optimizer, model, **extra_kwargs)
    else:
        if has_mup_enabled(config):
            optimizer = instantiate_torch_mup_optimizer(optimizer, model, **extra_kwargs)
        else:
            optimizer = instantiate_torch_optimizer(optimizer, model.parameters(), **extra_kwargs)
    optimizer = fabric.setup_optimizers(optimizer)
    print(f"Using an optimizer with {len(optimizer.param_groups)} parameter groups.")

    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train, model.max_seq_length)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    state = {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    # lenght of training and validation dataloaders
    fabric.print(f"Training dataloader length: {len(train_dataloader)}")
    fabric.print(f"Validation dataloader length: {len(val_dataloader)}")

    # first  batch of data
    fabric.print(f"First batch of data: {next(iter(train_dataloader))}")

    # shape of the first batch of data
    fabric.print(f"Shape of the first batch of data: {next(iter(train_dataloader)).shape}") 

    fabric.print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    fabric.print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # resume training, we load the model and optimizer state dicts, including the log_dict
    resume = find_resume_path(resume, out_dir) 
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_dataloader
        state['iter_num'] = checkpoint['iter_num']
        state['step_count'] = checkpoint['step_count']
        log_dict_path = resume.parent / f"log_dict_rank{fabric.global_rank}.pkl"
        with open(log_dict_path, "rb") as f:
            training_monitor.load_metrics(pickle.load(f))
        fabric.print(f"Resuming training from {resume}")
        del checkpoint

    torch.cuda.empty_cache()
    fabric.print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    fabric.print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    train_time = time.perf_counter()
    fit(fabric,
        devices, 
        state, 
        train_dataloader, 
        val_dataloader, 
        out_dir, 
        tokenizer_dir, 
        train, 
        eval,
        mse_loss,
        training_monitor,
        reference_model,
        reference_model_type,
        with_activation_differences,
        with_mup_coordinate_check,
        stop_after_step,
        get_lr_fn,
        use_pytorch_profiler,
        auto_cancel)

    # Save final checkpoint
    save_checkpoint(fabric, state, tokenizer_dir, out_dir / "final" / "lit_model.pth")
    
    # save final log_dict
    with open(out_dir / "final" / f"log_dict_rank{fabric.global_rank}.pkl", "wb") as f:
        pickle.dump(training_monitor.get_all_metrics(), f)

    total_tokens = state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size

    # Print formatted output
    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Total tokens  : {total_tokens:,}")
    fabric.print(f"| - Training Time : {(time.perf_counter()-train_time):.2f} s")
    fabric.print(f"| - Tok/sec       : {total_tokens / train_time:.2f} tok/s")
    fabric.print("| " + "-" * 40)

    if fabric.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        fabric.print("| Memory Usage")
        fabric.print(f"| - Memory Used   : {memory_used:.2f} GB")
    fabric.print(separator)

def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
    mse_loss: bool,
    training_monitor :ModuleMonitor = None,
    reference_model: Optional[nn.Module] = None,
    reference_model_type: str = None,
    with_activation_differences = False,
    with_mup_coordinate_check = False,
    stop_after_step: Optional[int] = None,
    get_lr_fn: Optional[callable] = None,
    use_pytorch_profiler: bool = False,
    auto_cancel: bool = False,
) -> None:
    if get_lr_fn is None:
        get_lr_fn = get_lr

    loss_fn = chunked_mse_loss_fn if mse_loss else chunked_cross_entropy

    model = state["model"]
    optimizer = state["optimizer"]

    if eval.initial_validation:
        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(fabric, model, val_dataloader, max_iters=2, verbose=False)   # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: loss_fn(y, x) #, chunk_size=0
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    fabric.barrier()
    total_t0 = time.perf_counter()

    warmup_iters = train.warmup_iters(devices, max_iters, train_dataloader)

    # put both models in training mode
    model.train()
    if reference_model is not None:
        reference_model.train()

    print(f"Model ln_f eps: {model.transformer.ln_f.eps}")
    print(f"Reference ln_f eps: {reference_model.transformer.ln_f.eps}")

    print(f"Model ln_f normalized_shape: {model.transformer.ln_f.normalized_shape}")
    print(f"Reference ln_f normalized_shape: {reference_model.transformer.ln_f.normalized_shape}")

    print(f"Model ln_f elementwise_affine: {model.transformer.ln_f.elementwise_affine}")
    print(f"Reference ln_f elementwise_affine: {reference_model.transformer.ln_f.elementwise_affine}")

    # print ln_f weights in both models
    print("L2 Norm of ln_f weights difference:", torch.norm(model.transformer.ln_f.weight - reference_model.transformer.ln_f.weight).item())
    print(model.transformer.ln_f.weight)
    print(reference_model.transformer.ln_f.weight)
    print(model.transformer.h[0].norm_1.weight)
    print(reference_model.transformer.h[0].norm_1.weight)
    
    # now the same for bias
    print("L2 Norm of ln_f bias difference:", torch.norm(model.transformer.ln_f.bias - reference_model.transformer.ln_f.bias).item())
    print(model.transformer.ln_f.bias)
    print(reference_model.transformer.ln_f.bias)
    print(model.transformer.h[0].norm_1.bias)
    print(reference_model.transformer.h[0].norm_1.bias)

    # DEBUGGING: ln_f bug
    def reference_model_ln_f_forward_hook(module, input, output):
        global reference_model_ln_f_output
        reference_model_ln_f_output = output.detach()

    def model_ln_f_forwad_hook(module, input, output):
        global model_ln_f_output
        model_ln_f_output = output.detach()
        print("L2 Norm of ln_f output difference:", torch.norm(model_ln_f_output - reference_model_ln_f_output).item())
        print(f"Model ln_f input shape: {model_ln_f_input.shape}")
        print(f"Reference ln_f input shape: {reference_model_ln_f_input.shape}")
        print(f"Model ln_f input dtype: {model_ln_f_input.dtype}")
        print(f"Reference ln_f input dtype: {reference_model_ln_f_input.dtype}")
        print("L2 Norm of ln_f input difference:", torch.norm(model_ln_f_input - reference_model_ln_f_input).item())

        manual_model_out = debug_layernorm(model.transformer.ln_f, model_ln_f_input)
        manual_ref_out = debug_layernorm(reference_model.transformer.ln_f, reference_model_ln_f_input)
        print(f"Manual computation difference: {torch.norm(manual_model_out - manual_ref_out).item()}")
        # compare manual with actual output
        print(f"Manual vs actual model output difference: {torch.norm(manual_model_out - model_ln_f_output).item()}")
        print(f"Manual vs actual reference output difference: {torch.norm(manual_ref_out - reference_model_ln_f_output).item()}")

    def reference_model_any_module_forward_hook(module, input, output):
        global reference_model_any_module_output
        reference_model_any_module_output = output.detach()

    def model_any_module_forward_hook(module, input, output):
        global model_any_module_output
        model.any_module_output = output.detach()
        print("L2 Norm of any module output difference:", torch.norm(model.any_module_output - reference_model_any_module_output).item())

    model.transformer.ln_f.register_forward_hook(model_ln_f_forwad_hook)
    reference_model.transformer.ln_f.register_forward_hook(reference_model_ln_f_forward_hook)

    # register at some random module
    model.transformer.h[3].mlp.register_forward_hook(model_any_module_forward_hook)
    reference_model.transformer.h[3].mlp.register_forward_hook(reference_model_any_module_forward_hook)

    # DEBUGGING: manual computation of the ln_f output
    def debug_layernorm(ln_module, input_tensor):
        """Manually compute layernorm to debug"""
        weight = ln_module.weight
        bias = ln_module.bias
        eps = ln_module.eps
        normalized_shape = ln_module.normalized_shape
        
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        output = (input_tensor - mean) / torch.sqrt(var + eps)
        if weight is not None:
            output = output * weight
        if bias is not None:
            output = output + bias
        return output

    # DEBUGGING: now also register hooks for the ln_f input difference
    def reference_model_ln_f_forward_pre_hook(module, input):
        global reference_model_ln_f_input
        reference_model_ln_f_input = input[0].detach()

    def model_ln_f_forward_pre_hook(module, input):
        global model_ln_f_input
        model_ln_f_input = input[0].detach()
        print("L2 Norm of ln_f input difference:", torch.norm(model_ln_f_input - reference_model_ln_f_input).item())

    model.transformer.ln_f.register_forward_pre_hook(model_ln_f_forward_pre_hook)
    reference_model.transformer.ln_f.register_forward_pre_hook(reference_model_ln_f_forward_pre_hook)

    # profile the training with the pytorch profiler (optional)
    if use_pytorch_profiler:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir / "torch_profiler"),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
        )
        profiler.start()
        fabric.print("Profiling with Pytorch profiler.")
    else:
        profiler = nullcontext()

    training_monitor.start_step(state["step_count"]+1)

    resume_data_iteration = False
    if state["iter_num"] > 0:
        fabric.print(f"We choose the hacky way to resume the dataloader by iterating over it again.")
        resume_data_iteration = True
        resume_iter_num = 0

    for train_data in train_iterator:
        # resume the dataloader by iterating over it again
        if resume_data_iteration:
            if resume_iter_num == state["iter_num"]:
                resume_data_iteration = False
                fabric.print(f"Resuming the dataloader after {resume_iter_num} iterations.")
            resume_iter_num += 1
            continue

        if state["iter_num"] >= max_iters:
            break

        if stop_after_step is not None and state["step_count"] >= stop_after_step:
            fabric.print(f"Stopping training after {stop_after_step} steps.")
            # we kill the python process, so we don't accidentally save a final checkpoint etc.
            raise SystemExit(1)

        # determine and set the learning rate for this iteration
        lr = get_lr_fn(optimizer.defaults["lr"], state["iter_num"], warmup_iters, max_iters, train.min_lr, state["step_count"]+1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            ### Begin muP code (this is general layer-wise learning rates code) ###
            if "lr_scale" in param_group:
                param_group["lr"] *= param_group["lr_scale"]
            ### End muP code ###

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.max_seq_length].contiguous().long()
        targets = train_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices) != 0

        # activation differences versus the reference model. we divide the micro batch into two parts to avoid gpu oom
        if training_monitor.is_monitoring() and reference_model is not None and train.micro_batch_size > 1:
            input_ids_first_one = input_ids[:train.micro_batch_size//2]
            input_ids_second_one = input_ids[train.micro_batch_size//2:]
            targets_first_one = targets[:train.micro_batch_size//2]
            targets_second_one = targets[train.micro_batch_size//2:]
            
            loss_sum = None
            for local_input_ids, local_targets in [(input_ids_first_one, targets_first_one), (input_ids_second_one, targets_second_one)]:
                #with fabric.no_backward_sync(model, enabled=is_accumulating):
                with nullcontext():
                    if with_activation_differences:                                      
                        with torch.no_grad():   
                            _ = reference_model(local_input_ids)     

                    logits = model(local_input_ids)
                    loss = loss_fn(logits, local_targets)
                    # fabric.backward(loss / train.gradient_accumulation_iters(devices))
                    loss = loss / train.gradient_accumulation_iters(devices)
                    loss.backward()
                    
                    if loss_sum is None:
                        loss_sum = loss.detach() / 2
                    else:
                        loss_sum += loss.detach() / 2

                    if with_mup_coordinate_check:
                        #with fabric.autocast():
                        with nullcontext(): 
                            training_monitor.mup_coordinate_check(fabric.device)

                training_monitor.after_micro_batch()                                                                    
            running_loss.update(loss_sum)

        # normal forward pass
        else:                                                                            
            #with fabric.no_backward_sync(model, enabled=is_accumulating):
            with nullcontext():
                # (micro-) batch
                logits = model(input_ids)
                loss = loss_fn(logits, targets)
                #fabric.backward(loss / train.gradient_accumulation_iters(devices))
                loss = loss / train.gradient_accumulation_iters(devices)
                loss.backward()

                if with_mup_coordinate_check:
                    #with fabric.autocast():
                    with nullcontext():
                        training_monitor.mup_coordinate_check(fabric.device)

            running_loss.update(loss.detach())
            
            training_monitor.after_micro_batch()

        if not is_accumulating:
            # monitor gradients before clip
            if fabric.world_size == 1:
                training_monitor.monitor_gradients(before_clip=True)
            elif fabric.global_rank == 0 and training_monitor.is_monitoring(): # FSDP
                with FullyShardedDataParallel.summon_full_params(model, with_grads=True, rank0_only=True): 
                        training_monitor.monitor_gradients(before_clip=True)
            
            # clip gradients
            grad_norm = fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
            if grad_norm is not None:
                training_monitor.log_scalars({"grad_norm": grad_norm})

            # monitor gradients and model parameters
            if fabric.world_size == 1:
                training_monitor.monitor_gradients()
                training_monitor.monitor_parameters()
            elif fabric.global_rank == 0 and training_monitor.is_monitoring(): # FSDP
                if reference_model is not None:
                    with FullyShardedDataParallel.summon_full_params(model, with_grads=True, rank0_only=True):
                        with FullyShardedDataParallel.summon_full_params(reference_model, rank0_only=True):                    
                            training_monitor.monitor_gradients()
                            training_monitor.monitor_parameters()
                else:
                    with FullyShardedDataParallel.summon_full_params(model, with_grads=True, rank0_only=True): 
                        training_monitor.monitor_gradients()
                        training_monitor.monitor_parameters()

            training_monitor.aggregate_step()

            # if we are monitoring the next step with reference model type "previous_step", we need to copy the current model weights into the reference model
            if training_monitor.is_step_monitored(training_monitor.step+1) and reference_model is not None and reference_model_type == "previous_step":
                if isinstance(fabric.strategy, FSDPStrategy):
                    with FullyShardedDataParallel.summon_full_params(model):
                        with FullyShardedDataParallel.summon_full_params(reference_model):
                            reference_model.load_state_dict(model.state_dict())
                else:
                    reference_model.load_state_dict(model.state_dict())

            # take the gradient step
            optimizer.step()
            optimizer.zero_grad()

            state["step_count"] += 1
            if use_pytorch_profiler:
                profiler.step()

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * model.max_seq_length,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size),
                "learning_rate": lr,
            }
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            training_monitor.log_scalars(metrics) 
            metrics.update(training_monitor.get_step_metrics())
            fabric.log_dict(metrics, step=state["step_count"])
            
        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
            with training_monitor.no_monitoring():
                t0 = time.perf_counter()
                val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters, loss_fn=loss_fn)
                val_loss = val_loss.item()
                if mse_loss:    # if we are using the mse loss, then we additionally validate with the cross-entropy loss
                    ce_val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters, loss_fn=chunked_cross_entropy)
                    ce_val_loss = ce_val_loss.item()
                td = time.perf_counter() - t0
            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss}
            if mse_loss:
                metrics["val_ce_loss"] = ce_val_loss
            else:
                metrics["val_ppl"] = math.exp(val_loss) # log validation perplexity only for training with cross-entropy loss
            training_monitor.log_scalars(metrics) 
            fabric.log_dict(metrics, step=state["step_count"])
            fabric.barrier()

            # auto-cancel
            if auto_cancel and val_loss > 18:
                fabric.print(f"Auto-canceling training due to high validation loss {val_loss:.4f}.")
                raise SystemExit(1)
 
        if (train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0) or (state["step_count"] == 0 and state["iter_num"] == 1):
            checkpoint_dir = out_dir / f"step-{state['step_count']:08d}" 
            save_checkpoint(fabric, state, tokenizer_dir, checkpoint_dir / "lit_model.pth")

            # save the current log_dict in the same directory as the checkpoint
            with open(checkpoint_dir / f"log_dict_rank{fabric.global_rank}.pkl", "wb") as f:
                pickle.dump(training_monitor.get_all_metrics(), f)

        # advance the training monitor to the gradient next step
        if not is_accumulating:
            training_monitor.start_step(state["step_count"]+1)

    if use_pytorch_profiler:
        profiler.stop()
        fabric.print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Final validation
    if eval.final_validation:
        with training_monitor.no_monitoring():
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters, loss_fn=loss_fn)
            if mse_loss:    # if we are using the mse loss, then we additionally validate with the cross-entropy loss
                ce_val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters, loss_fn=chunked_cross_entropy)
                ce_val_loss = ce_val_loss.item()
            metrics = {"val_loss": val_loss}
            if mse_loss:
                metrics["val_ce_loss"] = ce_val_loss
            else:
                metrics["val_ppl"] = math.exp(val_loss)
        training_monitor.log_scalars(metrics, force=True) 
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f}")

    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()


def mse_loss_fn(logits: torch.Tensor, targets: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Mean squared error loss on the logits."""
    logits = logits.reshape(-1, logits.size(-1))    # [B*S, V] contains the logits for each token
    targets = targets.reshape(-1)                   # [B*S]    contain the index of the target token
    # now convert the targets to one-hot encoding
    targets = torch.nn.functional.one_hot(targets, num_classes=logits.size(-1)).float()  # [B*S, V]
    # apply the softmax to the logits to get probabilities
    logits = torch.nn.functional.softmax(logits, dim=-1)  # [B*S, V]
    # now apply the MSE loss
    return torch.nn.functional.mse_loss(logits, targets, reduction="sum")   


def chunked_mse_loss_fn(logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 4096, *args, **kwargs) -> torch.Tensor:
    """Mean squared error loss on the logits with chunked computation."""
    # Reshape inputs
    logits = logits.reshape(-1, logits.size(-1))  # [B*S, V]
    targets = targets.reshape(-1)  # [B*S]
    
    # Split into chunks
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    
    loss_chunks = []
    
    for logit_chunk, target_chunk in zip(logit_chunks, target_chunks):
        # Convert targets to one-hot for this chunk
        targets_onehot = torch.nn.functional.one_hot(
            target_chunk, 
            num_classes=logit_chunk.size(-1)
        ).float()  # [chunk_size, V]
        
        # Apply softmax to logits for this chunk
        probs_chunk = torch.nn.functional.softmax(logit_chunk, dim=-1)  # [chunk_size, V]
        
        # Compute MSE loss for this chunk (reduction="none" to keep individual losses)
        chunk_loss = torch.nn.functional.mse_loss(
            probs_chunk, 
            targets_onehot, 
            reduction="none"
        )
        
        loss_chunks.append(chunk_loss)
    
    # Concatenate all loss chunks and compute mean
    all_losses = torch.cat(loss_chunks)  # This concatenates along the batch dimension
    return all_losses.sum()  # Returns a scalar tensor with grad_fn


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int, verbose: bool = True, loss_fn = chunked_cross_entropy) -> torch.Tensor:
    if max_iters == 0: # allow the user to skip validation
        return torch.tensor(42, device=fabric.device)

    fabric.barrier()
    if verbose:
        fabric.print("Validating ...")
    model.eval()

    losses = []
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = batch[:, 0 : model.max_seq_length].contiguous().long()
        targets = batch[:, 1 : (model.max_seq_length + 1)].contiguous().long()
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses.append(loss)

    val_loss = torch.stack(losses).mean()
    model.train()
    fabric.barrier()
    return val_loss


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs, block_size: int
) -> Tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float, step: int) -> float:
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


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

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


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        # this assumes we are calling pretrain.py as a script, but not how we are using it
        # save_hyperparameters(setup, checkpoint_file.parent)
        # if tokenizer_dir is not None:
        #     copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["max_steps", "epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))
