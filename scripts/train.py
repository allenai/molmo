"""Run this script with 'torchrun'."""

import logging
import sys
from os import listdir
from os.path import join
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from olmo.config import CheckpointType, TrainConfig
from olmo.data import build_train_dataloader
from olmo.eval import build_loss_evaluators, build_inf_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import Molmo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler, \
    build_multimodal_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
    freeze_parameters_by_name,
)
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")


def main(cfg: TrainConfig) -> None:
    if cfg.run_name is None:
        log_extra_field("run_name", cfg.run_name)

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

    if cfg.load_path is not None and cfg.model.low_cpu_fsdp:
        log.warning(
            "When loading a checkpoint to resume/finetune, the `low_cpu_fsdp` will be ignored."
        )
        cfg.model.low_cpu_fsdp = False
    
    if not cfg.data.multi_modal and cfg.save_dataloader_state:
        raise OLMoConfigurationError("You are saving the dataloader state, but the data is not multi-modal.")

    # Set CUDA device.
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    barrier()

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    # Display and save configuration.
    if get_global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)

    if cfg.allow_resume:
        config_path = Path(cfg.save_folder) / "config.yaml"
        if config_path.exists():
            lastest_checkpoint = Path(cfg.save_folder) / "latest"
            if lastest_checkpoint.exists():
                logging.info(f"Resuming from {lastest_checkpoint}")
                saved_config = TrainConfig.load(config_path)
                if saved_config.model != cfg.model:
                    logging.warning("Model config does not match the one resuming from")
                if saved_config.optimizer != cfg.optimizer:
                    logging.warning("Optimizer config does not match the one resuming from")
                if saved_config.data != cfg.data:
                    logging.warning("Data config does not match the one resuming from")
                cfg.load_path = str(lastest_checkpoint)
            else:
                logging.info("Not resuming since no latest checkpoint found")

    if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
        # Save config.
        save_path = Path(cfg.save_folder) / "config.yaml"
        if save_path.is_file() and not cfg.save_overwrite:
            raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
        else:
            log.info(f"Saving config to {save_path}")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            cfg.save(save_path)
        del save_path

    barrier()

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=cfg.asdict(exclude=["wandb"]),
        )

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = build_train_dataloader(cfg, device)

    # Construct evaluators.
    if cfg.eval_interval > 0 or cfg.eval_on_load:
        evaluators = build_loss_evaluators(cfg, device)
    else:
        evaluators = None
    inf_evaluators = build_inf_evaluators(cfg, device)
    barrier()

    # Initialize the model.
    olmo_model = Molmo(cfg.model)

    # Freeze model components.
    if cfg.model.vision_backbone is not None and not cfg.ft_connector:
        freeze_parameters_by_name(olmo_model, Molmo.get_connector_parameters(), warn=False)
    if cfg.model.vision_backbone is not None and not cfg.ft_vit:
        log.info(f"Freezing vision backbone")
        freeze_parameters_by_name(olmo_model, Molmo.get_vit_parameters(), warn=False)
    if not cfg.ft_llm:
        log.info(f"Freezing LLM")
        freeze_parameters_by_name(olmo_model, Molmo.get_llm_parameters(), warn=False)
    if cfg.ft_embedding != "all":
        if cfg.ft_embedding == "ln_f":
            log.info(f"Freezing LLM: wte.embedding, ff_out")
            freeze_names = ["transformer.wte.embedding", "transformer.wte.weight"]
            freeze_names += ["transformer.ff_out"]
        elif cfg.ft_embedding == "lm_head":
            log.info(f"Freezing LLM: wte.embedding")
            freeze_names = ["transformer.wte.embedding", "transformer.wte.weight"]
        else:
            assert cfg.ft_embedding == "wte"
            log.info(f"Freezing LLM: ln_f, ff_out")
            freeze_names = ["transformer.ln_f", "transformer.ff_out"]
        freeze_parameters_by_name(olmo_model, tuple(freeze_names), warn=False)

    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)

    listdir(cfg.save_folder)

    sync_module_states = True
    if cfg.load_path is None:
        # Sine we typically load some parameters from a pre-trained checkpoint, we init the rank0
        # model on the cpu and then use `sync_module_states` in FSDP to sync the parameters
        # with the rest of the devices
        init_weights = False
        if get_local_rank() == 0:
            if cfg.model.init_device == "meta":
                olmo_model.to_empty(device="cpu")
            if cfg.initial_model_checkpoint:
                state_dict = torch.load(join(cfg.initial_model_checkpoint, "model.pt"), map_location="cpu")
                olmo_model.load_state_dict(state_dict)
                del state_dict
            else:
                olmo_model.reset_with_pretrained_weights()
    else:
        init_weights = True

    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    if olmo_model.config.block_type == "moe":
        log.info(f"Number of active parameters: {olmo_model.num_params(include_inactive_params=False):,d}")    
    log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")

    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

    if init_weights or version.parse(torch.__version__) >= version.parse("2.1.0"):
        # Model is already initialized, so give FSDP a do-nothing init function
        # so it doesn't re-initialize the parameters
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=get_default_device(), recurse=False)

        param_init_fn = dummy_init_fn
    else:
        param_init_fn = None

    # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
    device_mesh = None
    hybrid_sharding_fsdp_kwargs = {}
    if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        if version.parse(torch.__version__) < version.parse("2.2.0"):
            # Device mesh was not added to PyTorch until v2.2.0
            raise OLMoConfigurationError(
                "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
            )

        from torch.distributed.device_mesh import init_device_mesh

        num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
            get_world_size() // get_local_world_size()
        )

        if num_model_replicas <= 0:
            raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

        if get_world_size() % num_model_replicas != 0:
            raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size")

        device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
        hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=cfg.fsdp_precision,
        auto_wrap_policy=wrap_policy,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
        sync_module_states=sync_module_states,
        param_init_fn=param_init_fn,
        **hybrid_sharding_fsdp_kwargs,
    )

    # This can prevent OOMs if loading a LLM checkpoint, presumably due to
    # reducing memory fragmentation
    torch.cuda.empty_cache()
    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(fsdp_model)

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, fsdp_model)
    if cfg.model.vision_backbone is not None:
        scheduler = build_multimodal_scheduler(cfg)
    else:
        scheduler = build_scheduler(cfg)

    # Consolidate components into `Trainer` object.
    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        fsdp_model=fsdp_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        inference_evaluators=inf_evaluators,
    ) as trainer:
        if not cfg.dry_run and not cfg.no_pre_train_checkpoint and cfg.load_path is None:
            checkpoint_type = (
                CheckpointType.sharded if cfg.save_num_checkpoints_to_keep != 0 else CheckpointType.unsharded
            )

            # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever).
            log.info("Saving pre-train checkpoint...")
            checkpoint_path, local_checkpoint_cache = trainer.save_checkpoint(checkpoint_type=checkpoint_type)
            log.info(f"Checkpoint saved to {checkpoint_path}")

            # And they we verify that we can load it.
            log.info("Attempting to load pre-train checkpoint...")
            trainer.restore_checkpoint(
                checkpoint_path,
                checkpoint_type=checkpoint_type,
                local_cache=local_checkpoint_cache,
                load_dataloader_state=False,
            )
            log.info("Checkpoint successfully loaded")

            # But now we can remove it so we don't take up unnecessary space.
            log.info("Removing pre-train checkpoint...")
            trainer.remove_checkpoint(checkpoint_type=checkpoint_type)
            log.info("Successfully removed checkpoint")

        if cfg.load_path is not None:
            log.info(f"Loading checkpoint from {cfg.load_path}...")
            trainer.restore_checkpoint(
                cfg.load_path,
                load_optimizer_state=not cfg.reset_optimizer_state,
                load_trainer_state=not cfg.reset_trainer_state,
                load_dataloader_state=cfg.save_dataloader_state and not cfg.reset_dataloader_state,
                sharded_checkpointer=cfg.load_path_sharded_checkpointer,
            )
            log.info("Checkpoint successfully loaded")

            # If we have to, set a new scheduler:
            if cfg.reset_optimizer_state and not cfg.reset_trainer_state:
                trainer.scheduler = BoltOnWarmupScheduler.wrap(
                    trainer.scheduler,
                    trainer.global_step,
                    int(trainer.global_step + cfg.scheduler.t_warmup),
                )

        if cfg.force_save_unsharded:
            log.info("Saving unsharded checkpoint...")
            checkpoint_path, _ = trainer.save_checkpoint(checkpoint_type=CheckpointType.unsharded)
            log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

        if cfg.compile is not None:
            # TODO (epwalsh): trying to compile the whole train step results in a compile-time error from within
            # the optimizer. We should investigate this further at some point.
            #  trainer.train_step = torch.compile(trainer.train_step, **cfg.compile.asdict())
            trainer.train_batch = torch.compile(trainer.train_batch, **cfg.compile.asdict())  # type: ignore
            # TODO (epwalsh): compiling the `eval_batch()` method is a little sketchy since the inputs will look
            # different for different eval tasks. That might be okay, but it might not be.
            #  trainer.eval_batch = torch.compile(trainer.eval_batch, **cfg.compile.asdict())  # type: ignore
            # Alternatively, could just do this:
            #  trainer.fsdp_model = torch.compile(trainer.fsdp_model, **cfg.compile.asdict())

        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Initialize process group.
    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
