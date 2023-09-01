# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import pkg_resources
import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from streaming.base import StreamingDataLoader
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# from utils import MemoryTrace

import policies
from configs import fsdp_config, train_config
from policies import AnyPrecisionAdamW

from utils import fsdp_auto_wrap_policy

from utils.config_utils import (
    update_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset

from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies, evaluation
)

from model_checkpointing.checkpoint_handler import (
    load_model_checkpoint,
    load_optimizer_checkpoint,
    load_checkpoint_params,
)

def main(**kwargs):

    print("Checking package versions...")
    # Read the requirements.txt file and extract package names and version constraints
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()

    # Iterate through the requirements and check installed versions
    for requirement in requirements:
        try:
            req = pkg_resources.Requirement.parse(requirement)
            package = pkg_resources.get_distribution(req.project_name)
            print(f"{package.project_name}: Installed version {package.version}")
        except Exception:
            print(f"{req.project_name}: Not installed or parsing error")
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.gradient_accumulation_steps
    resume_epoch = 0
    resume_step = -1
    model_checkpoint_found = False
    llama_config = LlamaConfig.from_pretrained(train_config.model_path)

    # Load the pre-trained model and setup its configuration
    if train_config.resume_from_checkpoint:
        resume_epoch, resume_step = load_checkpoint_params(train_config)
        model = LlamaForCausalLM(llama_config)
        model_checkpoint_found = load_model_checkpoint(model, rank, resume_epoch, resume_step, train_config)

    if not model_checkpoint_found:
        resume_epoch = 0
        resume_step = -1
        print("Model will be trained from the beginning")

    if not model_checkpoint_found and train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
        else:
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    elif not model_checkpoint_found:
        if rank == 0:
            print("Loading model")

            model = LlamaForCausalLM.from_pretrained(
                train_config.model_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                )
        else:
            print("Loading model with config")
            model = LlamaForCausalLM(llama_config)




    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        print("**************Using fast kernels**************")
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.tokenizer_name)
    tokenizer.add_special_tokens(
            {

                "pad_token": "<PAD>",
            }
        )
    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)

        # with MemoryTrace() as memtrace:  # track the memory usage
        model = FSDP(
            model,
            auto_wrap_policy=  wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
        # print(f"During training - Rank - {rank} - Max CUDA memory allocated was {memtrace.peak} GB")
        # print(f"During training - Rank - {rank} - Max CUDA memory reserved was {memtrace.max_reserved} GB")
        # print(f"During training - Rank - {rank} - Peak active CUDA memory was {memtrace.peak_active_gb} GB")
        # print(f"During training - Rank - {rank} - Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        # print(f"During training - Rank - {rank} - CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    ### gpu memory usage



    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = None
    # dataset_val = get_preprocessed_dataset(
    #     tokenizer,
    #     dataset_config,
    #     split="test",
    # )
    # if not train_config.enable_fsdp or rank == 0:
    #         print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    # if train_config.enable_fsdp:
    #     train_sampler = DistributedSampler(
    #         dataset_train,
    #         rank=dist.get_rank(),
    #         num_replicas=dist.get_world_size(),
    #         shuffle=True,
    #     )
    #     if train_config.run_validation:
    #         val_sampler = DistributedSampler(
    #             dataset_val,
    #             rank=dist.get_rank(),
    #             num_replicas=dist.get_world_size(),
    #         )

    # Create DataLoaders for the training and validation dataset
    train_dataloader = StreamingDataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = StreamingDataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )
    else:
        eval_dataloader = None

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )

    eval_ppl, eval_epoch_loss = evaluation(model,train_config, eval_dataloader,  local_rank if train_config.enable_fsdp else None, tokenizer)
    if not train_config.enable_fsdp or rank==0:
        print(f"Validation PPL: {eval_ppl}, Validation Loss: {eval_epoch_loss}")
        print(f"Validation completed!!!")

if __name__ == "__main__":
    fire.Fire(main)
