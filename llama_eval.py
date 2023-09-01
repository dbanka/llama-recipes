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
    get_policies, evaluation, evaluation_separate
)

from model_checkpointing.checkpoint_handler import (
    load_model_checkpoint,
    load_optimizer_checkpoint,
    load_checkpoint_params,
)

def main(**kwargs):
    rank = 0
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


    # Calculate gradient accumulation steps
    model_checkpoint_found = False

    # Load the pre-trained model and setup its configuration
    if train_config.resume_from_checkpoint:
        llama_config = LlamaConfig.from_pretrained(train_config.model_path)
        resume_epoch, resume_step = load_checkpoint_params(train_config)
        model = LlamaForCausalLM(llama_config)
        model_checkpoint_found = load_model_checkpoint(model, rank, resume_epoch, resume_step, train_config)


    if not model_checkpoint_found:
        print("Loading model")
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_path,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            )



    print_model_size(model, train_config, 0)
    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.tokenizer_name)
    tokenizer.add_special_tokens(
            {

                "pad_token": "<PAD>",
            }
        )

    model.to("cuda")

    ### gpu memory usage


    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    print(f"--> Validation Set Length = {len(dataset_val)}")


    # Create DataLoaders for the training and validation dataset

    eval_dataloader = StreamingDataLoader(
        dataset_val,
        batch_size=train_config.val_batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    eval_ppl, eval_epoch_loss = evaluation_separate(model,train_config, eval_dataloader, 0, tokenizer)
    print(f"Validation PPL: {eval_ppl}, Validation Loss: {eval_epoch_loss}")
    print(f"Validation completed!!!")

if __name__ == "__main__":
    fire.Fire(main)
