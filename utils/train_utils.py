# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List
import yaml
import time

import fire
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from torch.nn import functional as F

from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
from accelerate import skip_first_batches
from pkg_resources import packaging
from .memory_utils import MemoryTrace
import model_checkpointing
import torch.cuda.nccl as nccl
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from policies import bfSixteen, fpSixteen,bfSixteen_mixed, get_llama_wrapper

# TB_LOG_DIR="/opt/ml/output/tensorboard"

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, resume_epoch=0, resume_step=0):
    """
    Trains the model on the given dataloader
    
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
    
    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler() 
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    # tensorboard_callback = torch.utils.tensorboard.writer.SummaryWriter(log_dir = TB_LOG_DIR)
    ckpt_config = []
    for epoch in range(train_config.num_epochs):
        if epoch < resume_epoch:
            print(f"skipping epoch {epoch}...resuming from epoch {resume_epoch} and step {resume_step+1}")
            continue
        epoch_start_time = time.perf_counter()
        epoch_iterator = train_dataloader
        print(f"Instantiated new dataloader of len {len(epoch_iterator)}")
        print(f"Skipping {resume_step + 1} batches in dataloader")
        epoch_iterator = skip_first_batches(epoch_iterator, resume_step + 1)
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            # if train_config.enable_fsdp:
            #     print(f"Before training - Rank - {rank} - Max CUDA memory allocated was {memtrace.peak} GB")
            #     print(f"Before training - Rank - {rank} - Max CUDA memory reserved was {memtrace.max_reserved} GB")
            #     print(f"Before training - Rank - {rank} - Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            #     print(f"Before training - Rank - {rank} - Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            #     print(f"Before training - Rank - {rank} - CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

            total_loss = 0.0
            for step, batch in enumerate(tqdm(epoch_iterator, colour="blue", desc=f"Training Epoch{epoch}", initial=resume_step + 1)):
                global_step = resume_step + step + 1
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')              
                loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (global_step + 1) % gradient_accumulation_steps == 0 or global_step == len(train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (global_step + 1) % gradient_accumulation_steps == 0 or global_step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                if train_config.enable_fsdp:
                    if rank==0:       
                        print(f"\n step {global_step} is completed and loss is {loss.detach().float()}")
                else:
                    print(f"\n step {global_step} is completed and loss is {loss.detach().float()}")
                checkpoint_start_time = time.perf_counter()
                if (global_step > 0 and global_step % train_config.checkpoint_steps == 0) or global_step == len(train_dataloader) - 1:
                    if train_config.enable_fsdp:
                        print("Waiting for all the processes to sync")
                        dist.barrier()
                    if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        print(" Saving the FSDP model checkpoints using FULL_STATE_DICT")
                        model_checkpointing.save_model_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch, step=global_step
                        )
                    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        print("=====================================================")
                        # not implemented
                        # model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config)
                        # if train_config.save_optimizer:
                        #     model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config,
                        #                                                          optim=optimizer)
                        #     print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                        #     print("=====================================================")

                    if train_config.save_optimizer:
                        model_checkpointing.save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch = epoch, step = global_step
                        )
                        print("Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                        print("=====================================================")
                    model_checkpointing.save_checkpoint_params(train_config, epoch, global_step)
                    ckpt_config.append({
                        "epoch": epoch,
                        "step": global_step
                    })
                    if rank == 0:
                        if len(ckpt_config) > train_config.save_last:
                            model_checkpointing.cleanup_checkpoints(train_config, ckpt_config)
                            ckpt_config = ckpt_config[1:]
                        print(f"checkpoints saved - {len(ckpt_config)} - {ckpt_config}")
                    if train_config.enable_fsdp:
                        print("Waiting for all the processes to sync")
                        dist.barrier()
                checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                checkpoint_times.append(checkpoint_end_time)
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)    
        resume_step = -1
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        
        if train_config.enable_fsdp:
            if rank==0:
                print(f"During training - Rank - {rank} - Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"During training - Rank - {rank} - Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"During training - Rank - {rank} - Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"During training - Rank - {rank} - Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(f"During training - Rank - {rank} - CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        
        # Update the learning rate as needed
        lr_scheduler.step()
          
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer)
            checkpoint_start_time = time.perf_counter()
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    print("Waiting for all the processes to sync")
                    dist.barrier()
                if  fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    print(" Saving the FSDP model checkpoints using FULL_STATE_DICT")
                    model_checkpointing.save_model_checkpoint(
                        model, optimizer, rank, train_config, epoch=epoch
                    )
                elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                    print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                    print("=====================================================")
                    # not implemented
                    # model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config)
                    # if train_config.save_optimizer:
                    #     model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                    #     print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                    #     print("=====================================================")

                if train_config.save_optimizer:
                        model_checkpointing.save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                        print("=====================================================")                     
                if train_config.enable_fsdp:
                    print("Waiting for all the processes to sync")
                    dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)
            print(f"Validation completed for epoch {epoch}")
        
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
    avg_epoch_time = sum(epoch_times)/ len(epoch_times) 
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times)   
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep) 
        avg_eval_loss = sum(val_loss)/len(val_loss) 

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)
        
    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch")):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )
    
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)
    
    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")
        
    return eval_ppl, eval_epoch_loss

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")
                
                
def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries, 
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")