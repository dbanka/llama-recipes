# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import os
import json
import torch
import time

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg):
    # torch.manual_seed(103)
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
    )

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    if rank == 0:
         print(f"loading model from model path: {load_dir} ")
    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg,optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )


def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
    step=-1
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        if step >= 0:
            save_name = cfg.model_name + "-" + str(epoch) +"-"+str(step) + ".pt"
        else:
            save_name = cfg.model_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")

def load_model_checkpoint(model, rank, epoch, step, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    # where is the checkpoint at...
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        )
    load_dir = Path.cwd() / folder_name
    file_name = cfg.model_name + "-" + str(epoch) +"-"+str(step) + ".pt"
    load_full_path = load_dir / file_name
    # is it present...
    if not load_full_path.is_file():
        print(f"model checkpoint not found - {load_full_path} ")
        return False
    
    if rank == 0:
        model_checkpoint = torch.load(load_full_path)
        # integrate into loaded model
        model.load_state_dict(model_checkpoint)
        print(f"model checkpoint loaded to rank0 cpu")
    else:
        print(f"bypass on rank {rank}")
    return True


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1, step = -1):
    """save optimizer state via full state dict"""

   
    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if step >= 0:
            opt_save_name = (
                    "optimizer" + "-" + cfg.model_name + "-" + str(epoch)+"-"+ str(step) + ".pt"
            )
        else:
            opt_save_name = (
                    "optimizer" + "-" + cfg.model_name + "-" + str(epoch)+ ".pt"
            )
    
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")

def load_optimizer_checkpoint(model, rank, epoch, step, cfg):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        )
    load_dir = Path.cwd() / folder_name
    file_name = (
            "optimizer" + "-" + cfg.model_name + "-" + str(epoch)+"-"+ str(step) + ".pt"
    )
    load_full_path = load_dir / file_name

    if not load_full_path.is_file():
        raise Exception(f"optimizer checkpoint not found {load_full_path}")
        

    full_osd = None
    if rank == 0:
        full_osd = torch.load(load_full_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    print(f"optimizer shard loaded on rank {rank}")

    return sharded_osd

def load_sharded_model_single_gpu(model,model_path):
    
    reader = FileSystemReader(model_path)
    
    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )
    
    model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model

def save_checkpoint_params(cfg, epoch, step):
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        )
    save_dir = Path.cwd() / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    params_save_full_path = save_dir / "ckpt_params.json"
    params = {
        "last_epoch": epoch,
        "last_step": step,
    }
    print("Saving checkpoint params...")
    with open(params_save_full_path, "w") as f:
        json.dump(params, f)


def load_checkpoint_params(cfg):
    resume_epoch = 0
    resume_step = -1
    full_ckpt_params_path = (
        Path.cwd() / cfg.dist_checkpoint_root_folder / cfg.dist_checkpoint_folder / "ckpt_params.json"
    )
    if not full_ckpt_params_path.is_file():
        print(f"checkpoint params not found - {full_ckpt_params_path}")
        return resume_epoch, resume_step
    
    with open(full_ckpt_params_path, "r") as f:
        params = json.load(f)
    resume_epoch = params["last_epoch"]
    resume_step = params["last_step"]
    print(f"Resuming training from epoch {resume_epoch} and step {resume_step + 1}")
    return resume_epoch, resume_step


def delete_file(file_name):
    try:
        os.remove(file_name)
        print(f"'{file_name}' has been deleted successfully.")
    except FileNotFoundError:
        print(f"'{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred while deleting '{file_name}': {e}")


def cleanup_checkpoints(cfg):
    print(f"cleaning up old checkpoints - {cfg[0]}")
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        )
        save_dir = Path.cwd() / folder_name
    
    model_save_name = cfg.model_name + "-" + str(cfg[0]["epoch"]) +"-"+str(cfg[0]["step"]) + ".pt"
    opt_save_name = "optimizer" + "-" + cfg.model_name + "-" + str(cfg[0]["epoch"])+"-"+ str(cfg[0]["step"]) + ".pt"
            
    delete_file(save_dir/model_save_name)
    delete_file(save_dir/opt_save_name)
    
