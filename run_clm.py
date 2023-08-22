import os
import argparse

from transformers.trainer_utils import FSDPOption
from transformers import (
    set_seed,
    default_data_collator,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
import policies
from datasets import load_from_disk
import torch
from transformers import Trainer, TrainingArguments
import torch.distributed as dist
from configs import fsdp_config, train_config
from policies import AnyPrecisionAdamW

from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_dataset_config,
)
from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

from utils.dataset_utils import get_preprocessed_dataset
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def safe_save_model_for_hf_trainer(trainer: Trainer, tokenizer: LlamaTokenizer, output_dir: str):
    """Helper method to save model for HF Trainer."""
    # see: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    model = trainer.model
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()

    trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
    tokenizer.save_pretrained(output_dir)


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model id to use for training.",
    )
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get('SM_MODEL_DIR'),
                        help="Directory to save model files.")
    parser.add_argument("--dataset_path", type=str, default="lm_dataset", help="Path to dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--max_steps", type=int, default=None, help="Number of epochs to train for.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate to use for training.")
    parser.add_argument("--optimizer", type=str, default="adamw_hf", help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument("--fsdp", type=str, default=True, help="Whether to use fsdp.")
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        default=None,
        help="Which transformer layer to wrap with fsdp.",
    )

    args = parser.parse_known_args()
    print("argument parsed!!")
    return args


def training_function(train_config, fsdp_config, args):
    print("entered training function!!!")
    # set seed
    set_seed(args.seed)

    tokenizer = LlamaTokenizer.from_pretrained(train_config.tokenizer_name)
    print("loaded tokenizer!!!")
    dataset_config = generate_dataset_config(train_config, {})
    print("generated dataset config!!!")
     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    print("dataset loaded!!!")

    # dataset = load_from_disk(args.dataset_path)
    # load model from the hub
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_path,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    print("model loaded!!!")

    # if train_config.enable_fsdp and train_config.use_fast_kernels:
    #     """
    #     For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
    #     using of Flash Attention or Xformer memory-efficient kernels
    #     based on the hardware being used. This would speed up fine-tuning.
    #     """
    #     try:
    #         from optimum.bettertransformer import BetterTransformer
    #         model = BetterTransformer.transform(model)
    #     except ImportError:
    #         print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size

    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if fsdp_config.fsdp_activation_checkpointing:
        policies.apply_fsdp_checkpointing(model)
    print("setting activation checkpointing!!!")


    fsdp_config_dic = {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
    }

    print("setting training arguments!!!")

    # Define training args
    output_dir = train_config.checkpoint_folder
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=train_config.micro_batch_size,
        bf16=fsdp_config.pure_bf16,  # Use BF16 if available
        learning_rate=train_config.lr,
        num_train_epochs=train_config.num_epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=train_config.checkpoint_steps,
        optim="adamw_anyprecision",
        optim_args="use_kahan_summation=False,momentum_dtype:bfloat16,variance_dtype:bfloat16",
        ddp_timeout=7200,
        fsdp=[FSDPOption.FULL_SHARD,FSDPOption.AUTO_WRAP],
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=2,
        bf16_full_eval=True,
        resume_from_checkpoint=None,
        fsdp_config=fsdp_config_dic
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=default_data_collator,
    )
    print("starting trainer!!!")


    # Start training
    trainer.train()

    print("Training done!")

    # save model and tokenizer for easy inference
    safe_save_model_for_hf_trainer(trainer, tokenizer, args.model_dir)
    dist.barrier()


def main():
    args, _ = parse_arge()
    training_function(train_config, fsdp_config, args)


if __name__ == "__main__":
    main()