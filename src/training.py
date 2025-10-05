import argparse
import logging
import os
import warnings
from datetime import timedelta
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.config import WANDB_KEY
from src.utils import copy_files_to_folder, load_config, prepare_data

# Initialize logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# Login
load_dotenv()
wandb.login(key=WANDB_KEY)


# DDP Setup
def setup_ddp():
    deepspeed.init_distributed()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def get_rank():
    return dist.get_rank()


def is_main_process():
    return dist.get_rank() == 0


def barrier():
    dist.barrier()


# Datsets
def get_datasets(dataset_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    ds_dict = load_from_disk(dataset_name)
    train_df = ds_dict["train"].to_pandas()
    val_df = ds_dict["val"].to_pandas()

    train_dataset = prepare_data(train_df)
    val_dataset = prepare_data(val_df)
    return train_dataset, val_dataset


# Main function
def launch_sft_train(
    model_name, tokenizer_name, train_set, val_set, work_folder_path, job_id, config
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        use_fast=True,
    )

    sft_config = SFTConfig(
        output_dir=work_folder_path,
        deepspeed="ds_config.json",
        **config.get("run_config", {}).get("sft_args", {}),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print("Filter out too long examples")

    train_set = train_set.filter(
        lambda batch: [
            len(ids)
            <= config.get("run_config", {}).get("sft_args", {}).get("max_length", {})
            for ids in tokenizer(
                batch["prompt"] + batch["completion"], truncation=False
            )["input_ids"]
        ],
        batched=True,
        num_proc=8,
    )

    val_set = val_set.filter(
        lambda batch: [
            len(ids)
            <= config.get("run_config", {}).get("sft_args", {}).get("max_length", {})
            for ids in tokenizer(
                batch["prompt"] + batch["completion"], truncation=False
            )["input_ids"]
        ],
        batched=True,
        num_proc=8,
    )

    barrier()
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_set,
        eval_dataset=val_set,
        formatting_func=None,
    )

    if is_main_process():
        log.info("Training Configuration:")
        log.info(f"- Model: {model_name}")
        log.info(f"- Dataset size: {len(train_set)}")
        log.info(f"- Per-device batch size: {sft_config.per_device_train_batch_size}")
        log.info(
            f"- Global batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}"
        )
        log.info(
            f"- Total training steps: {len(train_set) * sft_config.num_train_epochs // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps)}"
        )
        log.info(f"- Learning rate: {sft_config.learning_rate}")
        log.info(f"- Max sequence length: {sft_config.max_length}")
        log.info("\nStarting DeepSpeed fine-tuning...")

    trainer.train()

    barrier()
    if is_main_process():
        log.info("Saving model...")
        trainer.save_model(work_folder_path)
        tokenizer.save_pretrained(work_folder_path)
        log.info(f"Training completed! Model saved to {work_folder_path}")


def main():
    parser = argparse.ArgumentParser(description="SFT")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--job_id", required=True, type=int)
    args = parser.parse_args()

    work_folder_path = Path(f"workfolders/workfolder_{args.job_id}")
    work_folder_path.mkdir(parents=True, exist_ok=True)

    config = load_config("config.yaml")

    copy_files_to_folder(
        work_folder_path,
        "config.yaml",
        config["slurm_config"]["slurm_file_name"],
        "hostfile",
    )

    setup_ddp()

    train_set, val_set = get_datasets(args.dataset_name)
    launch_sft_train(
        args.model_name,
        args.tokenizer_name,
        train_set,
        val_set,
        str(work_folder_path),
        args.job_id,
        config,
    )


if __name__ == "__main__":
    main()
