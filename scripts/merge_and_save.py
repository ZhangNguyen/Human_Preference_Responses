import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel
from configs.config import TrainConfig

cfg = TrainConfig()


def main():
    os.makedirs(cfg.final_model_dir, exist_ok=True)

    dpo_dir = cfg.target_output_dir if os.path.isdir(cfg.target_output_dir) else cfg.source_output_dir
    last_checkpoint = get_last_checkpoint(dpo_dir) if os.path.isdir(dpo_dir) else None
    model_path = last_checkpoint if last_checkpoint is not None else dpo_dir

    print("Merging from:", model_path)

    # 1) Load tokenizer from checkpoint/adpater dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Load base model from original base model
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        device_map="auto",
    )

    # 3) Resize embeddings to match tokenizer saved in checkpoint
    base_model.resize_token_embeddings(len(tokenizer))

    # 4) Load adapter on top of resized base model
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
    )

    # 5) Merge and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(cfg.final_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(cfg.final_model_dir)

    print("Saved merged model to:", cfg.final_model_dir)


if __name__ == "__main__":
    main()