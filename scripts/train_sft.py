import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from configs.config import TrainConfig
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_original_load_rng_state = Trainer._load_rng_state

def _safe_load_rng_state(self, checkpoint):
    try:
        return _original_load_rng_state(self, checkpoint)
    except Exception as e:
        logger.warning(f"Skip loading RNG state from checkpoint due to: {e}")

Trainer._load_rng_state = _safe_load_rng_state

cfg = TrainConfig()

def build_sft_text(example, tokenizer):
    messages = []
    messages.extend(example["prompt"])
    messages.extend(example["completion"])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

def formatting_func(example):
    return build_sft_text(example, tokenizer_global)

def main():
    global tokenizer_global

    # Cho phép torch.load đọc checkpoint cũ
    torch.serialization.add_safe_globals([
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
    ])

    ds = load_from_disk(cfg.sft_data_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    tokenizer_global = tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    args = SFTConfig(
        output_dir=cfg.sft_output_dir,
        num_train_epochs=cfg.sft_epochs,
        learning_rate=cfg.sft_lr,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        max_seq_length=cfg.max_length,
        bf16=cfg.use_bf16,
        fp16=cfg.use_fp16,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )

    last_checkpoint = None
    if os.path.isdir(cfg.sft_output_dir):
        last_checkpoint = get_last_checkpoint(cfg.sft_output_dir)

    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found. Training from scratch.")
        trainer.train()

    trainer.save_model(cfg.sft_output_dir)
    tokenizer.save_pretrained(cfg.sft_output_dir)

    print(f"Saved SFT adapter to: {cfg.sft_output_dir}")

if __name__ == "__main__":
    main()