import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from peft import AutoPeftModelForCausalLM
from trl import DPOTrainer, DPOConfig
from configs.config import TrainConfig

cfg = TrainConfig()


def convert_dpo_example(example, tokenizer):
    prompt_text = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )
    chosen_text = example["chosen"][0]["content"]
    rejected_text = example["rejected"][0]["content"]
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def main():
    ds = load_from_disk(cfg.dpo_data_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = ds.map(lambda ex: convert_dpo_example(ex, tokenizer))

    # load checkpoint từ outputs cũ
    source_output_dir = cfg.source_output_dir

    # save checkpoint mới sang outputs2
    target_output_dir = cfg.target_output_dir

    last_checkpoint = None
    if os.path.isdir(source_output_dir):
        last_checkpoint = get_last_checkpoint(source_output_dir)

    model_path = last_checkpoint if last_checkpoint is not None else cfg.sft_output_dir
    print(f"Loading model from: {model_path}")
    print(f"Saving new checkpoints to: {target_output_dir}")

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        is_trainable=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
    )
    model.config.use_cache = False

    args = DPOConfig(
        output_dir=target_output_dir,
        num_train_epochs=cfg.dpo_epochs,
        learning_rate=cfg.dpo_lr,
        beta=cfg.dpo_beta,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=10,
        save_steps=60,
        eval_steps=60,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=2,
        max_length=cfg.max_length,
        truncation_mode="keep_start",
        bf16=cfg.use_bf16,
        fp16=cfg.use_fp16,
        gradient_checkpointing=True,
        precompute_ref_log_probs=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
    )

    if last_checkpoint is not None:
        print(f"Found source checkpoint: {last_checkpoint}")
        print("Continuing from checkpoint weights only; skipping resume_from_checkpoint.")
    else:
        print("No source checkpoint found. Training from SFT adapter.")

    trainer.train()

    trainer.save_model(target_output_dir)
    tokenizer.save_pretrained(target_output_dir)
    print(f"Saved DPO adapter to: {target_output_dir}")


if __name__ == "__main__":
    main()