import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from configs.config import TrainConfig

cfg = TrainConfig()

def main():
    model_dir = cfg.dpo_output_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.eval()

    prompt = "What’s a hobby you think more people should try?"
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()