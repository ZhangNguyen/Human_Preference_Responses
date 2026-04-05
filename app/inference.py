import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configs.config import TrainConfig

cfg = TrainConfig()


class ChatModel:
    def __init__(self, model_dir: str, model_name: str = "model"):
        self.model_dir = model_dir
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map="auto",
            torch_dtype=dtype,
        )
        self.model.eval()

    def _build_messages(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ):
        history = history or []
        messages = []

        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_fallback_prompt(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        history = history or []
        parts = []

        for user_msg, bot_msg in history:
            parts.append(f"User: {user_msg}")
            parts.append(f"Assistant: {bot_msg}")

        parts.append(f"User: {prompt}")
        parts.append("Assistant:")

        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        messages = self._build_messages(prompt, history)

        # Nếu tokenizer có chat_template thì dùng
        if getattr(self.tokenizer, "chat_template", None):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            # Fallback nếu không có chat_template
            fallback_prompt = self._build_fallback_prompt(prompt, history)
            inputs = self.tokenizer(
                fallback_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_length,
            ).input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = outputs[0][inputs.shape[-1]:]
        assistant_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return assistant_text


class DualModelService:
    def __init__(
        self,
        base_model_dir: Optional[str] = None,
        finetuned_model_dir: Optional[str] = None,
    ):
        self.base_model_dir = base_model_dir or cfg.base_model
        self.finetuned_model_dir = finetuned_model_dir or cfg.final_model_dir

        self.base_model = ChatModel(
            model_dir=self.base_model_dir,
            model_name="base_model",
        )
        self.finetuned_model = ChatModel(
            model_dir=self.finetuned_model_dir,
            model_name="finetuned_model",
        )

    def _run_single(
        self,
        model: ChatModel,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            answer = model.generate(prompt, history)
            latency = round(time.perf_counter() - start, 3)
            return {
                "model_name": model.model_name,
                "answer": answer,
                "latency": latency,
                "error": None,
            }
        except Exception as e:
            latency = round(time.perf_counter() - start, 3)
            return {
                "model_name": model.model_name,
                "answer": "",
                "latency": latency,
                "error": str(e),
            }

    def ask(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_base = executor.submit(self._run_single, self.base_model, prompt, history)
            future_finetuned = executor.submit(self._run_single, self.finetuned_model, prompt, history)

            base_result = future_base.result()
            finetuned_result = future_finetuned.result()

        return {
            "question": prompt,
            "base_model": base_result,
            "finetuned_model": finetuned_result,
        }