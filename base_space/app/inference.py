import time
from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from configs.config import TrainConfig

cfg = TrainConfig()


class ChatModel:
    def __init__(self, model_dir: str, model_name: str = "model"):
        self.model_dir = model_dir
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            use_fast=True,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()

    def _build_messages(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        history = history or []
        messages: List[Dict[str, str]] = []

        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, bot_msg = item
                messages.append({"role": "user", "content": str(user_msg)})
                messages.append({"role": "assistant", "content": str(bot_msg)})

        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_fallback_prompt(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        history = history or []
        parts: List[str] = []

        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, bot_msg = item
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

        if getattr(self.tokenizer, "chat_template", None):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
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
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


class SingleModelService:
    def __init__(self):
        self.model_dir = cfg.model_dir
        self.model_name = cfg.model_name

        self.chat_model = ChatModel(
            model_dir=self.model_dir,
            model_name=self.model_name,
        )

    def ask(
        self,
        prompt: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            answer = self.chat_model.generate(prompt, history)
            latency = round(time.perf_counter() - start, 3)
            return {
                "question": prompt,
                "model_name": self.model_name,
                "model_dir": self.model_dir,
                "answer": answer,
                "latency": latency,
                "error": None,
            }
        except Exception as e:
            latency = round(time.perf_counter() - start, 3)
            return {
                "question": prompt,
                "model_name": self.model_name,
                "model_dir": self.model_dir,
                "answer": "",
                "latency": latency,
                "error": str(e),
            }