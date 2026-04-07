import os
import torch


class TrainConfig:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # model dùng cho space này
        self.model_name = "finetuned_model"

        # base model gốc để tham chiếu nếu cần
        self.base_model = "Qwen/Qwen2.5-1.5B-Instruct"

        # model finetune thực tế sẽ load từ Hugging Face Hub
        # đổi đúng tên repo model finetune của bạn ở đây
        self.finetuned_model = "ZhangNguyen/ZhangNguyen_qwen2.5-1.5b-dpo-finetuned-HumanPreference"

        cuda_available = torch.cuda.is_available()
        bf16_supported = cuda_available and torch.cuda.is_bf16_supported()

        self.use_bf16 = bf16_supported
        self.use_fp16 = cuda_available and not bf16_supported

        self.max_length = 1024

        self.max_new_tokens = 64

        self.do_sample = True
        self.temperature = 0.4
        self.top_p = 0.9
        self.repetition_penalty = 1.05

        self.flask_host = "0.0.0.0"
        self.flask_port = 5000

    def to_dict(self):
        return self.__dict__