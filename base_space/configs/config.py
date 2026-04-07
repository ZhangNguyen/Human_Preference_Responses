import torch


class TrainConfig:
    def __init__(self):
        self.model_name = "base_model"
        self.model_dir = "Qwen/Qwen2.5-1.5B-Instruct"

        cuda_available = torch.cuda.is_available()
        bf16_supported = cuda_available and torch.cuda.is_bf16_supported()

        self.use_bf16 = bf16_supported
        self.use_fp16 = cuda_available and not bf16_supported

        self.max_length = 1024
        self.max_new_tokens = 128
        self.do_sample = True
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.05

        self.flask_host = "0.0.0.0"
        self.flask_port = 5000

    def to_dict(self):
        return self.__dict__