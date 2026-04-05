import os
import torch


class TrainConfig:
    def __init__(self):
        # =========================
        # 1) PROJECT PATHS
        # =========================
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.data_dir = os.path.join(self.project_root, "data")
        self.models_saved_dir = os.path.join(self.project_root, "models_saved")
        self.outputs_dir = os.path.join(self.project_root, "outputs")
        self.outputs2_dir = os.path.join(self.project_root, "outputs2")

        # =========================
        # 2) DATASET
        # =========================
        # Đổi lại theo dataset thật của bạn nếu cần
        self.dataset_name = "json"

        # Nếu bạn dùng load_dataset từ HuggingFace Hub thì thay dataset_name
        # Ví dụ:
        # self.dataset_name = "my_username/my_dpo_dataset"

        self.sft_data_dir = os.path.join(self.data_dir, "sft_data")
        self.dpo_data_dir = os.path.join(self.data_dir, "dpo_data")

        # =========================
        # 3) MODEL PATHS
        # =========================
        # Model gốc để train / infer
        self.base_model = "Qwen/Qwen2.5-1.5B-Instruct"

        # Output SFT
        self.sft_output_dir = os.path.join(self.outputs_dir, "sft_adapter")

        # Nơi DPO đọc checkpoint cũ
        self.source_output_dir = self.sft_output_dir

        # Nơi DPO ghi checkpoint mới
        self.target_output_dir = os.path.join(self.outputs2_dir, "dpo_adapter")

        # Final merged model để inference
        self.final_model_dir = os.path.join(self.models_saved_dir, "final_merged_model")

        # =========================
        # 4) RANDOM SEED
        # =========================
        self.seed = 42

        # =========================
        # 5) PRECISION
        # =========================
        cuda_available = torch.cuda.is_available()
        bf16_supported = cuda_available and torch.cuda.is_bf16_supported()

        self.use_bf16 = bf16_supported
        self.use_fp16 = cuda_available and not bf16_supported

        # =========================
        # 6) TOKEN / SEQ LENGTH
        # =========================
        self.max_length = 1024

        # =========================
        # 7) LORA
        # =========================
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05

        # =========================
        # 8) SFT TRAINING
        # =========================
        self.sft_epochs = 1
        self.sft_lr = 2e-4
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 16

        # =========================
        # 9) DPO TRAINING
        # =========================
        self.dpo_epochs = 1
        self.dpo_lr = 5e-6
        self.dpo_beta = 0.1

        # =========================
        # 10) INFERENCE
        # =========================
        self.max_new_tokens = 128
        self.do_sample = True
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.05

        # =========================
        # 11) FLASK
        # =========================
        self.flask_host = "0.0.0.0"
        self.flask_port = 5000

    def to_dict(self):
        return self.__dict__