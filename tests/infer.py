from app.inference import ChatModel
from configs.config import TrainConfig

cfg = TrainConfig()

def main():
    model = ChatModel(cfg.final_model_dir, "finetuned_model")
    answer = model.generate("Xin chào, bạn là ai?")
    print("\n=== ANSWER ===")
    print(answer)

if __name__ == "__main__":
    main()