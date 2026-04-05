import shutil
from configs.config import TrainConfig

cfg = TrainConfig()

def main():
    zip_path = shutil.make_archive(
        base_name=cfg.final_model_dir,
        format="zip",
        root_dir=cfg.final_model_dir
    )
    print("Created zip:", zip_path)

if __name__ == "__main__":
    main()