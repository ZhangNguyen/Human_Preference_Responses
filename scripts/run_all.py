import os

steps = [
    "python scripts/prepared_data.py",
    "python scripts/train_sft.py",
    "python scripts/train_dpo.py",
    "python scripts/merge_and_save.py",
    "python scripts/zip_model.py",
]

for cmd in steps:
    print(f"Running: {cmd}")
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(f"Command failed: {cmd}")