from datasets import load_dataset, DatasetDict
from configs.config import TrainConfig

cfg= TrainConfig()

def clean_examples(ex):
    return {
        "prompt": ex["prompt"].strip(),
        "chosen": ex["chosen"].strip(),
        "rejected": ex["rejected"].strip(),
    }

def is_valid(ex):
    return (
        len(ex["prompt"]) > 0 and
        len(ex["chosen"]) > 0 and
        len(ex["rejected"]) > 0 and
        ex["chosen"] != ex["rejected"]
    )

def to_sft(ex):
    return {
        "prompt": [{"role": "user", "content": ex["prompt"]}],
        "completion": [{"role": "assistant", "content": ex["chosen"]}],
    }

def to_dpo(ex):
    return {
        "prompt": [{"role": "user", "content": ex["prompt"]}],
        "chosen": [{"role": "assistant", "content": ex["chosen"]}],
        "rejected": [{"role": "assistant", "content": ex["rejected"]}],
    }
def main():
    ds = load_dataset(cfg.dataset_name)["train"]
    ds = ds.map(clean_examples)
    ds = ds.filter(is_valid)

    split_1 = ds.train_test_split(test_size=0.1, seed=cfg.seed)
    split_2 = split_1["test"].train_test_split(test_size=0.50, seed=cfg.seed)

    split_ds = DatasetDict({
        "train": split_1["train"],
        "validation": split_2["train"],
        "test": split_2["test"],
    })
    sft_ds = DatasetDict({
        k: v.map(to_sft, remove_columns=v.column_names)
        for k, v in split_ds.items()
    })
    dpo_ds = DatasetDict({
        k: v.map(to_dpo, remove_columns=v.column_names)
        for k, v in split_ds.items()
    })
    sft_ds.save_to_disk(cfg.sft_data_dir)
    dpo_ds.save_to_disk(cfg.dpo_data_dir)

    print(split_ds)
    print("Saved SFT dataset to:", cfg.sft_data_dir)
    print("Saved DPO dataset to:", cfg.dpo_data_dir)
if __name__ == "__main__":
    main()



