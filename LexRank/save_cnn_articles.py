import os
import json
from datasets import load_dataset

def save_cnn_dailymail_subset(
        split: str = "train",
        n_samples: int = 5000,
        save_path: str = "data/cnn_500.jsonl"
):
    """
    Loads a subset of CNN/DailyMail and saves article + highlights to JSONL.
    """
    subset_spec = f"{split}[:{n_samples}]"
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=subset_spec)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for article in dataset:
            record = {
                "id": article["id"],
                "article": article["article"],
                "highlights": article["highlights"],
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Saved {len(dataset)} samples to {save_path}")


if __name__ == "__main__":
    save_cnn_dailymail_subset()
