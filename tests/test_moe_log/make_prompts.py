from datasets import load_dataset
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
prompt_path = cur_dir / "prompts.txt"
ds = load_dataset("openai/gsm8k", "main", split="test")  # MIT
prompts = [ex["question"] for ex in ds.select(list(range(25)))]

open(prompt_path, "w", encoding="utf-8").write("\n\n---\n\n".join(prompts))
print(f"Wrote {prompt_path} with", len(prompts), "prompts")
