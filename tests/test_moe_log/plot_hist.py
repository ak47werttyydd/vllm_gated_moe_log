import json
from collections import Counter
import matplotlib.pyplot as plt
import os

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    moe_log_path = os.getenv("VLLM_LOG_MOE") if os.getenv("VLLM_LOG_MOE","").strip() !="" else None
    assert moe_log_path is not None, "VLLM_LOG_MOE environment variable not set. Plotting reads jsonl from the path set by VLLM_LOG_MOE."
    cnt = Counter()
    total_routes = 0

    for obj in iter_jsonl(moe_log_path):
        t = obj.get("type", "")
        # skip meta/header lines
        if t in ("meta", "header"): 
            continue
        # only process route/record lines
        if t not in ("route", "record"):  
            continue
        topk_ids = obj.get("topk_ids", [])
        for eid in topk_ids:
            cnt[int(eid)] += 1
        total_routes += 1

    if not cnt:
        raise RuntimeError("No route records found in JSONL.")

    # sort by expert id
    experts = sorted(cnt.keys())
    values = [cnt[e] for e in experts]

    plt.figure()
    plt.bar(experts, values)
    plt.xlabel("expert_id")
    plt.ylabel("count (times selected in topk)")
    plt.title(f"MoE topk expert selection histogram (records={total_routes})")
    plt.tight_layout()
    plt.savefig("expert_hist.png", dpi=200)
    print("Saved expert_hist.png")

if __name__ == "__main__":
    main()
