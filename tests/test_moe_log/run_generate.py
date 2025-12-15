import json, os, time
from pathlib import Path
from vllm import envs

SEED = 1234
#set environment variable LOG_MOE_SEED to SEED for reproducibility if not set
# if envs.LOG_MOE_SEED is None:
#     os.environ["LOG_MOE_SEED"] = str(SEED)
os.environ.setdefault("LOG_MOE_SEED", str(SEED))

PROMPT_SPLIT = "\n\n---\n\n"
cur_dir = Path(__file__).resolve().parent
prompt_path = cur_dir / "prompts.txt"
moe_log_path = cur_dir / "moe_log.jsonl"
timing_path = cur_dir / "timing.json"

from vllm import LLM, SamplingParams, envs

def load_prompts():
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().split(PROMPT_SPLIT)

def run_once(tag: str):
    prompts = load_prompts()

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        seed=SEED,
    )

    # fix max_model_len = 512
    llm = LLM(
        model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        max_model_len=512,
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    t1 = time.time()

    tokens_generated = sum(len(o.outputs[0].token_ids) for o in outs)

    return {
        "wall_time_sec": t1 - t0,
        "tokens_generated": tokens_generated,
        "num_prompts": len(prompts),
        "seed": SEED,
        "max_new_tokens": 128,
        "temperature": 0.0,
    }

def main():
    data = {}
    if os.path.exists(timing_path):
        with open(timing_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # log is on or off based on env var VLLM_LOG_MOE
    log_on =True if envs.VLLM_LOG_MOE is not None else False
    tag_log = "log" if log_on else "no_log"
    tag_timestamp = time.time()
    tag = f"{tag_log}_ts{int(tag_timestamp)}"

    data[tag] = run_once(tag)

    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Wrote", timing_path, "with keys:", list(data.keys()))

if __name__ == "__main__":
    main()