import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_cmd(cmd, cwd=None, env=None):
    print("\n[RUN]", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_lm_eval_json(path: Path):
    if not path.exists():
        return {}

    data = json.loads(path.read_text())
    results = data.get("results", {})

    out = {}
    for task, metrics in results.items():
        for k in ["acc,none", "exact_match,strict-match", "mc2,none", "acc"]:
            if k in metrics:
                out[task] = {"metric": k, "value": metrics[k]}
                break

    return out


def run_lm_eval(model_path, output_dir, tasks, limit, batch_size, device):
    out_file = output_dir / "lm_eval_results.json"

    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", batch_size,
        "--device", device,
        "--output_path", str(out_file),
        "--apply_chat_template",
    ]

    if limit:
        cmd += ["--limit", str(limit)]

    proc = run_cmd(cmd)

    save_text(output_dir / "lm_eval.stdout.txt", proc.stdout)
    save_text(output_dir / "lm_eval.stderr.txt", proc.stderr)

    return {
        "ok": proc.returncode == 0,
        "parsed": parse_lm_eval_json(out_file),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu", "gsm8k", "truthfulqa_mc2"])
    parser.add_argument("--lm_limit", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ",".join(args.benchmarks)

    results = {
        "started_at": datetime.utcnow().isoformat(),
        "lm_eval": run_lm_eval(
            model_path=args.finetuned_model,
            output_dir=out_dir,
            tasks=tasks,
            limit=args.lm_limit,
            batch_size="auto",
            device=args.device,
        ),
    }

    summary = results["lm_eval"]["parsed"]

    save_json(out_dir / "results.json", results)
    save_json(out_dir / "summary.json", summary)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()