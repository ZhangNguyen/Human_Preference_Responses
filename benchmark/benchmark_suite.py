import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_cmd(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    print("\n[RUN]", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_first_float(text: str, patterns: List[str]) -> Optional[float]:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def parse_lm_eval_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    out: Dict[str, Any] = {}
    for task, metrics in results.items():
        metric_candidates = [
            "acc,none",
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "mc2,none",
            "acc_norm,none",
            "acc",
        ]
        chosen = None
        for k in metric_candidates:
            if k in metrics:
                chosen = (k, metrics[k])
                break
        if chosen is None and metrics:
            first_key = next(iter(metrics.keys()))
            chosen = (first_key, metrics[first_key])
        if chosen is not None:
            out[task] = {"metric": chosen[0], "value": chosen[1]}
    return out


def build_vllm_model_args(model_path: str, tensor_parallel_size: int, max_model_len: int, gpu_memory_utilization: float) -> str:
    return (
        f"pretrained={model_path},tensor_parallel_size={tensor_parallel_size},"
        f"gpu_memory_utilization={gpu_memory_utilization},max_model_len={max_model_len},dtype=auto"
    )


def run_lm_eval(
    model_path: str,
    output_dir: Path,
    tasks: str,
    limit: Optional[int],
    batch_size: str,
    num_fewshot: Optional[int],
    device: str,
) -> Dict[str, Any]:
    out_file = output_dir / "lm_eval_results.json"
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path}",
        "--tasks",
        tasks,
        "--batch_size",
        batch_size,
        "--output_path",
        str(out_file),
        "--device",
        device,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]

    proc = run_cmd(cmd)
    save_text(output_dir / "lm_eval.stdout.txt", proc.stdout)
    save_text(output_dir / "lm_eval.stderr.txt", proc.stderr)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "parsed": parse_lm_eval_json(out_file),
        "raw_file": str(out_file),
    }


def run_alpaca_eval(
    model_path: str,
    output_dir: Path,
    openai_api_key: Optional[str],
) -> Dict[str, Any]:
    if not openai_api_key:
        return {
            "ok": False,
            "skipped": True,
            "reason": "OPENAI_API_KEY not set. AlpacaEval 2.0 needs an auto-annotator/judge.",
        }

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = openai_api_key

    # Let AlpacaEval call the local model through a thin wrapper script if needed later.
    # For simplicity and portability, this suite expects the user to first generate model outputs with their own script
    # or to expose the model via a local OpenAI-compatible endpoint. Here we only support the standard invocation path.
    return {
        "ok": False,
        "skipped": True,
        "reason": (
            "AlpacaEval 2.0 is kept in the suite because it is standard and popular, but integrating a local HF model "
            "requires either a model-specific generator wrapper or an OpenAI-compatible serving endpoint. "
            "Use the provided README steps to wire it in after you confirm your serving path."
        ),
    }


def run_mt_bench(
    model_path: str,
    output_dir: Path,
    openai_api_key: Optional[str],
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> Dict[str, Any]:
    if not openai_api_key:
        return {
            "ok": False,
            "skipped": True,
            "reason": "OPENAI_API_KEY not set. MT-Bench in FastChat uses an LLM judge such as GPT-4.",
        }

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = openai_api_key

    answers_file = output_dir / "mt_bench_model_answer.jsonl"
    judgment_dir = output_dir / "fastchat_judgment"
    judgment_dir.mkdir(parents=True, exist_ok=True)

    model_id = Path(model_path).name.replace(" ", "_")
    model_args = build_vllm_model_args(model_path, tensor_parallel_size, max_model_len, gpu_memory_utilization)

    gen_cmd = [
        sys.executable,
        "-m",
        "fastchat.llm_judge.gen_model_answer",
        "--model-path",
        model_path,
        "--model-id",
        model_id,
        "--answer-file",
        str(answers_file),
        "--num-gpus-total",
        str(tensor_parallel_size),
        "--max-new-token",
        "1024",
        "--temperature",
        "0.0",
    ]
    gen = run_cmd(gen_cmd, env=env)
    save_text(output_dir / "mt_bench_gen.stdout.txt", gen.stdout)
    save_text(output_dir / "mt_bench_gen.stderr.txt", gen.stderr)
    if gen.returncode != 0:
        return {"ok": False, "returncode": gen.returncode, "reason": "gen_model_answer failed"}

    judge_cmd = [
        sys.executable,
        "-m",
        "fastchat.llm_judge.gen_judgment",
        "--model-list",
        model_id,
        "--parallel",
        "1",
        "--mode",
        "single",
        "--judge-model",
        "gpt-4o-mini",
        "--answer-file",
        str(answers_file),
        "--output-dir",
        str(judgment_dir),
    ]
    judge = run_cmd(judge_cmd, env=env)
    save_text(output_dir / "mt_bench_judge.stdout.txt", judge.stdout)
    save_text(output_dir / "mt_bench_judge.stderr.txt", judge.stderr)

    parsed_score = extract_first_float(
        judge.stdout + "\n" + judge.stderr,
        [r"average\s+score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", r"score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
    )
    return {
        "ok": judge.returncode == 0,
        "returncode": judge.returncode,
        "metric": "mt_bench_score",
        "value": parsed_score,
        "answer_file": str(answers_file),
        "judgment_dir": str(judgment_dir),
    }


def build_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    lm = results.get("lm_eval", {}).get("parsed", {})
    for task_name, item in lm.items():
        summary[task_name] = item
    if "mt_bench" in results:
        mt = results["mt_bench"]
        if mt.get("value") is not None:
            summary["mt_bench"] = {"metric": mt.get("metric"), "value": mt.get("value")}
    if "alpaca_eval" in results and results["alpaca_eval"].get("value") is not None:
        ae = results["alpaca_eval"]
        summary["alpaca_eval"] = {"metric": ae.get("metric"), "value": ae.get("value")}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Objective benchmark suite for instruction/chat models.")
    parser.add_argument("--finetuned_model", required=True, help="Path to merged finetuned model")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--lm_limit", type=int, default=None, help="Optional sample cap for lm-eval tasks")
    parser.add_argument("--mmlu_fewshot", type=int, default=5)
    parser.add_argument("--gsm8k_fewshot", type=int, default=5)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["truthfulqa_mc2", "mmlu", "gsm8k", "mt_bench", "alpaca_eval"],
        choices=["truthfulqa_mc2", "mmlu", "gsm8k", "mt_bench", "alpaca_eval"],
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "args.json", vars(args))

    results: Dict[str, Any] = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "model": args.finetuned_model,
        "benchmarks_requested": args.benchmarks,
    }

    lm_tasks: List[str] = []
    if "truthfulqa_mc2" in args.benchmarks:
        lm_tasks.append("truthfulqa_mc2")
    if "mmlu" in args.benchmarks:
        lm_tasks.append("mmlu")
    if "gsm8k" in args.benchmarks:
        lm_tasks.append("gsm8k")

    if lm_tasks:
        results["lm_eval"] = run_lm_eval(
            model_path=args.finetuned_model,
            output_dir=out_dir,
            tasks=",".join(lm_tasks),
            limit=args.lm_limit,
            batch_size=args.batch_size,
            num_fewshot=None,
            device=args.device,
        )

    if "mt_bench" in args.benchmarks:
        results["mt_bench"] = run_mt_bench(
            model_path=args.finetuned_model,
            output_dir=out_dir,
            openai_api_key=args.openai_api_key,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    if "alpaca_eval" in args.benchmarks:
        results["alpaca_eval"] = run_alpaca_eval(
            model_path=args.finetuned_model,
            output_dir=out_dir,
            openai_api_key=args.openai_api_key,
        )

    results["summary"] = build_summary(results)
    results["finished_at"] = datetime.utcnow().isoformat() + "Z"
    save_json(out_dir / "results.full.json", results)
    save_json(out_dir / "summary.json", results["summary"])

    print("\n=== SUMMARY ===")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
