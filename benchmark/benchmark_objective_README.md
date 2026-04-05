# Objective benchmark suite for your DPO model

This suite removes the custom `HeuristicJudge` idea and keeps only benchmarks that are well-known and widely used.

## Keep
- **MMLU** via EleutherAI `lm-evaluation-harness`
- **GSM8K** via EleutherAI `lm-evaluation-harness`
- **TruthfulQA MC2** via EleutherAI `lm-evaluation-harness`
- **MT-Bench** via LMSYS/FastChat with LLM-as-a-judge
- **AlpacaEval 2.0** kept as the standard pairwise / win-rate benchmark for instruction following

## Remove
- Custom heuristic judge
- Custom local pairwise score without a standard benchmark protocol
- TruthfulQA generation-mode judge for the default setup

## Why this set
- `lm-evaluation-harness` is a standard framework used widely in research and leaderboards.
- **MT-Bench** is a recognized chat benchmark from LMSYS / FastChat.
- **AlpacaEval 2.0** is a recognized automatic instruction-following benchmark with leaderboard and length-controlled win-rate.
- **TruthfulQA MC2** is more objective than open-ended generation scoring because it uses a fixed multiple-choice setup.

## Installation

```bash
pip install -U lm-eval transformers accelerate datasets
pip install -U "fschat[model_worker,webui]"
pip install -U alpaca-eval
```

If you want MT-Bench or AlpacaEval with an external judge model, set:

```bash
export OPENAI_API_KEY=your_key
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key"
```

## Basic usage

### Run the objective offline benchmarks first

```bash
python benchmark_suite.py \
  --finetuned_model models_saved/final_merged_model \
  --output_dir benchmark_outputs \
  --benchmarks truthfulqa_mc2 mmlu gsm8k \
  --lm_limit 100
```

### Add MT-Bench

```bash
python benchmark_suite.py \
  --finetuned_model models_saved/final_merged_model \
  --output_dir benchmark_outputs \
  --benchmarks truthfulqa_mc2 mmlu gsm8k mt_bench \
  --lm_limit 100
```

### Full suite

```bash
python benchmark_suite.py \
  --finetuned_model models_saved/final_merged_model \
  --output_dir benchmark_outputs \
  --benchmarks truthfulqa_mc2 mmlu gsm8k mt_bench alpaca_eval \
  --lm_limit 100
```

## Output files
- `args.json`
- `lm_eval_results.json`
- `summary.json`
- `results.full.json`
- `*.stdout.txt` / `*.stderr.txt`

## Important notes
1. **MMLU / GSM8K / TruthfulQA MC2** are the most objective and easiest to reproduce.
2. **MT-Bench** is popular, but still uses an LLM judge.
3. **AlpacaEval 2.0** is popular for pairwise / win-rate, but wiring a local HF model into AlpacaEval usually requires either:
   - a local model-serving endpoint, or
   - a custom generator wrapper.
4. For Kaggle, start with:
   - `truthfulqa_mc2`
   - `mmlu`
   - `gsm8k`

Then add MT-Bench only after the base script works.
