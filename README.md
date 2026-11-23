# Prompt-level defenses against indirect prompt injection in a small local LLM agent

Code, traces and analysis for the ICIMPACT 2027 paper "Balancing Security and Task Completion: Evaluating Prompt-Level Defenses against Indirect Prompt Injection in Tool-Using LLM Agents".

We ran Qwen3.5-4B (Ollama, `qwen3.5:4b`, Q4_K_M) as a tool-using agent on 400 records of NVIDIA's [Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1) dataset under eight defense conditions, 3,200 episodes in total.

| Condition | Defenses | ASR | Task progress |
|---|---|---|---|
| D0 | none | 42.0% | 81.0% |
| D1 | delimiters | 50.0% | 89.3% |
| D2 | priority rule | 28.3% | 76.5% |
| D3 | reminder | 9.8% | 81.5% |
| D5 | delimiters + rule | 44.3% | 79.8% |
| D6 | delimiters + reminder | 12.8% | 84.0% |
| D7 | rule + reminder | 7.8% | 77.8% |
| D4 | delimiters + rule + reminder | 15.0% | 81.0% |

## Contents

- `run_ipi.py` runs the agent episodes and writes one trace per record to `results/<condition>.jsonl`. It can be stopped and restarted.
- `score.py` scores the traces and writes `results/summary.md` and `results/metrics.csv`.
- `results/` holds all 3,200 traces (`D0.jsonl` to `D7.jsonl`), the 400 sampled record IDs (`sample_ids.json`) and the run settings.
- `analysis/` holds the statistics (paired McNemar tests with Holm correction, paired bootstrap intervals, GEE models), the paper's figures and the Table II rows.
- `data/` is where the dataset goes. It is not included here; download it from Hugging Face.

## Reproducing

```
pip install -r requirements.txt
hf download nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1 --repo-type dataset --local-dir data
```

To score and analyse the included traces:

```
python score.py --data data/train.jsonl
python analysis/analysis.py
python analysis/figs.py
python analysis/table2.py
```

To rerun the experiment, install Ollama, run `ollama pull qwen3.5:4b`, move or rename the `results` folder except `sample_ids.json`, and run:

```
python run_ipi.py --n 400 --conditions D0,D1,D2,D3,D4,D5,D6,D7
```

Settings: thinking off, temperature 0, seed 42, context 8,192 tokens, up to 1,024 output tokens per call, up to six model calls per episode. The paper used Ollama 0.34.2 on an RTX 4050 Laptop GPU (6 GB). GPU inference is not bit-for-bit reproducible, so a rerun may differ slightly.

## Trace format

Each line in `results/D*.jsonl` is one episode: record `id`, `condition`, `domain`, `attack_category`, `target_tool`, the list of tool `calls` with arguments, `injection_seen`, the model's `final_text`, turn and token counts, and `seconds`.

## License

The dataset is NVIDIA's, under CC BY 4.0. The injection texts that appear inside the traces come from that dataset.
