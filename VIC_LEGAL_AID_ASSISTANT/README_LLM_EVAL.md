# Full LLM Evaluation (Ollama)

## Prereqs
- Ollama running locally
- Models: `llama3.2` and `mxbai-embed-large`

## One-command run
```bash
bash run_all_llm.sh
```
This will:
1. Pull models (if needed)
2. Ensure vector DB is available (via `vector.py`)
3. Evaluate on **evaluation/evaluation_queries_100.csv** (100 queries, fairness-paired)
4. Generate plots and an HTML report
5. Package everything into `evaluation/LLM_Evaluation_Outputs.zip`

## Manual steps (if you prefer)
```bash
pip install -r requirements.txt
python evaluator_full_llm.py
python plots_llm.py
python postprocess_spans.py
python report_llm.py
```

## Outputs
- `evaluation/results_advanced.csv` — per query answers & metrics
- `evaluation/summary_advanced.json` — overall, fairness, topic metrics
- `evaluation/results_with_spans.csv` — evidence sentences from retrieved docs (span-level)
- `evaluation/plots_llm/*.png` — bar charts and histograms
- `evaluation/LLM_Evaluation_Report.html` — slide-ready report
- `evaluation/LLM_Evaluation_Outputs.zip` — everything bundled
