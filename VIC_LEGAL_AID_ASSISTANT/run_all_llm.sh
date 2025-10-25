#!/usr/bin/env bash
set -e

# Ensure models are available
echo "[INFO] Pulling Ollama models (if needed)"
ollama pull llama3.2 || true
ollama pull mxbai-embed-large || true

echo "[INFO] Installing Python requirements"
pip install -r requirements.txt

echo "[INFO] Building/refreshing vector DB (implicitly on import)"
python - <<'PY'
from vector import retriever
print("Retriever ready:", type(retriever))
PY

echo "[INFO] Running full LLM evaluation on 100 queries"
python evaluator_full_llm.py

echo "[INFO] Generating LLM plots"
python plots_llm.py

echo "[INFO] Post-processing span-level evidence"
python postprocess_spans.py

echo "[INFO] Building HTML report"
python report_llm.py

echo "[INFO] Packaging outputs"
python - <<'PY'
from pathlib import Path
import zipfile
out_zip = Path("evaluation/LLM_Evaluation_Outputs.zip")
if out_zip.exists(): out_zip.unlink()
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in Path("evaluation").glob("results_advanced.csv"): z.write(p, arcname=p.name)
    for p in Path("evaluation").glob("results_with_spans.csv"): z.write(p, arcname=p.name)
    for p in Path("evaluation").glob("summary_advanced.json"): z.write(p, arcname=p.name)
    for p in Path("evaluation/plots_llm").glob("*.png"): z.write(p, arcname=f"plots_llm/{p.name}")
    for p in Path("evaluation").glob("LLM_Evaluation_Report.html"): z.write(p, arcname=p.name)
print(out_zip)
PY

echo "[DONE] See evaluation/LLM_Evaluation_Outputs.zip"
