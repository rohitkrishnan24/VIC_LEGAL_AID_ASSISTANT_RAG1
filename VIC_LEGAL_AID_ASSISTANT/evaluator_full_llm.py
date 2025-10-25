import pandas as pd
from pathlib import Path
from rag_shared import retriever, get_chain, format_docs
import numpy as np
import json
import re

# Prefer the 100-query file if present; else fallback to template
IN_100 = Path("evaluation/evaluation_queries_100.csv")
IN_TEMPLATE = Path("evaluation/evaluation_queries_template.csv")
IN_PATH = IN_100 if IN_100.exists() else IN_TEMPLATE

OUT_CSV = Path("evaluation/results_advanced.csv")
SUMMARY_JSON = Path("evaluation/summary_advanced.json")

UNANSWERED_TOKEN = "I cannot provide an answer from the current legal documents."

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def unigram_overlap_ratio(answer: str, sources: str) -> float:
    a = set(normalize_text(answer).split())
    b = set(normalize_text(sources).split())
    return (len(a & b) / len(a)) if a else 0.0

def run(model_name="llama3.2", k=5):
    df = pd.read_csv(IN_PATH)
    for col in ["group", "label", "expected_hint"]:
        if col not in df.columns:
            df[col] = ""

    chain = get_chain(model_name)
    retriever.search_kwargs["k"] = int(k)

    rows = []
    for i, row in df.iterrows():
        q = str(row.get("query", "")).strip()
        expected_hint = str(row.get("expected_hint", "")).strip()
        group = str(row.get("group", "")).strip()
        label = str(row.get("label", "")).strip()

        docs = retriever.invoke(q)
        reviews, titles = format_docs(docs)
        ans = chain.invoke({"reviews": reviews, "question": q})

        unanswered = int(UNANSWERED_TOKEN.lower() in ans.lower())
        cited = int(any(t in ans for t in titles))
        retrieved_hit = int(expected_hint and any(expected_hint in t for t in titles))
        answer_mentions_expected = int(expected_hint and (expected_hint in ans))

        faithfulness = unigram_overlap_ratio(ans, reviews)
        answer_len = len(ans.split())

        rows.append({
            "query": q,
            "group": group,
            "label": label,
            "expected_hint": expected_hint,
            "answer": ans,
            "retrieved_titles": "; ".join(titles),
            "unanswered_flag": unanswered,
            "has_citation": cited,
            "retrieved_hit": retrieved_hit,
            "answer_mentions_expected": answer_mentions_expected,
            "faithfulness_overlap": faithfulness,
            "answer_length_tokens": answer_len
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(OUT_CSV, index=False)

    # Overall metrics
    summary = {
        "num_queries": int(len(out)),
        "unanswered_rate": float(out["unanswered_flag"].mean() if len(out) else 0.0),
        "citation_rate": float(out["has_citation"].mean() if len(out) else 0.0),
        "retrieved_hit_rate": float(out["retrieved_hit"].mean() if len(out) else 0.0),
        "expected_mentioned_rate": float(out["answer_mentions_expected"].mean() if len(out) else 0.0),
        "faithfulness_overlap_mean": float(out["faithfulness_overlap"].mean() if len(out) else 0.0),
        "answer_length_tokens_mean": float(out["answer_length_tokens"].mean() if len(out) else 0.0),
    }

    # Fairness by group
    fairness = {}
    if out["group"].astype(str).str.strip().replace("", np.nan).notna().any():
        grouped = out.groupby("group")
        fairness = {
            "by_group": {
                g: {
                    "count": int(len(d)),
                    "unanswered_rate": float(d["unanswered_flag"].mean()),
                    "citation_rate": float(d["has_citation"].mean()),
                    "faithfulness_overlap_mean": float(d["faithfulness_overlap"].mean()),
                }
                for g, d in grouped
            }
        }
        def disparity(metric):
            vals = [v[metric] for v in fairness["by_group"].values()]
            return float(max(vals) - min(vals)) if vals else 0.0
        fairness["disparity"] = {
            "unanswered_rate_gap": disparity("unanswered_rate"),
            "citation_rate_gap": disparity("citation_rate"),
            "faithfulness_gap": disparity("faithfulness_overlap_mean"),
        }

    # Topic breakdown by label
    by_label = {}
    if "label" in out.columns:
        grouped = out.groupby("label")
        by_label = {
            l: {
                "count": int(len(d)),
                "unanswered_rate": float(d["unanswered_flag"].mean()),
                "citation_rate": float(d["has_citation"].mean()),
                "faithfulness_overlap_mean": float(d["faithfulness_overlap"].mean()),
            }
            for l, d in grouped
        }

    SUMMARY_JSON.write_text(json.dumps({"overall": summary, "fairness": fairness, "by_label": by_label}, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_CSV} and {SUMMARY_JSON}")
    return out

if __name__ == "__main__":
    run()
