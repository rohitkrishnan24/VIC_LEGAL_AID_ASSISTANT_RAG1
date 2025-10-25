import pandas as pd
from pathlib import Path
from rag_shared import retriever, get_chain, format_docs

IN_PATH = Path("evaluation/evaluation_queries_template.csv")
OUT_CSV = Path("evaluation/results.csv")
SUMMARY_JSON = Path("evaluation/summary.json")

UNANSWERED_TOKEN = "I cannot provide an answer from the current legal documents."

def run(model_name="llama3.2", k=5):
    df = pd.read_csv(IN_PATH)
    chain = get_chain(model_name)
    retriever.search_kwargs["k"] = int(k)

    rows = []
    for i, row in df.iterrows():
        q = str(row.get("query", "")).strip()
        expected_hint = str(row.get("expected_hint", "")).strip()
        docs = retriever.invoke(q)
        reviews, titles = format_docs(docs)
        ans = chain.invoke({"reviews": reviews, "question": q})
        unanswered = int(UNANSWERED_TOKEN.lower() in ans.lower())
        cited = any(t in ans for t in titles)
        rows.append({
            "query": q,
            "expected_hint": expected_hint,
            "answer": ans,
            "retrieved_titles": "; ".join(titles),
            "unanswered_flag": unanswered,
            "has_citation": int(cited),
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(OUT_CSV, index=False)

    # Simple metrics
    n = len(out)
    unanswered_rate = out["unanswered_flag"].mean() if n else 0.0
    citation_rate = out["has_citation"].mean() if n else 0.0
    summary = {
        "num_queries": n,
        "unanswered_rate": unanswered_rate,
        "citation_rate": citation_rate,
        "notes": "Extend metrics to fairness and faithfulness as needed."
    }
    SUMMARY_JSON.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(f"Wrote {OUT_CSV} and {SUMMARY_JSON}")
    return out, summary

if __name__ == "__main__":
    run()
