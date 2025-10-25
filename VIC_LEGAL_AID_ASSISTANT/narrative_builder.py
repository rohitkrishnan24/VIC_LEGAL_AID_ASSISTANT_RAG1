import json, pandas as pd
from pathlib import Path

SUM = Path("evaluation/summary_advanced.json")
RES = Path("evaluation/results_advanced.csv")
OUT_TXT = Path("evaluation/LLM_Evaluation_Summary.txt")

def pct(x): 
    return f"{100.0*float(x):.1f}%"

def main():
    overall = {}
    fairness = {}
    by_label = {}
    if SUM.exists():
        data = json.loads(SUM.read_text(encoding="utf-8"))
        overall = data.get("overall", {})
        fairness = data.get("fairness", {})
        by_label = data.get("by_label", {})
    df = pd.read_csv(RES) if RES.exists() else pd.DataFrame()

    lines = []
    lines.append("== LLM Evaluation Summary ==")
    if overall:
        lines.append("\n-- Overall --")
        lines.append(f"Queries: {int(overall.get('num_queries', 0))}")
        lines.append(f"Unanswered rate: {pct(overall.get('unanswered_rate', 0))}")
        lines.append(f"Citation rate: {pct(overall.get('citation_rate', 0))}")
        lines.append(f"Retrieved-hit rate: {pct(overall.get('retrieved_hit_rate', 0))}")
        lines.append(f"Expected-mentioned rate: {pct(overall.get('expected_mentioned_rate', 0))}")
        lines.append(f"Faithfulness (overlap mean): {overall.get('faithfulness_overlap_mean', 0):.3f}")
        lines.append(f"Answer length (avg tokens): {overall.get('answer_length_tokens_mean', 0):.1f}")

    if fairness:
        lines.append("\n-- Fairness --")
        by_group = fairness.get("by_group", {})
        for g, m in by_group.items():
            lines.append(f"[{g}] count={int(m.get('count',0))} | "
                         f"unanswered={pct(m.get('unanswered_rate',0))} | "
                         f"citation={pct(m.get('citation_rate',0))} | "
                         f"faithfulness_mean={m.get('faithfulness_overlap_mean',0):.3f}")
        disparity = fairness.get("disparity", {})
        if disparity:
            lines.append("Disparities (max - min across groups):")
            lines.append(f"- Unanswered gap: {disparity.get('unanswered_rate_gap',0):.3f}")
            lines.append(f"- Citation gap: {disparity.get('citation_rate_gap',0):.3f}")
            lines.append(f"- Faithfulness gap: {disparity.get('faithfulness_gap',0):.3f}")

    if by_label:
        lines.append("\n-- Topic Breakdown --")
        for l, m in by_label.items():
            lines.append(f"[{l}] n={int(m.get('count',0))} | "
                         f"unanswered={pct(m.get('unanswered_rate',0))} | "
                         f"citation={pct(m.get('citation_rate',0))} | "
                         f"faithfulness_mean={m.get('faithfulness_overlap_mean',0):.3f}")

    if not df.empty:
        lines.append("\n-- Suggested Talking Points --")
        # Simple heuristics
        if overall.get('unanswered_rate', 0) > 0.2:
            lines.append("* Focus next sprint on recall: expand KB, tune retriever (k, BM25 + dense).")
        else:
            lines.append("* Unanswered rate within acceptable range; maintain KB hygiene and prompts.")
        if overall.get('citation_rate', 0) < 0.8:
            lines.append("* Improve source-citation prompting and response post-processing.")
        if overall.get('faithfulness_overlap_mean', 0) < 0.35:
            lines.append("* Increase faithfulness: stricter grounding, re-rank by MMR, add 'no answer' bias.")
        lines.append("* Present fairness bars and discuss any group gaps and mitigations.")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_TXT}")

if __name__ == "__main__":
    main()
