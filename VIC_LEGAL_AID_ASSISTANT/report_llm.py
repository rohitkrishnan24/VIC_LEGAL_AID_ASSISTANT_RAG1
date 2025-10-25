from pathlib import Path
import json
import pandas as pd

SUM_JSON = Path("evaluation/summary_advanced.json")
RES_CSV = Path("evaluation/results_advanced.csv")
PLOTS = Path("evaluation/plots_llm")
OUT = Path("evaluation/LLM_Evaluation_Report.html")

def img(path):
    return f'<img src="{path}" alt="{path}" style="max-width: 720px; width: 100%; margin: 12px 0;"/>'

def main():
    summary = json.loads(SUM_JSON.read_text(encoding="utf-8"))
    overall = summary.get("overall", {})
    fairness = summary.get("fairness", {})
    by_label = summary.get("by_label", {})

    html = [
        "<html><head><meta charset='utf-8'><title>LLM Evaluation Report</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif; padding:24px; line-height:1.5} h1,h2{margin:0.4em 0}</style>",
        "</head><body>",
        "<h1>LLM Evaluation Report</h1>",
        "<h2>Overall Metrics</h2>",
        "<pre>" + json.dumps(overall, indent=2) + "</pre>",
        img("plots_llm/overall_metrics_llm.png"),
        img("plots_llm/answer_length_hist_llm.png"),
    ]

    if fairness:
        html += [
            "<h2>Fairness (by Group)</h2>",
            "<pre>" + json.dumps(fairness, indent=2) + "</pre>",
            img("plots_llm/unanswered_flag_by_group_llm.png"),
            img("plots_llm/has_citation_by_group_llm.png"),
            img("plots_llm/faithfulness_overlap_by_group_llm.png"),
        ]

    if by_label:
        html += [
            "<h2>Topic Breakdown (by Label)</h2>",
            "<pre>" + json.dumps(by_label, indent=2) + "</pre>",
            img("plots_llm/unanswered_flag_by_topic_llm.png"),
            img("plots_llm/has_citation_by_topic_llm.png"),
            img("plots_llm/faithfulness_overlap_by_topic_llm.png"),
        ]

    html += ["</body></html>"]
    OUT.write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
