import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt

RES_CSV = Path("evaluation/results_advanced.csv")
SUM_JSON = Path("evaluation/summary_advanced.json")
PLOTS_DIR = Path("evaluation/plots_llm")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

def main():
    df = pd.read_csv(RES_CSV)
    summary = json.loads(Path(SUM_JSON).read_text(encoding="utf-8"))

    # Overall metrics
    m = summary["overall"]
    plt.figure()
    keys = ["unanswered_rate", "citation_rate", "retrieved_hit_rate", "expected_mentioned_rate", "faithfulness_overlap_mean"]
    labels = ["Unanswered", "Citation", "Retrieved-hit", "Expected-mentioned", "Faithfulness (mean)"]
    vals = [m[k] for k in keys]
    plt.bar(labels, vals)
    plt.title("Overall Metrics — LLM")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overall_metrics_llm.png")
    plt.close()

    # Answer length
    plt.figure()
    df["answer_length_tokens"].hist(bins=20)
    plt.title("Answer Length Distribution (tokens) — LLM")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "answer_length_hist_llm.png")
    plt.close()

    # Group-wise (if present)
    if "group" in df.columns and df["group"].astype(str).str.strip().replace("", None).notna().any():
        g = df.groupby("group")
        for metric, title in [
            ("unanswered_flag", "Unanswered Rate by Group — LLM"),
            ("has_citation", "Citation Rate by Group — LLM"),
            ("faithfulness_overlap", "Faithfulness (mean) by Group — LLM"),
        ]:
            plt.figure()
            (g[metric].mean()).plot(kind="bar")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{metric}_by_group_llm.png")
            plt.close()

    # Topic-wise by label (if present)
    if "label" in df.columns and df["label"].astype(str).str.strip().replace("", None).notna().any():
        l = df.groupby("label")
        for metric, title in [
            ("unanswered_flag", "Unanswered Rate by Topic — LLM"),
            ("has_citation", "Citation Rate by Topic — LLM"),
            ("faithfulness_overlap", "Faithfulness (mean) by Topic — LLM"),
        ]:
            plt.figure()
            (l[metric].mean()).plot(kind="bar")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{metric}_by_topic_llm.png")
            plt.close()

if __name__ == "__main__":
    main()
