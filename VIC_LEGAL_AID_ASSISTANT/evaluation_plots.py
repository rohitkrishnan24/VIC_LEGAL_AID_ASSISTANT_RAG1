
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json

RES_CSV = Path("evaluation/results_advanced.csv")
SUM_JSON = Path("evaluation/summary_advanced.json")
PLOTS_DIR = Path("evaluation/plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

def main():
    df = pd.read_csv(RES_CSV)
    summary = json.loads(Path(SUM_JSON).read_text(encoding="utf-8"))

    # 1) Overall metrics bar
    metrics = {
        "Unanswered rate": summary["overall"]["unanswered_rate"],
        "Citation rate": summary["overall"]["citation_rate"],
        "Retrieved-hit rate": summary["overall"]["retrieved_hit_rate"],
        "Expected-mentioned rate": summary["overall"]["expected_mentioned_rate"],
        "Faithfulness overlap (mean)": summary["overall"]["faithfulness_overlap_mean"],
    }
    plt.figure()
    plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.title("Overall Metrics")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overall_metrics.png")
    plt.close()

    # 2) Answer length histogram
    plt.figure()
    df["answer_length_tokens"].hist(bins=20)
    plt.title("Answer Length Distribution (tokens)")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "answer_length_hist.png")
    plt.close()

    # 3) Group-wise metrics (if any group present)
    if "group" in df.columns and df["group"].astype(str).str.strip().replace("", None).notna().any():
        g = df.groupby("group")
        for metric, title in [
            ("unanswered_flag", "Unanswered Rate by Group"),
            ("has_citation", "Citation Rate by Group"),
            ("faithfulness_overlap", "Faithfulness (mean) by Group"),
        ]:
            plt.figure()
            vals = g[metric].mean()
            vals.plot(kind="bar")
            plt.title(title)
            plt.tight_layout()
            fname = f"{metric}_by_group.png"
            plt.savefig(PLOTS_DIR / fname)
            plt.close()

if __name__ == "__main__":
    main()
