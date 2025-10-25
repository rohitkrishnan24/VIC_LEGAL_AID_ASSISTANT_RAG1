import pandas as pd
from pathlib import Path
import json, re

DOC_DIR = Path("data/legal_docs")
IN_CSV = Path("evaluation/results_advanced.csv")
OUT_CSV = Path("evaluation/results_with_spans.csv")

def tokenize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]

def best_sentences(answer, doc_text, topn=2):
    a = set(tokenize(answer))
    sentences = re.split(r'(?<=[.!?])\s+', doc_text.strip())
    scored = []
    for s in sentences:
        stoks = set(tokenize(s))
        score = len(a & stoks)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for sc, s in scored[:topn] if sc > 0]

def main():
    df = pd.read_csv(IN_CSV)
    spans = []
    for i, row in df.iterrows():
        answer = str(row.get("answer", ""))
        titles = str(row.get("retrieved_titles", ""))
        retrieved = [t.strip() for t in titles.split(";") if t.strip()]
        evid = []
        for t in retrieved:
            # map title to file
            stem = None
            if "Legal Aid Overview" in t: stem = "legal_aid_overview.txt"
            elif "Family Violence Intervention Orders" in t: stem = "family_violence_intervention_orders.txt"
            elif "Tenancy Rights" in t: stem = "tenancy_rights_vic.txt"
            elif "Traffic Offences" in t: stem = "traffic_offences_vic.txt"
            elif "Unfair Dismissal" in t: stem = "employment_unfair_dismissal.txt"
            if stem:
                text = (DOC_DIR / stem).read_text(encoding="utf-8")
                top = best_sentences(answer, text, topn=2)
                if top:
                    evid.append({"title": t, "evidence": top})
        spans.append(json.dumps(evid, ensure_ascii=False))
    df["evidence_spans"] = spans
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
