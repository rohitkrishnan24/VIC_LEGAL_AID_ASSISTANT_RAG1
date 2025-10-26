# ⚖️ RAG-Based Legal Assistance Chatbot (Victoria Legal FAQ Agent)

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

> 🧠 An explainable **Retrieval-Augmented Generation (RAG)** chatbot built to deliver *accurate, fair, and cited* legal information specific to **Victoria, Australia**.  
> This project extends the [Walert](https://github.com/rmit-ir/walert) benchmark into the *legal information domain*, ensuring factual grounding, fairness, and transparency.

---

## 📋 Overview

Many Australians struggle to access free, trustworthy legal information without technical or legal training.  
Our chatbot bridges this gap by combining **retrieval-based evidence** and **LLM reasoning** to generate *faithful, bias-aware, and jurisdiction-specific* responses.  

The system was developed as part of **RMIT’s Case Studies in Data Science (COSC2669)** WIL Project (Group 36).

---

## 🎯 Objectives

- ✅ Provide users with **accurate, cited legal information** from real Victorian legislation and public documents.  
- ⚖️ Ensure **fairness** — equal answer quality across linguistic or cultural phrasing.  
- 🧪 Implement a **quantitative evaluation harness** adapted from the [Walert](https://github.com/rmit-ir/walert) framework.  
- 🖥️ Build an **interactive Streamlit demo** for public legal self-help.

---

## 🏗️ Architecture

<img src="docs/ArchitectureDiagram-WIL.png" width="750">

### 🔹 Serving Pipeline
1. **Streamlit Interface** – Users input free-text legal queries.  
2. **Query Normaliser** – Cleans & standardises the query for embedding.  
3. **Retriever (ChromaDB + Ollama)** – Uses `mxbai-embed-large` embeddings with MMR ranking (`fetch_k=12 → top_k=8`).  
4. **LLM (LLaMA 3-32B via Ollama)** – Generates concise, cited answers grounded in retrieved context.  
5. **Post-processing** – Ensures brevity and adds citations to sources.

### 🔹 Knowledge Base & Indexing
- Curated Victorian legal documents (Tenancy, Traffic, Employment, etc.)  
- Chunked into ~800-character passages (150 overlap).  
- Embedded via **Ollama runtime** and stored in **Chroma vector DB**.

### 🔹 Evaluation Harness
- Test-driven framework (~120 test queries in CSV).  
- Measures **Answer Rate**, **Faithfulness**, **Fairness**, **Citation Accuracy**, and **Hallucination Rate**.  
- Generates automatic plots for transparency and reproducibility.

---

## ⚙️ Tech Stack

| Layer | Technology |
|:--|:--|
| Embeddings | `mxbai-embed-large` (Ollama) |
| Retriever | ChromaDB + LangChain |
| Generator | LLaMA 3-32B (local inference) |
| Frontend | Streamlit |
| Evaluation | Python, Pandas, Matplotlib |
| Deployment | Localhost (Ollama runtime) |

---

## 📊 Evaluation Framework

The evaluation framework is inspired by *Walert* and includes expanded diagnostic visualisations.

| Metric | Description |
|:--|:--|
| **Answer Rate** | % of queries successfully answered |
| **Faithfulness** | Semantic overlap between response and retrieved text |
| **Citation Correctness** | % of responses citing correct documents |
| **Retrieval Hit@k** | Proportion of correct docs within top-*k* retrievals |
| **Fairness Index** | Δ in performance between phrasing groups |
| **Hallucination Rate** | % of unfaithful or unsupported answers |

---

## 📈 Results Summary

| Metric | Score |
|:--|:--|
| Answer Rate | **86.7%** |
| Faithfulness | **0.82** |
| Citation Correctness | **91.4%** |
| Fairness Index (Δ) | **0.01** |
| Hallucination Rate | **4.8%** |

### 📉 Key Evaluation Plots
| Figure | Description |
|:--|:--|
| ![Overall Metrics](evaluation/eval_overall.png) | Overall evaluation metrics |
| ![Retrieval by Group](evaluation/retrieval_by_group.png) | Retrieval hit rate by user group |
| ![Faithfulness vs Answered](evaluation/faithfulness_vs_answered_scatter.png) | Group-wise faithfulness vs answered rate |
| ![Faithful vs Hallucinated](evaluation/faithful_pie.png) | Breakdown of faithful vs hallucinated answers |
| ![Group Comparison](evaluation/group_comparison.png) | Comparison of fairness metrics across groups |

### 🧩 Diagnostic Visualisations (LLM Output)
| Figure | Description |
|:--|:--|
| ![Overall LLM Metrics](evaluation/overall_metrics_llm.png) | LLM overall output metrics |
| ![Answer Length Distribution](evaluation/answer_length_hist_llm.png) | Token length distribution |
| ![Faithfulness by Group](evaluation/faithfulness_overlap_by_group_llm.png) | Group-level mean faithfulness |
| ![Citation Rate by Group](evaluation/has_citation_by_group_llm.png) | Citation accuracy by group |
| ![Unanswered Rate by Group](evaluation/unanswered_flag_by_group_llm.png) | Unanswered rate comparison |

---

## 📚 Repository Structure

