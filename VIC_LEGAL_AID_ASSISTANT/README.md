# Legal Assistance Chatbot (RAG) — Local Ollama

This project adapts the original restaurant-review RAG into a **Legal Assistance Chatbot** for Australia.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure [Ollama](https://ollama.com/) is running locally and pull models:
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

3. (Optional) Add more legal documents as plain text under `data/legal_docs/`.
   - Include a first line like `Title: Your Document Title` for better citations.

4. Run the app:
   ```bash
   python main.py
   ```

## How it Works

- `vector.py` builds a persistent Chroma vector store from `data/legal_docs/*.txt` using `mxbai-embed-large`.
- `main.py` retrieves the top-5 documents and prompts `llama3.2` to answer **only** from those docs.
- If the answer is not found, the bot will say it cannot answer from the current documents.

## Evaluation

A starter CSV is included: `evaluation/evaluation_queries_template.csv` with ~5 example queries.
You can extend it to 100 queries and compute metrics such as:
- % unanswered (effectiveness)
- Faithfulness/citation checks (manual or programmatic)
- Fairness across demographics (craft parallel queries and compare outcomes)

## Notes

- Be mindful: This is **not legal advice**. Add disclaimers if deploying publicly.
- For a clean rebuild of the vector DB, delete `./chroma_langchain_db_legal/` and run again.
