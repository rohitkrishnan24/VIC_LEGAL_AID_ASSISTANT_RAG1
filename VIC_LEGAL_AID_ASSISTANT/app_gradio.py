import gradio as gr
from rag_shared import retriever, get_chain, format_docs

def answer(question, model_name="llama3.2", k=5):
    chain = get_chain(model_name)
    retriever.search_kwargs["k"] = int(k)
    docs = retriever.invoke(question)
    reviews, titles = format_docs(docs)
    result = chain.invoke({"reviews": reviews, "question": question})
    sources = "\n".join([f"- {t}" for t in titles])
    return result, sources

with gr.Blocks(title="Legal Assistance Chatbot (RAG)") as demo:
    gr.Markdown("# ⚖️ Legal Assistance Chatbot (RAG Prototype)")
    gr.Markdown("Answers are limited to the curated legal documents. This is not legal advice.")
    with gr.Row():
        question = gr.Textbox(label="Ask your legal question", lines=2)
    with gr.Row():
        model = gr.Textbox(label="Ollama model", value="llama3.2")
        k = gr.Slider(1, 10, value=5, step=1, label="Retrieved docs (k)")
    btn = gr.Button("Ask")
    answer_box = gr.Markdown(label="Answer")
    sources_box = gr.Markdown(label="Sources")

    btn.click(fn=answer, inputs=[question, model, k], outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()
