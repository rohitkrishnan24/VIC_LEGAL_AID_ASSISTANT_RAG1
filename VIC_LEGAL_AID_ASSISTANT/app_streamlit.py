import streamlit as st
from rag_shared import retriever, get_chain, format_docs

st.set_page_config(page_title="Legal Assistance Chatbot (RAG)", layout="centered")
st.title("⚖️ Legal Assistance Chatbot (RAG Prototype)")
st.caption("Answers are limited to the curated legal documents. This is not legal advice.")

model_name = st.sidebar.text_input("Ollama model", value="llama3.2")
k = st.sidebar.slider("Number of retrieved docs (k)", 1, 10, 5)

if "chain" not in st.session_state or st.session_state.get("model_name") != model_name:
    st.session_state["chain"] = get_chain(model_name)
    st.session_state["model_name"] = model_name

question = st.text_input("Ask your legal question")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating..."):
        # Temporarily adjust retriever k
        r = retriever
        r.search_kwargs["k"] = k
        docs = r.invoke(question)
        reviews, titles = format_docs(docs)
        result = st.session_state["chain"].invoke({"reviews": reviews, "question": question})
    st.subheader("Answer")
    st.write(result)
    with st.expander("Sources"):
        for t in titles:
            st.write(f"- {t}")
