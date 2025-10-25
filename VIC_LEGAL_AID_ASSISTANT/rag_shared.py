from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

SYSTEM_PROMPT = """
You are a helpful, cautious Legal Assistance Chatbot for Australia.
Answer user questions using ONLY the retrieved legal documents provided below.
If the documents do not contain the answer, say: "I cannot provide an answer from the current legal documents."
Always be unbiased and clear. Where relevant, reference the document title as: Source: <Title>.
"""

TEMPLATE = SYSTEM_PROMPT + """

Retrieved legal documents:
{reviews}

Question: {question}
"""

def get_chain(model_name: str = "llama3.2"):
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    return prompt | model

def format_docs(docs):
    formatted = []
    titles = []
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        title = meta.get("title") or meta.get("source") or f"Doc {i}"
        titles.append(title)
        formatted.append(f"- {title}\n{d.page_content}")
    return "\n\n".join(formatted), titles

__all__ = ["retriever", "get_chain", "format_docs"]
