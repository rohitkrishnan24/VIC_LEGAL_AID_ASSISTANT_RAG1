from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a helpful, cautious Legal Assistance Chatbot for Australia.
Answer user questions using ONLY the retrieved legal documents provided below.
If the documents do not contain the answer, say: "I cannot provide an answer from the current legal documents."
Always be unbiased and clear. Where relevant, reference the document title.
 
Retrieved legal documents:
{reviews}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def format_docs(docs):
    formatted = []
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {})
        title = meta.get("title") or meta.get("source") or f"Doc {i}"
        formatted.append(f"- {title}\n{d.page_content}")
    return "\n\n".join(formatted)

if __name__ == "__main__":
    while True:
        print("\n\n-------------------------------")
        question = input("Ask your legal question (q to quit): ")
        print("\n\n")
        if question.strip().lower() == "q":
            break
        
        docs = retriever.invoke(question)
        reviews = format_docs(docs)
        result = chain.invoke({"reviews": reviews, "question": question})
        print(result)
