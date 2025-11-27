import os
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from openai import OpenAI


class SimpleRAG:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.chat_history = []
        self.qa_chain = None
        self.candidate_name = None
        self.vectorstore = None

    # -------------------------
    # Load and split resume text
    # -------------------------
    def load_resume(self, resume_path):
        ext = os.path.splitext(resume_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(resume_path)
        else:
            loader = TextLoader(resume_path, encoding="utf-8")

        docs = loader.load()
        return docs

    # -------------------------
    # Extract candidate name
    # -------------------------
    def extract_name(self, resume_text: str):
        # Try a simple regex (first line likely to contain name)
        lines = resume_text.strip().splitlines()
        if lines:
            first_line = lines[0].strip()
            if len(first_line.split()) <= 4 and re.match(r"^[A-Za-z\s]+$", first_line):
                return first_line

        # If regex fails, ask GPT (fallback)
        try:
            client = OpenAI(api_key=self.openai_api_key)
            prompt = f"Extract the candidate's full name from this resume:\n\n{resume_text[:2000]}"
            response = client.responses.create(
                model="gpt-3.5-turbo",
                input=prompt
            )
            name = response.output_text.strip()
            if len(name.split()) <= 5:
                return name
        except Exception:
            pass

        return None

    # -------------------------
    # Create embeddings and retriever
    # -------------------------
    def create_vectorstore(self, docs):
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        self.vectorstore = vectorstore
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    # -------------------------
    # Build RAG chatbot
    # -------------------------
    def build_chatbot(self, resume_path):
        docs = self.load_resume(resume_path)
        full_text = " ".join([d.page_content for d in docs])
        self.candidate_name = self.extract_name(full_text)

        retriever = self.create_vectorstore(docs)

        prompt_template = """You are an AI assistant analyzing a candidate's resume.
If some information (like name, skills, or email) is missing, politely say it is not available.
If the name is mentioned, respond clearly with it.

Question: {question}
Context: {context}
Answer:"""

        prompt = PromptTemplate.from_template(prompt_template)

        llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever, combine_docs_chain_kwargs={"prompt": prompt}
        )

    # -------------------------
    # Ask chatbot a question
    # -------------------------
    def ask(self, query: str):
        if "name" in query.lower() and self.candidate_name:
            return f"The candidate's name is {self.candidate_name}."

        if "name" in query.lower() and not self.candidate_name:
            return "This information is not available in the resume."

        if not self.qa_chain:
            return "Please upload or load a resume first."

        result = self.qa_chain({
            "question": query,
            "chat_history": self.chat_history
        })

        answer = result["answer"]
        self.chat_history.append((query, answer))
        return answer


# -------------------------
# Example usage (for testing)
# -------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python chat_rag.py <OPENAI_API_KEY> <resume.pdf>")
        sys.exit(1)

    key = sys.argv[1]
    path = sys.argv[2]

    chatbot = SimpleRAG(openai_api_key=key)
    chatbot.build_chatbot(path)

    print("Chatbot ready! Ask your questions below (type 'exit' to stop).")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        print("Bot:", chatbot.ask(q))
