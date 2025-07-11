import os
from typing import Dict, List, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings  # Optional, if needed
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from MultimodalPDFProcessor import MultimodalPDFProcessor
from VectorDBManager import VectorDBManager

google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAu9JXweyYbZxiDopgOdRsIIv1D4UkfuXM")  # Just for test

class ChatBot:
    """A chatbot that answers questions based on a multimodal PDF vector database."""
    
    def __init__(
        self,
        pdf_path: str,
        llm_model: str = "gemini-1.5-pro",
        embedding_model: str = "models/embedding-001",
        persist_directory: str = "vector_db"
    ):
        """
        Initialize the chatbot with a PDF processor and vector database manager.
        
        Args:
            pdf_path (str): Path to the PDF file.
            llm_model (str): Name of the Google Generative AI model.
            embedding_model (str): Name of the embedding model.
            persist_directory (str): Directory for persisting the vectorstore.
        """
        self.processor = MultimodalPDFProcessor(
            path=pdf_path,
            model=llm_model,
            embedding_model=embedding_model,
            persist_directory=persist_directory
        )
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=google_api_key
        )
        self.retriever = None
        self.persist_directory = persist_directory

    def initialize(self):
        """Initialize the retriever, loading existing or creating new."""
        self.retriever = self.processor.load_existing_retriever()
        if not self.retriever:
            self.processor.extract_content()
            self.retriever = self.processor.create_knowledge_retriever()

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user question and generate a response using the vector database.
        
        Args:
            question (str): The user's question.
            top_k (int): Number of top documents to retrieve.
        
        Returns:
            dict: Contains the answer and retrieved documents.
        """
        if not self.retriever:
            self.initialize()

        # Step 1: Retrieve relevant documents
        try:
            docs = self.retriever.get_relevant_documents(question)[:top_k]
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return {"answer": "Unable to retrieve documents.", "documents": []}

        # Step 2: Construct context from retrieved documents
        context_chunks = []
        for doc in docs:
            doc_type = doc.metadata.get("type", "text" if "text_as_html" not in doc.metadata else "table")
            content = doc.page_content
            if doc_type == "table":
                content = f"Table (HTML format):\n{doc.metadata.get('text_as_html', content)}"
            elif doc_type == "image":
                content = f"Image Description:\n{content}"
            else:
                content = f"Text:\n{content}"
            context_chunks.append(content)

        context = "\n\n".join(context_chunks)

        # Step 3: Construct prompt
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an expert in training planning. Below is a set of information (CONTEXT) extracted from documents, including text, image descriptions, and tables. Please use ONLY this information to accurately and comprehensively answer the user's question (QUESTION).
            
            ---
            CONTEXT:
            {context}
            ---
            QUESTION:
            {question}
            ---
            ANSWER:
            """
        )

        # Step 4: Generate response
        try:
            chain = prompt_template | self.llm
            response = chain.invoke({"question": question, "context": context})
            answer = response.content if hasattr(response, 'content') else str(response)
            return {
                "answer": answer,
                "documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "type": doc.metadata.get("type", "text" if "text_as_html" not in doc.metadata else "table")
                    }
                    for doc in docs
                ]
            }
        except Exception as e:
            print(f"Error during answer generation: {e}")
            return {"answer": "Unable to generate answer.", "documents": []}

    def chat(self):
        """Run an interactive chat session with the user."""
        print("Welcome to the PDF Chatbot! Type 'exit' to quit.")
        while True:
            question = input("Enter your question: ")
            if question.lower() == "exit":
                print("Goodbye!")
                break
            result = self.query(question)
            print("\nAnswer:", result["answer"])
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result["documents"], 1):
                print(f"\nDocument {i}:")
                print(f"Type: {doc['type']}")
                print(f"Content: {doc['content'][:200]}...")  # Truncate for readability
            print("\n" + "="*50 + "\n")

# Example usage
if __name__ == "__main__":
    chatbot = ChatBot(
        pdf_path="attention_research_paper.pdf",
        llm_model="gemini-1.5-pro",
        embedding_model="models/embedding-001",
        persist_directory="vector_db"
    )
    chatbot.chat()
