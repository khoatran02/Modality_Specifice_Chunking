import os

from typing import List, Dict
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

# Import the provided classes
from MultimodalPDFProcessor import MultimodalPDFProcessor
from VectorDBManager import VectorDBManager

google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAu9JXweyYbZxiDopgOdRsIIv1D4UkfuXM")  # Just for test

class Chatbot:
    def __init__(
        self,
        pdf_path: str,
        model: str = "gemini-1.5-pro",
        embedding_model: str = "models/embedding-001",
        collection_name: str = "pdf_collection",
        persist_directory: str = "vector_db"
    ):
        """
        Initialize the chatbot with a PDF processor and retriever.
        Check if a vector DB exists; if not, create it; otherwise, load it.
        
        Args:
            pdf_path (str): Path to the PDF file to process.
            model (str): LLM model name (e.g., gemini-1.5-pro).
            embedding_model (str): Embedding model name.
            collection_name (str): Name of the vectorstore collection.
            persist_directory (str): Directory to persist the vectorstore.
        """
        self.model = model
        self.pdf_path = pdf_path  # Store pdf_path for processing
        self.pdf_processor = MultimodalPDFProcessor(
            path=pdf_path,
            model=model,
            embedding_model=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.db_manager = VectorDBManager(
            embedding_model=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=SecretStr(google_api_key),
            model=self.model
        )

        # Check if vector DB exists
        if self.db_manager.exists():
            print(f"Vector database found at {persist_directory}. Loading existing database...")
            self.retriever = self.db_manager.load_retriever()
        else:
            print(f"No vector database found at {persist_directory}. Creating new database...")
            self.pdf_processor.extract_content()
            self.retriever = self.pdf_processor.create_knowledge_retriever()

    def _construct_prompt(self, question: str, documents: List[Document]) -> str:
        """
        Construct a prompt by combining the user question with retrieved content.
        
        Args:
            question (str): The user's question.
            documents (List[Document]): Retrieved documents from the vectorstore.
        
        Returns:
            str: The constructed prompt.
        """
        prompt_template = """
        You are an expert assistant answering questions based on a provided PDF document.
        Below is the user's question and relevant information extracted from the PDF, including text, tables, and image descriptions.
        Use this context to provide an accurate, coherent, and concise answer. If the context is insufficient, state so clearly and provide the best possible response.

        **User Question:**
        {question}

        **Context:**
        {context}

        **Instructions:**
        - Analyze the context carefully to address the user's question.
        - If the context includes tables, interpret the table data (provided as text or HTML).
        - If the context includes image descriptions, use them to provide relevant details.
        - Do not invent information not present in the context.
        - Format the answer clearly and professionally.
        """
        
        # Build context from retrieved documents
        context_parts = []
        for doc in documents:
            content_type = doc.metadata.get('type', 'text')
            if content_type == 'image':
                context_parts.append(f"Image Description: {doc.page_content}")
            elif 'text_as_html' in doc.metadata:
                context_parts.append(f"Table (HTML): {doc.metadata['text_as_html']}\nTable Text: {doc.page_content}")
            else:
                context_parts.append(f"Text: {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Combine question and context into the prompt
        return prompt_template.format(question=question, context=context)
    
    def query(self, user_question: str, top_k: int = 5) -> str:
            """
            Process a user question and return the LLM-generated response.
            
            Args:
                user_question (str): The user's question.
                top_k (int): Number of relevant chunks to retrieve.
            
            Returns:
                str: The LLM-generated response.
            """
            # Step 1: Receive the query
            print(f"Received question: {user_question}")

            # Step 2: Retrieval - Perform similarity search
            retrieved_docs = self.retriever.vectorstore.similarity_search(user_question)[:top_k]

            print(f"Retrieved {len(retrieved_docs)} documents for the question.")
            
            # Step 3: Prompt Construction
            prompt = self._construct_prompt(user_question, retrieved_docs)
            print(f"Constructed prompt:\n{prompt}")

            # Step 4: Response Generation
            try:
                response = self.llm.invoke(prompt)
                response_text = getattr(response, 'content', str(response))
                print(f"Generated response: {response_text}")
                
                # Step 5: Reply to the user
                return response_text
            except Exception as e:
                print(f"Error during response generation: {e}")
                return "Sorry, an error occurred while processing your request."

# Example usage
if __name__ == "__main__":
    # Initialize the chatbot with a sample PDF
    chatbot = Chatbot(
        pdf_path="attention_research_paper.pdf",
        model="gemini-2.5-flash",  
        embedding_model="models/embedding-001",
        collection_name="multi_modal_rag",
        persist_directory="vector_db"
    )
    
    # Example user question
    user_question = "What is the attention?"
    response = chatbot.query(user_question)
    print(f"Chatbot Response: {response}")

