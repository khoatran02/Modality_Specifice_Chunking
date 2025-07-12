import os
from typing import Optional
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAu9JXweyYbZxiDopgOdRsIIv1D4UkfuXM") # Just for test

class VectorDBManager:
    """Handles persistence and loading of multimodal vector databases"""
    
    def __init__(self, embedding_model: str,  collection_name: str, persist_directory: str = "vector_db"):
        
        self.collection_name=collection_name
        self.embedding_model=embedding_model
        self.persist_directory=persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Initializes embeddings for the vectorstore."""
        return GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=SecretStr(google_api_key)
        )

    def create_vectorstore(self) -> Chroma:
        """Creates a new vectorstore with the specified embedding model."""

        embeddings = self.create_embeddings()
        
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory
        )
        
        return vectorstore
    
    
    def load_retriever(self, id_key = "doc_id") -> MultiVectorRetriever:
        """
        Loads a persisted MultiVectorRetriever from disk.
        
        Returns:
            MultiVectorRetriever: The loaded retriever with the persisted vectorstore and docstore.
        
        Raises:
            FileNotFoundError: If the persist_directory or chroma.sqlite3 file does not exist.
        """
        if not self.exists():
            raise FileNotFoundError(f"No vector database found at {self.persist_directory}")
                
        # Load the persisted Chroma vectorstore
        vectorstore = self.create_vectorstore()

        # Initialize an InMemoryStore for the docstore
        store = InMemoryStore()
        
        # Create a MultiVectorRetriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )
        
        return retriever
    
    def exists(self) -> bool:
        """Checks if a persisted database exists"""
        return os.path.exists(self.persist_directory) and os.path.exists(self.persist_directory + "/chroma.sqlite3")
    
    
    def delete(self) -> None:
        """Deletes the persisted database"""
        if os.path.exists(self.persist_directory):
            for f in os.listdir(self.persist_directory):
                os.remove(os.path.join(self.persist_directory, f))
            os.rmdir(self.persist_directory)