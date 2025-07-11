import os
from typing import Optional
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

class VectorDBManager:
    """Handles persistence and loading of multimodal vector databases"""
    
    def __init__(self, persist_directory: str = "vector_db"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def save_retriever(
        self, 
        retriever: MultiVectorRetriever,
        collection_name: str = "multi_modal_rag"
    ) -> None:
        """Persists the vector database to disk"""
        if isinstance(retriever.vectorstore, Chroma):
            retriever.vectorstore.persist()
        else:
            raise ValueError("Only Chroma vectorstores are supported for persistence")
    
    def load_retriever(self, embedding_model: str = "models/embedding-001", google_api_key: Optional[str] = None, collection_name: str = "multi_modal_rag") -> MultiVectorRetriever:
        """Loads a persisted retriever from disk"""
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=SecretStr(google_api_key or os.getenv("GOOGLE_API_KEY"))
        )
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory
        )
        
        return MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=InMemoryStore(),  
            id_key="doc_id"
        )
    
    def exists(self) -> bool:
        """Checks if a persisted database exists"""
        return os.path.exists(self.persist_directory) and any(
            f.endswith('.parquet') for f in os.listdir(self.persist_directory))
    
    def delete(self) -> None:
        """Deletes the persisted database"""
        if os.path.exists(self.persist_directory):
            for f in os.listdir(self.persist_directory):
                os.remove(os.path.join(self.persist_directory, f))
            os.rmdir(self.persist_directory)
