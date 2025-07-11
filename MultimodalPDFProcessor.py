import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pydantic import SecretStr
import os
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import time
from VectorDBManager import VectorDBManager
from typing import Optional

google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAu9JXweyYbZxiDopgOdRsIIv1D4UkfuXM") # Just for test

class MultimodalPDFProcessor:
    
    def __init__(self, path: str, model: str, embedding_model: str, persist_directory: str = "vector_db"):
        self.path = path
        self.model = model
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.db_manager = VectorDBManager(self.persist_directory)
        self.tables_list = []
        self.texts_list = []
        self.images_list = []


    def extract_content(self):
        """"
        - infer_table_structure: For hi_res strategy only, preserves table structure in HTML (text_as_html) while always extracting plain text (text).
        - strategy:
             - hi_res: Uses layout detection for detailed extraction.
             - ocr_only: Extracts text using OCR.
             - fast: Directly extracts text.
             - auto: Combines fast and hi_res based on page content.
        - extract_image_block_types: For hi_res, extracts specified element types (e.g., images, tables) for output or metadata.
        - extract_image_block_to_payload: For hi_res, encodes extracted images (as base64) into metadata fields for direct use.
        - chunking_strategy: Divides content into chunks based on criteria, like by_title (headings/titles).
        - max_characters: Hard limit for chunk size; splits large elements.
        - new_after_n_chars: Preferred chunk size; stops chunk extension beyond this limit.
        - combine_text_under_n_chars: Merges small sections into larger chunks to avoid excessively small divisions.
        """

        chunks = partition_pdf(
            filename=self.path,
            infer_table_structure=True,
            strategy="hi_res",

            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True, # if true, will extract base64 for API usage

            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=6000,
            new_after_n_chars=6000,
        )

        
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                self.tables_list.append(chunk)
            if "CompositeElement" in str(type(chunk)):
                self.texts_list.append(chunk)

        self.images_list = self.get_images_base64(chunks)
    
    def get_images_base64(self, chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)

        return images_b64
    
    def generate_visual_descriptions(self, image_bytes: bytes) -> str:
        VISUAL_ANALYSIS_PROMPT = """
            TASK 
            Analyze the provided image. First, identify the type of content: "Flowchart," "Chart," or "Other Image." Based on the identified type, describe the content in detail according to the guidelines below. 
            --- 

            ANALYSIS GUIDELINES 

            • If it is a "Flowchart, Graph": Interpret the process step by step, including the starting point, actions, decision branches (if/else), and the endpoint. Briefly summarize the purpose of the process. 
            • If it is a "Chart": Specify the title, type of chart, and describe the axes. Extract the main data and analyze trends or highlights. 
            • If it is "Other Image": Describe the general context, main objects or characters, ongoing actions, and other important details (text, logos, etc.). 
            --- 

            OUTPUT FORMAT 

            Start by clearly stating the type of image, then present the detailed analysis. 
            Example: Type of image: Flowchart Analysis: The process begins with...
            """

        llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=self.model)
        
        # Create the message payload
        message = {
            "type": "user",
            "content": [
                {"type": "text", "text": VISUAL_ANALYSIS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_bytes}"},
                },
            ],
        }
        
        try:
            response = llm.invoke([message])  # Changed from ainvoke to invoke
            response = getattr(response, 'content', response)
            time.sleep(5)
            print(response)
            return response
    
        except Exception as e:
            print(f"Error during visual analysis: {e}")
            return "Fallback description"


    def create_knowledge_retriever(self):
        embeddings = GoogleGenerativeAIEmbeddings(
            model= self.embedding_model,
            google_api_key= SecretStr(google_api_key)
        )


        # Initialize Chroma with Google's embeddings
        vectorstore = Chroma(
            collection_name="multi_modal_rag", 
            embedding_function= embeddings,
            persist_directory= self.persist_directory  # Add if you want persistence
        )
        
        store = InMemoryStore()
        id_key = "doc_id"
        
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # Process texts
        if self.texts_list:
            doc_ids = [str(uuid.uuid4()) for _ in self.texts_list]
            text_docs = [
                Document(page_content=chunk.text, metadata={id_key: doc_ids[i]})
                for i, chunk in enumerate(self.texts_list)
            ]
            retriever.vectorstore.add_documents(text_docs)
            retriever.docstore.mset(list(zip(doc_ids, self.texts_list)))

        # Process tables
        if self.tables_list:
            table_ids = [str(uuid.uuid4()) for _ in self.tables_list]
            table_docs = [
                Document(
                    page_content=chunk.text,
                    metadata={id_key: table_ids[i], "text_as_html": chunk.metadata.text_as_html}
                )
                for i, chunk in enumerate(self.tables_list)
            ]
            retriever.vectorstore.add_documents(table_docs)
            retriever.docstore.mset(list(zip(table_ids, self.tables_list)))

        # Process images
        if self.images_list:
            img_ids = [str(uuid.uuid4()) for _ in self.images_list]
            image_summaries = [self.generate_visual_descriptions(img) for img in self.images_list]
            image_docs = [
                Document(
                    page_content=summary,
                    metadata={
                        id_key: img_ids[i],
                        "type": "image",
                        "base64": self.images_list[i]
                    }
                )
                for i, summary in enumerate(image_summaries)
            ]
            retriever.vectorstore.add_documents(image_docs)
            retriever.docstore.mset(list(zip(img_ids, self.images_list)))
        
        self.db_manager.save_retriever(retriever)

        return retriever
    
    def load_existing_retriever(self) -> Optional[MultiVectorRetriever]:
        """Loads existing retriever if available"""
        if self.db_manager.exists():
            return self.db_manager.load_retriever(
                embedding_model=self.embedding_model
            )
        return None
