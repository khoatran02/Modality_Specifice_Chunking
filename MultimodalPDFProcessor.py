import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_community.embeddings import OpenAIEmbeddings
from pydantic import SecretStr
import os
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import time
from VectorDBManager import VectorDBManager
from typing import Optional

google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAu9JXweyYbZxiDopgOdRsIIv1D4UkfuXM") # Just for test

class MultimodalPDFProcessor:
    
    def __init__(self, path: str, model: str, embedding_model: str, collection_name: str, persist_directory: str = "vector_db"):
        self.path = path
        self.model = model
        self.db_manager = VectorDBManager(embedding_model=embedding_model, collection_name=collection_name, persist_directory=persist_directory)
        self.tables_list = []
        self.texts_list = []
        self.images_list = []


    def extract_content(self):
        """
        Extracts tables, text, and images from the PDF using unstructured.partition.pdf.partition_pdf.
        Handles fallback if OCR is unavailable.
        """
        def process_chunks(chunks):
            for chunk in chunks:
                chunk_type = str(type(chunk))
                if "Table" in chunk_type:
                    self.tables_list.append(chunk)
                elif "CompositeElement" in chunk_type:
                    self.texts_list.append(chunk)
            self.images_list = self.get_images_base64(chunks)

        try:
            chunks = partition_pdf(
                filename=self.path,
                infer_table_structure=True,
                strategy="auto",
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=6000,
                new_after_n_chars=6000,
            )
            process_chunks(chunks)
        except Exception as e:
            if "OCRAgent" in str(e):
                print("OCR initialization failed, falling back to non-OCR extraction")
                try:
                    chunks = partition_pdf(
                        filename=self.path,
                        strategy="fast",
                        chunking_strategy="by_title",
                        max_characters=10000,
                        combine_text_under_n_chars=6000,
                        new_after_n_chars=6000,
                    )
                    process_chunks(chunks)
                except Exception as inner_e:
                    print(f"Fallback extraction failed: {inner_e}")
                    self.tables_list, self.texts_list, self.images_list = [], [], []
            else:
                print(f"Extraction failed: {e}")
                self.tables_list, self.texts_list, self.images_list = [], [], []

    
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
            return str(response)
    
        except Exception as e:
            print(f"Error during visual analysis: {e}")
            return "Fallback description"


    def create_knowledge_retriever(self):

        """Creates a knowledge retriever from the extracted content."""   
             
        id_key = "doc_id"
        retriever = self.db_manager.load_retriever(id_key = "doc_id")

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

        retriever.vectorstore.similarity_search("")
        return retriever
    