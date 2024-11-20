
import os
import yaml
import json
from pathlib import Path
from typing import Any, List
import threading
import logging
from langchain.schema import Document
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.azure_ai_data import AzureAIDataLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.text_splitter import TokenTextSplitter
from fastapi.encoders import jsonable_encoder
from unstructured.cleaners.core import clean_extra_whitespace
from api.core.config import settings
from api.utils.embedding_model import get_embedding_model


class DocumentExtractionPipeline:
    """ Pipeline for extracting text from PDFs and loading them into a vector store """
    
    db: PGVector | None = None
    embedding: CacheBackedEmbeddings

    def __init__(self, azure_endpoint: str, azure_key: str) -> None:
        logging.info("Initializing PDFExtraction Pipeline")

        self.embedding_model = get_embedding_model()
        self.connection_str = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=settings.DATABASE_HOST,
            port=settings.DATABASE_PORT,
            database=settings.DATABASE_NAME,
            user=settings.DATABASE_USER,
            password=settings.DATABASE_PASSWORD,
        )
        logging.debug(f"Database Connecting string: {self.connection_str}")
        self.azure_endpoint = azure_endpoint
        self.azure_key = azure_key

    def run(self, collection_name: str, folder_path: str) -> None:
        logging.info(f"Running extraction pipeline for collection: {collection_name}")
        self.__load_documents(folder_path=folder_path, collection_name=collection_name)
        logging.info(f"Pipeline completed for collection: {collection_name}")

    def __load_documents(self, folder_path: str, collection_name: str) -> PGVector:
        """ Load documents into the vectorstore """
        text_documents = self.__load_docs(folder_path)
        logging.info(f"Loaded {len(text_documents)} documents")

        text_splitter = TokenTextSplitter(
            chunk_size=220,
            chunk_overlap=100,
        )
        texts = text_splitter.split_documents(text_documents)
        for text in texts:
            text.metadata["type"] = "Text"
        docs = [*texts]

        vector_store = PGVector.from_documents(
            embedding=self.embedding_model,
            collection_name=collection_name,
            documents=docs,
            connection_string=self.connection_str,
            pre_delete_collection=True,
        )

        return vector_store
    
    def __load_docs(self, folder_path: str) -> List[Document]:
        """ Unsing Unstructured PDF miner to convert PDF documents to raw text chunks"""
        documents = []
        threads = []
        thread_lock = threading.Lock()

        file_path = os.path.join(folder_path, file_name)

        def process_pdf(filename: str):
            try:
                loader = UnstructuredFileLoader(
                    file_path=file_path,
                    strategy="hi_res",
                    post_process=[clean_extra_whitespace],
                )
                file_docs = loader.load()
                with thread_lock:
                    documents.extend(file_docs)
                json_path = os.path.join(
                    settings.EXTRACTION_FOLDER,
                    f"{os.path.splitext(file_name)[0]}.json"
                )
                with open(json_path, "w") as json_file:
                    json.dump(jsonable_encoder(file_docs), json_file, indent=4)
                    logging.info(f"{file_name} loaded and saved in JSON format successfully.")

            except Exception as e:
                logging.error(f"Error processing {file_name}: {repr(e)}")

        def process_office(filename: str):
            file_path = os.path.join(folder_path, filename)
            try:
                loader = AzureAIDataLoader(
                    api_endpoint=self.azure_endpoint,
                    api_key=self.azure_key,
                    file_path=file_path,
                    api_model="prebuilt-layout"
                )
                file_docs = loader.load()
                with thread_lock:
                    documents.extend(file_docs)
                json_path = os.path.join(settings.EXTRACTION_FOLDER, f"{os.path.splitext(file_name)[0]}.json")
                with open(json_path, "w") as json_file:
                    json.dump(jsonable_encoder(file_docs), json_file, indent=4)
                logging.info(f"{file_name} loaded and saved in JSON format successfully.")

            except Exception as e:
                logging.error(f"Error process {file_name}: {repr(e)}")


        for file_name in os.listdir(folder_path):
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension == ".pdf":
                thread = threading.Thread(target=process_pdf, args=(file_name,))
                threads.append(thread)
                thread.start()
            elif file_extension in {".pptx", ".xlsx", ".docx"}:
                thread = threading.Thread(target=process_office, args=(file_name,))
                threads.append(thread)
                thread.start()
            if len(threads) >= settings.MAX_THREADS:
                for t in threads:
                    t.join()
                threads = []
                
        for t in threads:
            t.join()

        return documents


