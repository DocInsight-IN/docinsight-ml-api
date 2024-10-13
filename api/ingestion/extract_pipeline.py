
import os
import yaml
import json
from pathlib import Path
from typing import Any, List

import logging
from langchain.schema import Document
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.text_splitter import TokenTextSplitter
from fastapi.encoders import jsonable_encoder
from unstructured.cleaners.core import clean_extra_whitespace

from api.utils.embedding_model import get_embedding_model

LOADER_DICT = {
    ''
}

DATABASE_HOST = ''
DATABASE_PORT = ''
DATABASE_USER = ''
DATABASE_PASSWORD = ''
db_name = ''
path_input_folder = ''
path_extraction_folder = ''


class PDFExtractionPipeline:
    """ Pipeline for extracting text from PDFs and loading them into a vector store """
    
    db: PGVector | None = None
    embedding: CacheBackedEmbeddings

    def __init__(self):
        logging.info("Initializing PDFExtraction Pipeline")

        self.embedding_model = get_embedding_model()

        self.connection_str = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=DATABASE_HOST,
            port=DATABASE_PORT,
            database=db_name,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
        )

        logging.debug(f"Connecting string: {self.connection_str}")

    def run(self, collection_name: str):
        logging.info(f"Running extraction pipeline for collection: {collection_name}")
        self.__load_documents(folder_path=path_input_folder, collection_name=collection_name)

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
        for file_name in os.listdir(folder_path):
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension == ".pdf":
                file_path = f"{folder_path}/{file_name}"
                logging.debug(f"Loading {file_name} from {file_path}")
                try:
                    loader = UnstructuredFileLoader(
                        file_path=file_path,
                        strategy="hi_res",
                        post_process=[clean_extra_whitespace],
                    )

                    file_docs = loader.load()
                    documents.extend(file_docs)

                    json_path = os.path.join(
                        path_extraction_folder,
                        os.path.splitext(file_name)[0] + ".json"
                    )

                    with open(json_path, "w") as json_file:
                        json.dump(jsonable_encoder(file_docs), json_file, indent=4)
                    logging.info(f"{file_name} loaded and saved in JSON format successfully.")

                except Exception as e:
                    logging.error(f"Could not extract text from PDF {file_name}: {repr(e)}")

        return documents


