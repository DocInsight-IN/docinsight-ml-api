import os

from langchain.prompts import PromptTemplate
from langchain.load import dumps, loads
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            prev_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x:x[1], reverse=True)
    ]
    return reranked_results


def get_retreiver(index_path=os.environ['FAISS_INDEX_PATH']):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings)
    return vectorstore.as_retriever()