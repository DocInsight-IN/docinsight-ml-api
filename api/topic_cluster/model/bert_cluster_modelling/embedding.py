import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
from sentence_transformers import SentenceTransformer

from ..base import BaseMLModel

class BERTEmbedding(BaseMLModel):
    def __init__(self, model_name='bert-base-nli-mean-tokens'):
        super(BERTEmbedding, self).__init__()
        self._model = None
        self.model_name = model_name

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def predict(self, texts: Union[str, List[str]], batch_size: int=32, verbose:bool=False):
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            batch_embs = np.array(self.model.encode(batch_texts, show_progress_bar=verbose))
            embeddings.append(batch_embs)

        return np.concatenate(embeddings, axis=0)