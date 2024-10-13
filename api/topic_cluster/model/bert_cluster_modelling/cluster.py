import numpy as np
from sklearn.cluster import KMeans

from ..base import BaseMLModel

class KMeansClustering(BaseMLModel):
    def __init__(self, k:int, model=KMeans):
        super(KMeansClustering, self).__init__()
        self.k = k
        self.cluster_model = model(self.k)

    def set_model(self, model):
        """ Set the clustering model """
        self.cluster_model = model(n_clusters=self.k)
    
    def predict(self, embeddings: np.ndarray):
        """ Predict cluster labels for input embeddings """
        labels = self.cluster_model.predict(embeddings)
        return labels

    def fit(self, embeddings: np.ndarray):
        """ Fit the clustering model to input embeddings """
        self.cluster_model.fit_transform(embeddings)
        labels = self.cluster_model.labels_
        return labels
    