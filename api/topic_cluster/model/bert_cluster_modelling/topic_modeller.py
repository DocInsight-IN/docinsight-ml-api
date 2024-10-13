import os
import torch
import umap
import hdbscan
import pandas as pd
from sklearn.metrics import silhouette_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List

import numpy as np

from ..lda.autoencoder import AutoEncoder
from .embedding import BERTEmbedding
from .cluster import KMeansClustering
from ..lda.lda_model import LDATopicModel
from ..topic_mapper import TopicMapper
from ..siamese_bert import SiameseBERT, load_model

class TopicClassifier:
    def __init__(self, k=10, bert_embedding_model='bert-base-uncased', dropout=0.3):
        super(TopicClassifier, self).__init__()

        self.seed_topic_list = []

        self.embedding_model = BERTEmbedding(bert_embedding_model)
        self.cluster_model = KMeansClustering(k=k)
        self.umap_model = umap.UMAP()
        self.ae_model = AutoEncoder()
        self.bert_tokenizer = None

        self.k = k
        self.num_top_terms = 5

        # LDA stuff
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

        self.topics = []
        self.topic_mapping = TopicMapper(self.topics)
        self.hierarchial_mapping = {}


    def fit_transform_lda(self, sentences, embeddings, tokenized_texts):
        vec_bert = self.embedding_model.predict(sentences)
        vec_lda = self.fit_lda_model(vec_bert, embeddings)
        vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
        if not self.ae_model:
            self.ae_model = AutoEncoder()
            self.ae_model.fit(vec_ldabert)
        vec = self.ae_model.predict(vec_ldabert)
        return vec
    
        
    def _cluster_embeddings(self, embeddings):
        """ Cluster reduces embeddings to find topics """
        self.cluster_model.fit(embeddings)
        cluster_labels = self.cluster_model.cluster_model.labels_
        self.topics = cluster_labels
        self.topic_sizes = len(self.topics)
        return cluster_labels
    
    def _hdbscan_cluster(self, embeddings):
        self.hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=self.k, min_samples=2)
        cluster_labels = self.hdbscan_cluster.fit_predict(embeddings)
        self.hierarchial_mapping = self._aggregate_cluster_indices(cluster_labels)
        return cluster_labels

    def _calculate_ctfidf_representation(self, sentences, cluster_lables):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        num_clusters = len(set(cluster_lables))
        ctfifg_repr = np.zeros((num_clusters, tfidf_matrix.shape[1]))

        for i, label in enumerate(set(cluster_lables)):
            cluster_indices = np.where(cluster_lables == label)[0]
            if label in self.hierarchical_mapping:
                hierarchical_label = self.hierarchical_mapping[label]
                cluster_indices = self._aggregate_cluster_indices_for_label(cluster_indices, hierarchical_label)

            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
            ctfifg_repr[i] = cluster_tfidf

        return ctfifg_repr
    
    def _aggregate_cluster_indices_for_label(cluster_labels, cluster_indices, hierarchical_label, hierarchical_mapping):
        """ Aggregate cluster indices based on hierarchial structure """
        aggregated_indices = set(cluster_indices)

        while hierarchical_label in hierarchical_mapping:
            parent_label = hierarchical_mapping[hierarchical_label]
            parent_indices = []

            for idx, label in enumerate(cluster_labels):
                if label == parent_label:
                    parent_indices.append(idx)

            aggregated_indices.update(parent_indices)
            hierarchical_label = parent_label

        return list(aggregated_indices)
    
    def _aggregate_cluster_indices(self, cluster_labels):
        """ Aggregate cluster indices based on hierarchial structure """
        hierarchial_mapping = {}
        for label in set(cluster_labels):
            if label != -1:
                parent_label = label
                while parent_label in set(cluster_labels):
                    parent_label = self._find_parent_label(cluster_labels, parent_label)
                hierarchial_mapping[label] = parent_label
        return hierarchial_mapping
    
    def _find_parent_label(self, cluster_lables, label):
        parent_label = None
        for i, l in enumerate(cluster_lables):
            if l == label:
                if parent_label is None or cluster_lables[i] < parent_label:
                    parent_label = cluster_lables[i]
        return parent_label

    def _extract_topics(self, ctfidf_representation, num_topics):
        vectorizer = TfidfVectorizer()
        ctfidf_cluster_matrix = vectorizer.fit_transform(ctfidf_representation)

        km = KMeans(n_clusters=num_topics, random_state=42)
        km.fit(ctfidf_cluster_matrix)

        terms = vectorizer.get_feature_names_out()
        sorted_centroids = km.cluster_centers_.argsort()[:, ::-1]

        topics = []
        for i in range(num_topics):
            topic_str = " ".join([terms[ind] for ind in sorted_centroids[i, :self.num_top_terms]])
            topics.append(topic_str)

        self.topics = topics
        return topics

    def _reduce_dims(self, embeddings, y=None):
        """ Reduce dimensionality of embeddings using UMAP and train a UMAP model """
        try:
            y = np.array(y) if y is not None else None
            self.umap_model.fit(embeddings, y=y)
        except TypeError:
            self.umap_model.fit(embeddings)

        umap_embeddings = self.umap_model.transform(embeddings)
        return np.nan_to_num(umap_embeddings)
    
    def _create_fit_result_df(self, sentences, topics):
        df_data = {'Sentence': sentences, 'Topic': topics}
        result_df = pd.DataFrame(df_data)
        return result_df
    
    def fit_predict(self, sentences, embeddings):
        if embeddings is None:
            embeddings = self.embedding_model.predict(sentences)
        
        umap_embeddings = self._reduce_dims(embeddings)
        cluster_labels = self._hdbscan_cluster(umap_embeddings)
        ctfidf_repr = self._calculate_ctfidf_representation(sentences, cluster_labels, self.hierarchical_mapping)
        topics = self._extract_topics(ctfidf_repr, self.k)

        result_df = self._create_fit_result_df(sentences, topics)
        return result_df

    def fit_lda_model(self, sentences: List[str], embeddings: np.ndarray = None):
        if not self.lda_model:
            self.lda_model = LDATopicModel(self.k)
        if not embeddings:
            embeddings = self.embedding_model.predict(sentences)
        self.lda_model.fit(embeddings)
        return self.lda_model.predict()