import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

class LDATopicModel:
    def __init__(self, k=10):
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    def get_lda_vectors(self, token_lists):
        self.dictionary = Dictionary(token_lists)
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
        
        def get_vec_lda(model, corpus, k):
            """ Get the LDA vector representation (probs topic assingment for all documents) """
            n_doc = len(corpus)
            vec_lda = np.zeros((n_doc, k))
            for i in range(n_doc):
                for topic, prob in model.get_document_topics(corpus[i]):
                    vec_lda[i, topic] = prob
            return vec_lda
        vec = get_vec_lda(self.lda_model, self.corpus, self.k)
        return vec

    def fit(self, token_lists):
        if not self.dictionary:
            self.dictionary = Dictionary(token_lists)
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
        if not self.lda_model:
            self.lda_model = LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary, passes=20)

    def predict(self):
        labels = np.array(list(map(lambda x: sorted(self.lda_model.get_document_topics(x), key=lambda x: x[1], reverse=True)[0][0], self.corpus)))
        return labels