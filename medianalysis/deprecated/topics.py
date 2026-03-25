import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
import logging

logger = logging.getLogger(__name__)

class TopicModeller:
    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        self.models = {}

    def fit_lda(self, 
        texts: list[str], 
        max_features: int = 1000,
        max_df: float = 0.8,
        min_df: int = 5,
        ngram_range: tuple = (1,2)
    ):
        logger.info("Entrenando TF-IDF + LDA...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range
        )
        matrix = vectorizer.fit_transform(texts)
        
        model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20
        )
        topics = model.fit_transform(matrix)
        
        self.models['tfidf_lda'] = {
            'model': model,
            'vectorizer': vectorizer,
            'topics': topics,
            'feature_names': vectorizer.get_feature_names_out()
        }
        return self.models['tfidf_lda']

    def fit_nmf(self, 
        texts: list[str],
        max_features: int = 1000,
        max_df: float = 0.8, 
        min_df: int = 5
    ):
        logger.info("Entrenando TF-IDF + NMF...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df
        )
        matrix = vectorizer.fit_transform(texts)
        
        model = NMF(n_components=self.n_topics, random_state=42, max_iter=200)
        topics = model.fit_transform(matrix)
        
        self.models['tfidf_nmf'] = {
            'model': model,
            'vectorizer': vectorizer,
            'topics': topics,
            'feature_names': vectorizer.get_feature_names_out()
        }
        return self.models['tfidf_nmf']

    def fit_lsa(self, 
        texts: list[str], 
        max_features: int = 1000
    ):
        logger.info("Entrenando TF-IDF + LSA...")
        vectorizer = TfidfVectorizer(max_features=max_features)
        matrix = vectorizer.fit_transform(texts)
        
        model = TruncatedSVD(n_components=self.n_topics, random_state=42)
        topics = model.fit_transform(matrix)
        
        self.models['tfidf_lsa'] = {
            'model': model,
            'vectorizer': vectorizer,
            'topics': topics,
            'feature_names': vectorizer.get_feature_names_out()
        }
        return self.models['tfidf_lsa']

    def fit_all(self,
        texts: list[str]
    ):
        self.fit_lda(texts)
        self.fit_nmf(texts)
        self.fit_lsa(texts)
        return self.models

    def get_top_words(self,
        model_name: str,
        n_words: int = 15
    ):
        if model_name not in self.models:
            return []

        model_data = self.models[model_name]
        model = model_data['model']
        feature_names = model_data['feature_names']

        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-n_words][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': [float(topic[i]) for i in top_indices]
            })
        return topics