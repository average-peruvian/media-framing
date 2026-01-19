"""
Módulo de generación de embeddings
"""

import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import logging

logger = logging.getLogger(__name__)

class Embeddings:
    def __init__(self):
        self.models = {}

    def train_word2vec(self,
        tokenized_texts: list[list[str]],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 10
    ) -> Word2Vec:
        logger.info("Entrenando Word2Vec...")
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1,
            epochs=epochs
        )
        self.models['word2vec'] = model
        logger.info(f"Word2Vec entrenado: {len(model.wv)} palabras")
        return model

    def get_word2vec_vectors(self,
        tokenized_texts: list[list[str]]
    ):
        if 'word2vec' not in self.models:
            raise ValueError('No trained Word2Vec.')

        model = self.models['word2vec']
        vectors = []
        for doc in tokenized_texts:
            word_vecs = [model.wv[word] for word in doc if word in model.wv]
            if word_vecs:
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                vectors.append(np.zeros(model.vector_size))
        return np.array(vectors)

    def train_doc2vec(self,
        tokenized_texts: list[list[str]],
        vector_size: int = 100,
        min_count: int = 2,
        epochs: int = 40
    ) -> Doc2Vec:
        logger.info("Entrenando Doc2Vec...")
        tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) 
                      for i, doc in enumerate(tokenized_texts)]
        model = Doc2Vec(
            documents=tagged_docs,
            vector_size=vector_size,
            min_count=min_count,
            epochs=epochs,
            workers=4
        )
        self.models['doc2vec'] = model
        logger.info(f"Doc2Vec entrenado con {len(tokenized_texts)} documentos")
        return model

    def get_doc2vec_vectors(self,
        tokenized_texts: list[list[str]]
    ):
        if 'doc2vec' not in self.models:
            raise ValueError("No trained Doc2Vec.")

        model = self.models['doc2vec']
        vectors = []
        for doc in tokenized_texts:
            vectors.append(model.infer_vector(doc))
        return np.array(vectors)

    def load_bert_embeddings(self,
        model_name: str = 'distiluse-base-multilingual-cased-v2'
    ):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f'Cargando BERT: {model_name}')
            model = SentenceTransformer(model_name)
            self.models['bert'] = model
            return model
        except ImportError:
            logger.warning('sentence-transformers not available.')
            return None

    def get_bert_vectors(self,
        texts: list[str],
        batch_size: int = 32
    ):
        if 'bert' not in self.models:
            self.load_bert_embeddings()
            if 'bert' not in self.models:
                return None

        logger.info('Generando embeddings BERT.')
        model = self.models['bert']
        vectors = model.encode(
            texts, 
            show_progress_bar = True,
            batch_size = batch_size
        )
        return vectors

    def save_models(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.models, f)
        logger.info(f'Modelos guardados en {path}')

    def load_models(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            self.models = pickle.load(f)
        logger.info(f'Modelos cargados desde {path}')

class SpacyEmbeddings:
    def __init__(self, model_name: str = 'es_core_news_md'):
        import spacy
        self.nlp = spacy.load(model_name)
        logger.info(f'spaCy cargado: {model_name}')

    def get_document_vectors(self, texts: list[str]):
        vectors = []
        for text in texts:
            doc = self.nlp(text[:10000])
            vectors.append(doc.vector)
        return np.array(vectors)

    def get_entity_vectors(self, texts: list[str]):
        results = []
        for text in texts:
            doc = self.nlp(text[:10000])
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'vector': ent.vector
                })
            results.append(entities)
        return results
    