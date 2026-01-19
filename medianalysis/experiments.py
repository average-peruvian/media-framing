"""
Módulo de experimentación y optimización
"""
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ExperimentConfig:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuración cargada: {self.config_path}")
        return config
    
    def get(self, key: str, default = None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value if value is not None else default
    
    def get_preprocessing_config(self):
        return self.get('preprocessing', {})
    
    def get_embeddings_config(self):
        return self.get('embeddings', {})
    
    def get_topic_modeling_config(self):
        return self.get('topic_modeling', {})

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, output_dir: str = 'experiments'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {}
    
    def run(self, data_path: str):
        logger.info(f"Iniciando experimento: {self.experiment_id}")
        
        # Cargar datos
        df = self._load_data(data_path)
        
        # Preprocesamiento
        if self.config.get('preprocessing.enabled', True):
            df = self._run_preprocessing(df)
        
        # Embeddings
        if self.config.get('embeddings.enabled', True):
            embeddings_results = self._run_embeddings(df)
            self.results['embeddings'] = embeddings_results
        
        # Topic modeling
        if self.config.get('topic_modeling.enabled', True):
            topic_results = self._run_topic_modeling(df)
            self.results['topic_modeling'] = topic_results
        
        # Redes
        if self.config.get('networks.enabled', True):
            network_results = self._run_network_analysis(df)
            self.results['networks'] = network_results
        
        # Guardar resultados
        self._save_results(df)
        
        logger.info(f"Experimento completado: {self.experiment_id}")
        return self.results
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        logger.info(f"Cargando datos: {data_path}")
        df = pd.read_excel(data_path)
        
        # Aplicar sample si está configurado
        sample_size = self.config.get('data.sample_size')
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Muestra tomada: {sample_size} documentos")
        
        return df
    
    def _run_preprocessing(self, df: pd.DataFrame):
        from .preprocessing import TextPreprocessor
        
        logger.info("Ejecutando preprocesamiento...")
        config = self.config.get_preprocessing_config()
        
        preprocessor = TextPreprocessor(
            custom_stopwords = config.get('custom_stopwords') # pyright: ignore[reportOptionalMemberAccess]
        )
        
        text_column = config.get('text_column', 'body')
        df = preprocessor.process_df(df, text_column)
        
        # Filtrar textos cortos
        min_length = config.get('min_length', 50)
        df = df[df['texto_procesado'].str.len() > min_length]
        
        logger.info(f"Preprocesamiento completado: {len(df)} documentos válidos")
        return df
    
    def _run_embeddings(self, df: pd.DataFrame):
        """Ejecuta generación de embeddings"""
        from .embeddings import Embeddings, SpacyEmbeddings
        
        logger.info("Generando embeddings...")
        config = self.config.get_embeddings_config()
        results = {}
        
        embedder = Embeddings()
        tokenized_texts = df['tokens'].tolist()
        
        # Word2Vec
        if config.get('word2vec.enabled', True):
            w2v_params = config.get('word2vec', {})
            embedder.train_word2vec(
                tokenized_texts,
                vector_size=w2v_params.get('vector_size', 100),
                window=w2v_params.get('window', 5),
                min_count=w2v_params.get('min_count', 2),
                epochs=w2v_params.get('epochs', 10)
            )
            w2v_vectors = embedder.get_word2vec_vectors(tokenized_texts)
            results['word2vec'] = w2v_vectors
            logger.info(f"Word2Vec: {w2v_vectors.shape}")
        
        # Doc2Vec
        if config.get('doc2vec.enabled', True):
            d2v_params = config.get('doc2vec', {})
            embedder.train_doc2vec(
                tokenized_texts,
                vector_size=d2v_params.get('vector_size', 100),
                min_count=d2v_params.get('min_count', 2),
                epochs=d2v_params.get('epochs', 40)
            )
            d2v_vectors = embedder.get_doc2vec_vectors(tokenized_texts)
            results['doc2vec'] = d2v_vectors
            logger.info(f"Doc2Vec: {d2v_vectors.shape}")
        
        # BERT
        if config.get('bert.enabled', False):
            bert_params = config.get('bert', {})
            model_name = bert_params.get('model_name', 'distiluse-base-multilingual-cased-v2')
            embedder.load_bert_embeddings(model_name)
            
            # Limitar cantidad para BERT (es lento)
            max_texts = bert_params.get('max_texts', 1000)
            texts_for_bert = df['body'].head(max_texts).tolist()
            bert_vectors = embedder.get_bert_vectors(texts_for_bert)
            if bert_vectors is not None:
                results['bert'] = bert_vectors
                logger.info(f"BERT: {bert_vectors.shape}")
        
        # spaCy
        if config.get('spacy.enabled', True):
            spacy_params = config.get('spacy', {})
            model_name = spacy_params.get('model_name', 'es_core_news_md')
            try:
                spacy_embedder = SpacyEmbeddings(model_name)
                max_texts = spacy_params.get('max_texts', 1000)
                texts_for_spacy = df['body'].head(max_texts).tolist()
                spacy_vectors = spacy_embedder.get_document_vectors(texts_for_spacy)
                results['spacy'] = spacy_vectors
                logger.info(f"spaCy: {spacy_vectors.shape}")
            except Exception as e:
                logger.warning(f"Error con spaCy: {e}")
        
        return results
    
    def _run_topic_modeling(self, df: pd.DataFrame):
        """Ejecuta topic modeling"""
        from .topics import TopicModeller
        
        logger.info("Ejecutando topic modeling...")
        config = self.config.get_topic_modeling_config()
        
        n_topics = config.get('n_topics', 10)
        modeler = TopicModeller(n_topics=n_topics)
        
        texts = df['texto_procesado'].tolist()
        
        methods = config.get('methods', ['lda', 'nmf', 'lsa'])
        results = {}
        
        if 'lda' in methods:
            results['lda'] = modeler.fit_lda(texts)
        if 'nmf' in methods:
            results['nmf'] = modeler.fit_nmf(texts)
        if 'lsa' in methods:
            results['lsa'] = modeler.fit_lsa(texts)
        
        logger.info(f"Topic modeling completado: {len(results)} modelos")
        return results
    
    def _run_network_analysis(self, df: pd.DataFrame):
        """Ejecuta análisis de redes"""
        from .networks import SemanticNetworkAnalyzer
        
        logger.info("Ejecutando análisis de redes...")
        config = self.config.get('networks', {})
        
        # Asegurarse que tenemos las columnas de actores
        actor_cols = [col for col in df.columns if col.startswith('actor_')]
        if not actor_cols:
            logger.warning("No hay columnas de actores, saltando análisis de redes")
            return {}
        
        analyzer = SemanticNetworkAnalyzer()
        threshold = config.get('threshold', 5)
        
        graph = analyzer.build_cooccurrence_network(df, actor_cols, threshold)
        metrics = analyzer.compute_centrality_metrics()
        communities = analyzer.detect_communities()
        stats = analyzer.get_network_stats()
        
        results = {
            'graph': graph,
            'metrics': metrics,
            'communities': communities,
            'stats': stats
        }
        
        logger.info(f"Red construida: {stats['nodes']} nodos, {stats['edges']} aristas")
        return results
    
    def _save_results(self, df: pd.DataFrame):
        """Guarda resultados del experimento"""
        exp_dir = self.output_dir / self.experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # DataFrame principal
        main_file = exp_dir / 'resultados.xlsx'
        df.to_excel(main_file, index=False)
        logger.info(f"Resultados guardados: {main_file}")
        
        # Configuración usada
        config_file = exp_dir / 'config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config.config, f, allow_unicode=True)
        
        # Resumen JSON
        summary = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'n_documents': len(df),
            'modules_executed': list(self.results.keys())
        }
        
        summary_file = exp_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Experimento guardado en: {exp_dir}")


class HyperparameterOptimizer:
    """Optimización de hiperparámetros con Optuna"""
    
    def __init__(self, config: ExperimentConfig, n_trials: int = 50):
        self.config = config
        self.n_trials = n_trials
        self.best_params = None
    
    def optimize_embeddings(self, df: pd.DataFrame):
        """Optimiza hiperparámetros de embeddings"""
        try:
            import optuna
        except ImportError:
            logger.error("Optuna no disponible. Instalar con: pip install optuna")
            return {}
        
        logger.info("Iniciando optimización de embeddings...")
        
        def objective(trial):
            from .embeddings import Embeddings
            
            # Hiperparámetros a optimizar
            vector_size = trial.suggest_int('vector_size', 50, 200, step=50)
            window = trial.suggest_int('window', 3, 10)
            min_count = trial.suggest_int('min_count', 1, 5)
            epochs = trial.suggest_int('epochs', 5, 20)
            
            # Entrenar modelo
            embedder = Embeddings()
            tokenized_texts = df['tokens'].tolist()
            embedder.train_word2vec(tokenized_texts, vector_size, window, min_count, epochs)
            
            # Métrica: coherencia (simplificada)
            # En un caso real, usar métricas más sofisticadas
            model = embedder.models['word2vec']
            score = len(model.wv)  # Vocabulario como proxy
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        logger.info(f"Mejores parámetros: {self.best_params}")
        
        return self.best_params
    
    def optimize_topic_modeling(self, df: pd.DataFrame):
        """Optimiza hiperparámetros de topic modeling"""
        try:
            import optuna
        except ImportError:
            logger.error("Optuna no disponible")
            return {}
        
        logger.info("Iniciando optimización de topic modeling...")
        
        def objective(trial):
            from .topics import TopicModeller
            
            n_topics = trial.suggest_int('n_topics', 5, 20)
            max_features = trial.suggest_int('max_features', 500, 2000, step=500)
            max_df = trial.suggest_float('max_df', 0.5, 0.9)
            min_df = trial.suggest_int('min_df', 2, 10)
            
            modeler = TopicModeller(n_topics=n_topics)
            texts = df['texto_procesado'].tolist()
            result = modeler.fit_lda(texts, max_features, max_df, min_df)
            
            # Métrica: perplexity del modelo LDA
            score = -result['model'].perplexity(result['vectorizer'].transform(texts))
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        logger.info(f"Mejores parámetros: {self.best_params}")
        
        return self.best_params
