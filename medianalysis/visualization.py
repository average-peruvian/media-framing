"""
Módulo de visualización limpio
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import logging

logger = logging.getLogger(__name__)

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
<<<<<<< HEAD
    def __init__(self, figsize = (12, 8)):
        self.figsize = figsize
=======
    def __init__(self, style: str = 'seaborn', figsize = (12, 8)):
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
>>>>>>> main
    
    def plot_topic_distribution(self, 
        topic_results, 
        model_name: str = 'tfidf_lda',
        top_n: int = 10,
        save_path = None
    ):
        fig, ax = plt.subplots(figsize=(14, 10))
        
        if model_name not in topic_results:
            logger.warning(f"Modelo {model_name} no encontrado")
            return fig
        
        topics = topic_results[model_name]['topics']
        
        topic_sizes = topics.sum(axis=0)
        topic_indices = np.argsort(topic_sizes)[::-1][:top_n]
        
        labels = [f'Topic {i}' for i in topic_indices]
        sizes = [topic_sizes[i] for i in topic_indices]
        
        colors = sns.color_palette("hls", top_n)
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Distribución de Topics - {model_name.upper()}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {save_path}")
        
        return fig
    
    def plot_topic_words(self, 
        topics_data,
        n_words: int = 10,
        save_path = None
    ):
        n_topics = len(topics_data)
        fig, axes = plt.subplots(nrows=(n_topics + 1) // 2, ncols=2, 
                                figsize=(16, 4 * ((n_topics + 1) // 2)))
        axes = axes.flatten()
        
        for idx, topic in enumerate(topics_data):
            if idx >= len(axes):
                break
            
            words = topic['words'][:n_words]
            weights = topic['weights'][:n_words]
            
            axes[idx].barh(range(len(words)), weights, 
                          color=sns.color_palette("viridis", len(words)))
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].set_xlabel('Peso')
            axes[idx].set_title(f'Topic {topic["topic_id"]}', fontweight='bold')
            axes[idx].invert_yaxis()
        
        # Ocultar ejes vacíos
        for idx in range(len(topics_data), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Top Palabras por Topic', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_network(self, 
        graph: nx.Graph, 
        layout: str = 'spring',
        node_size_attr = None,
        node_color_attr = None,
        save_path = None
    ):
        fig, ax = plt.subplots(figsize=(14, 14))
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(graph, k=0.5, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Tamaño de nodos basado en degree centrality
        if node_size_attr:
            node_attrs = nx.get_node_attributes(graph, node_size_attr)
            if node_attrs:
                node_sizes = [node_attrs.get(node, 1) * 100 for node in graph.nodes()]
            else:
                node_sizes = [graph.degree(node) * 100 for node in graph.nodes()]
        else:
            node_sizes = [graph.degree(node) * 100 for node in graph.nodes()]
        
        # Color de nodos
        node_colors = 'lightblue'
        if node_color_attr:
            node_attrs = nx.get_node_attributes(graph, node_color_attr)
            if node_attrs:
                unique_values = list(set(node_attrs.values()))
                color_map = dict(zip(unique_values, sns.color_palette("husl", len(unique_values))))
                node_colors = [color_map[node_attrs.get(node, unique_values[0])] 
                              for node in graph.nodes()]
        
        # Ancho de aristas basado en peso
        edges = graph.edges()
        weights = [graph[u][v].get('weight', 1) for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [w / max_weight * 5 for w in weights]
        
        # Dibujar
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(graph, pos, width=edge_widths, 
                              alpha=0.5, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
        
        ax.set_title('Red de Co-ocurrencia', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_embeddings_tsne(self,
            embeddings: np.ndarray, 
            perplexity: int = 30,
            labels = None,
            save_path = None):
        from sklearn.manifold import TSNE
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        logger.info("Aplicando t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        if labels is not None:
            unique_labels = list(set(labels))
            colors = sns.color_palette("husl", len(unique_labels))
            label_to_color = dict(zip(unique_labels, colors))
            
            for label in unique_labels:
                mask = np.array([l == label for l in labels])
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          label=label, alpha=0.6, s=50, 
                          color=label_to_color[label])
            ax.legend()
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      alpha=0.6, s=50, color='steelblue')
        
        ax.set_xlabel('t-SNE Dimensión 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimensión 2', fontsize=12)
        ax.set_title('Visualización de Embeddings (t-SNE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_embeddings_umap(self, 
            embeddings: np.ndarray,
            labels = None,
            n_neighbors: int = 15,
            min_dist: float = 0.1,
            save_path = None):
        try:
            import umap
        except ImportError:
            logger.error("UMAP no disponible. Instalar con: pip install umap-learn")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        logger.info("Aplicando UMAP...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        if labels is not None:
            unique_labels = list(set(labels))
            colors = sns.color_palette("husl", len(unique_labels))
            label_to_color = dict(zip(unique_labels, colors))
            
            for label in unique_labels:
                mask = np.array([l == label for l in labels])
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          label=label, alpha=0.6, s=50,
                          color=label_to_color[label])
            ax.legend()
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      alpha=0.6, s=50, color='steelblue')
        
        ax.set_xlabel('UMAP Dimensión 1', fontsize=12)
        ax.set_ylabel('UMAP Dimensión 2', fontsize=12)
        ax.set_title('Visualización de Embeddings (UMAP)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, 
        df: pd.DataFrame, 
        columns: list[str],
        save_path = None
    ):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        corr = df[columns].corr()
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, 
                   cbar_kws={'label': 'Correlación'})
        
        ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_clustering_results(self, 
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        method: str = 'tsne',
        save_path = None
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Reducir dimensionalidad
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(random_state=42)
            except ImportError:
                logger.warning("UMAP no disponible, usando t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plotear clusters
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        colors = sns.color_palette("husl", n_clusters)
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      label=f'Cluster {i}', alpha=0.6, s=50, color=colors[i])
        
        # Noise points (si hay)
        if -1 in cluster_labels:
            mask = cluster_labels == -1
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      label='Noise', alpha=0.3, s=30, color='gray', marker='x')
        
        ax.set_xlabel(f'{method.upper()} Dimensión 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Dimensión 2', fontsize=12)
        ax.set_title(f'Resultados de Clustering ({n_clusters} clusters)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_topic_evolution(self,
        df: pd.DataFrame, 
        topic_col: str,
        time_col: str,
        save_path = None
    ):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Convertir a datetime
        df_time = df.copy()
        df_time[time_col] = pd.to_datetime(df_time[time_col])
        
        # Contar topics por período
        df_time = df_time.set_index(time_col)
        topic_counts = df_time.groupby([pd.Grouper(freq='M'), topic_col]).size().unstack(fill_value=0)
        
        # Plotear
        topic_counts.plot(ax=ax, linewidth=2, marker='o')
        
        ax.set_xlabel('Tiempo', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title('Evolución Temporal de Topics', fontsize=14, fontweight='bold')
        ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_word_cloud(self, 
        text: str, 
        save_path = None
    ):
        try:
            from wordcloud import WordCloud
        except ImportError:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis').generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud del Corpus', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(self,
        df: pd.DataFrame,
        topic_col = None,
        cluster_col = None,
        save_path = None
    ):
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Distribución de longitud de textos
        ax1 = fig.add_subplot(gs[0, 0])
        if 'texto_procesado' in df.columns:
            lengths = df['texto_procesado'].str.len()
            ax1.hist(lengths, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Longitud de texto')
            ax1.set_ylabel('Frecuencia')
            ax1.set_title('Distribución de Longitud de Textos', fontweight='bold')
            ax1.axvline(lengths.median(), color='red', linestyle='--', 
                       label=f'Mediana: {lengths.median():.0f}')
            ax1.legend()
        
        # Distribución de topics
        ax2 = fig.add_subplot(gs[0, 1])
        if topic_col and topic_col in df.columns:
            topic_counts = df[topic_col].value_counts().head(10)
            ax2.bar(range(len(topic_counts)), topic_counts.values, 
                   color=sns.color_palette("husl", len(topic_counts)))
            ax2.set_xticks(range(len(topic_counts)))
            ax2.set_xticklabels([f'T{i}' for i in topic_counts.index])
            ax2.set_ylabel('Frecuencia')
            ax2.set_title('Top 10 Topics', fontweight='bold')
        
        # Distribución de clusters
        ax3 = fig.add_subplot(gs[1, 0])
        if cluster_col and cluster_col in df.columns:
            cluster_counts = df[cluster_col].value_counts()
            ax3.bar(range(len(cluster_counts)), cluster_counts.values,
                   color=sns.color_palette("Set2", len(cluster_counts)))
            ax3.set_xticks(range(len(cluster_counts)))
            ax3.set_xticklabels([f'C{i}' for i in cluster_counts.index])
            ax3.set_ylabel('Frecuencia')
            ax3.set_title('Distribución de Clusters', fontweight='bold')
        
        # Estadísticas generales
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        stats_text = f"""
        ESTADÍSTICAS GENERALES
        
        Total de documentos: {len(df):,}
        """
        
        if 'texto_procesado' in df.columns:
            lengths = df['texto_procesado'].str.len()
            stats_text += f"""
        Longitud promedio: {lengths.mean():.0f} caracteres
        Longitud mediana: {lengths.median():.0f} caracteres
        """
        
        if topic_col and topic_col in df.columns:
            n_topics = df[topic_col].nunique()
            stats_text += f"""
        Número de topics: {n_topics}
        Topic dominante: {df[topic_col].mode()[0]}
        """
        
        if cluster_col and cluster_col in df.columns:
            n_clusters = df[cluster_col].nunique()
            stats_text += f"""
        Número de clusters: {n_clusters}
        Cluster más grande: {df[cluster_col].value_counts().index[0]} 
                            ({df[cluster_col].value_counts().values[0]} docs)
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Espacio para más gráficos si es necesario
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        ax5.text(0.5, 0.5, 'Espacio reservado para análisis adicional', 
                transform=ax5.transAxes, ha='center', va='center',
                fontsize=14, style='italic', color='gray')
        
        fig.suptitle('Dashboard de Análisis NLP', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard guardado: {save_path}")
        
        return fig