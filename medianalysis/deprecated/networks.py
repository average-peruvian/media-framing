"""
Análisis en grafo
"""

import numpy as np
import pandas as pd
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class SemanticNetworkAnalyzer:
    """Análisis de redes de co-ocurrencia"""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def build_cooccurrence_network(self, 
        df: pd.DataFrame, 
        columns: list[str],
        threshold: int = 5
    ):
        logger.info("Construyendo red de co-ocurrencia...")
        
        n_nodes = len(columns)
        cooccurrence = np.zeros((n_nodes, n_nodes))
        
        for idx, row in df.iterrows():
            present_nodes = [i for i, col in enumerate(columns) if row[col] > 0]
            for i in present_nodes:
                for j in present_nodes:
                    if i != j:
                        cooccurrence[i][j] += 1
        
        for i, node_i in enumerate(columns):
            for j, node_j in enumerate(columns):
                if i < j and cooccurrence[i][j] >= threshold:
                    self.graph.add_edge(
                        node_i,
                        node_j,
                        weight=float(cooccurrence[i][j])
                    )
        
        logger.info(f"Red construida: {self.graph.number_of_nodes()} nodos, "
                   f"{self.graph.number_of_edges()} aristas")
        return self.graph
    
    def compute_centrality_metrics(self):
        logger.info("Calculando métricas de centralidad...")
        metrics = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality(self.graph, max_iter=1000)
        }
        return metrics
    
    def detect_communities(self):
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(self.graph)
            logger.info(f"Comunidades detectadas: {len(communities)}")
            return list(communities)
        except Exception as e:
            logger.warning(f"Error detectando comunidades: {e}")
            return []
    
    def export_network(self, filepath: str, format: str = 'gexf'):
        if format == 'gexf':
            nx.write_gexf(self.graph, filepath)
        elif format == 'gml':
            nx.write_gml(self.graph, filepath)
        elif format == 'edgelist':
            nx.write_edgelist(self.graph, filepath)
        logger.info(f"Red exportada: {filepath}")
    
    def get_network_stats(self):
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }