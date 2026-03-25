#!/usr/bin/env python3
"""
Script de visualización de resultados
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import json
<<<<<<< HEAD
import networkx as nx
import numpy as np
=======
>>>>>>> main

# Añadir directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medianalysis.visualization import Visualizer
from medianalysis.utils import setup_logging
import logging

logger = logging.getLogger(__name__)

<<<<<<< HEAD
def load_embeddings(exp_dir: Path) -> dict:
    embeddings = {}
    embeddings_dir = exp_dir / 'embeddings'
    
    if not embeddings_dir.exists():
        logger.warning("No se encontró carpeta de embeddings")
        return embeddings
    
    for npy_file in embeddings_dir.glob('*.npy'):
        emb_type = npy_file.stem
        try:
            embeddings[emb_type] = np.load(npy_file)
            logger.info(f"{emb_type}: {embeddings[emb_type].shape}")
        except Exception as e:
            logger.error(f"Error cargando {emb_type}: {e}")
    
    return embeddings


def load_topics(exp_dir: Path) -> dict:
    topics_file = exp_dir / 'topics' / 'all_topics.json'
    
    if not topics_file.exists():
        logger.warning("No se encontró archivo de topics")
        return {}
    
    with open(topics_file, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    
    logger.info(f"Topics cargados: {list(topics.keys())}")
    return topics


def load_network(exp_dir: Path):
    network_file = exp_dir / 'networks' / 'network.gexf'
    
    if not network_file.exists():
        logger.warning("No se encontró archivo de red")
        return None
    
    try:
        G = nx.read_gexf(network_file)
        logger.info(f"Red cargada: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
        return G
    except Exception as e:
        logger.error(f"Error cargando red: {e}")
        return None
=======
>>>>>>> main

def main():
    parser = argparse.ArgumentParser(description='Generar visualizaciones de resultados')
    parser.add_argument('experiment_dir', help='Directorio del experimento')
    parser.add_argument('--output-dir', '-o', help='Directorio de salida para gráficos')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'pdf', 'svg'],
                       help='Formato de salida')
    parser.add_argument('--dpi', '-d', type=int, default=300,
                       help='DPI para gráficos')
<<<<<<< HEAD
    parser.add_argument('--skip-embeddings', action='store_true',
                       help='Saltar visualización de embeddings (puede ser lento)')
=======
    parser.add_argument('--style', '-s', default='seaborn',
                       help='Estilo de matplotlib')
>>>>>>> main
    
    args = parser.parse_args()
    
    setup_logging(level='INFO')
    
    exp_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
<<<<<<< HEAD
    logger.info("="*80)
    logger.info(f"GENERANDO VISUALIZACIONES")
    logger.info("="*80)
    logger.info(f"Experimento: {exp_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
=======
    logger.info(f"Generando visualizaciones para: {exp_dir}")
    logger.info(f"Output: {output_dir}")
>>>>>>> main
    
    # Cargar datos
    results_file = exp_dir / 'resultados.xlsx'
    if not results_file.exists():
        logger.error(f"No se encontró {results_file}")
        return
    
    df = pd.read_excel(results_file)
    logger.info(f"Datos cargados: {len(df)} registros")
    
    # Crear visualizador
<<<<<<< HEAD
    viz = Visualizer()
    
    logger.info("\n" + "="*80)
    logger.info("CARGANDO DATOS")
    logger.info("="*80)

    # 1. CARGAR DATOS AUXILIARES
    embeddings = load_embeddings(exp_dir)
    topics = load_topics(exp_dir)
    network = load_network(exp_dir)

    # 2. GENERAR VISUALIZACIONES DE TOPICS
    if topics:
        logger.info("\n" + "="*80)
        logger.info("VISUALIZACIONES DE TOPICS")
        logger.info("="*80)
        
        for model_name, model_topics in topics.items():
            logger.info(f"\n  Procesando modelo: {model_name}")
            
            # Palabras por topic
            logger.info(f"    → plot_topic_words...")
            fig = viz.plot_topic_words(
                model_topics[:10],  # Primeros 10 topics
                n_words=15,
                save_path=output_dir / f'topics_{model_name}_words.{args.format}'
            )
            
        # Evolución temporal si hay columna de tiempo
        topic_cols = [col for col in df.columns if col.startswith('topic_')]
        if topic_cols and 'fecha' in df.columns:
            logger.info(f"    → plot_topic_evolution...")
            fig = viz.plot_topic_evolution(
                df,
                topic_col=topic_cols[0],
                time_col='fecha',
                save_path=output_dir / f'topics_evolution.{args.format}'
            )
    
    # 3. GENERAR VISUALIZACIONES DE EMBEDDINGS
    if embeddings and not args.skip_embeddings:
        logger.info("\n" + "="*80)
        logger.info("VISUALIZACIONES DE EMBEDDINGS")
        logger.info("="*80)
        
        # Detectar topic labels para colorear
        topic_cols = [col for col in df.columns if col.startswith('topic_')]
        labels = None
        if topic_cols:
            labels = df[topic_cols[0]].astype(str).tolist()
            logger.info(f"  Usando {topic_cols[0]} para colorear puntos")
        
        for emb_type, vectors in embeddings.items():
            logger.info(f"\n  Procesando embeddings: {emb_type}")
            
            # Limitar cantidad para t-SNE (es lento)
            max_points = 2000
            if len(vectors) > max_points:
                logger.info(f"    Tomando muestra de {max_points} puntos (total: {len(vectors)})")
                indices = np.random.choice(len(vectors), max_points, replace=False)
                vectors_sample = vectors[indices]
                labels_sample = [labels[i] for i in indices] if labels else None
            else:
                vectors_sample = vectors
                labels_sample = labels
            
            # t-SNE
            logger.info(f"    → plot_embeddings_tsne...")
            fig = viz.plot_embeddings_tsne(
                vectors_sample,
                labels=labels_sample,
                save_path=output_dir / f'embeddings_{emb_type}_tsne.{args.format}'
            )
            
            # UMAP (si está disponible)
            try:
                logger.info(f"    → plot_embeddings_umap...")
                fig = viz.plot_embeddings_umap(
                    vectors_sample,
                    labels=labels_sample,
                    save_path=output_dir / f'embeddings_{emb_type}_umap.{args.format}'
                )
            except:
                logger.warning("    UMAP no disponible, saltando...")
    
    # 4. GENERAR VISUALIZACIONES DE REDES
    if network:
        logger.info("\n" + "="*80)
        logger.info("VISUALIZACIONES DE REDES")
        logger.info("="*80)
        
        logger.info("  → plot_network (spring layout)...")
        fig = viz.plot_network(
            network,
            layout='spring',
            save_path=output_dir / f'network_spring.{args.format}'
        )
        
        logger.info("  → plot_network (circular layout)...")
        fig = viz.plot_network(
            network,
            layout='circular',
            save_path=output_dir / f'network_circular.{args.format}'
        )
    
    # 5. MATRIZ DE CORRELACIÓN
    # Identificar columnas numéricas relevantes
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir embeddings (demasiadas dimensiones)
    corr_cols = [col for col in numeric_cols 
                 if not any(col.startswith(prefix) for prefix in 
                           ['word2vec_dim_', 'doc2vec_dim_', 'spacy_dim_', 'bert_dim_'])]
    
    if len(corr_cols) > 2:
        logger.info("\n" + "="*80)
        logger.info("MATRIZ DE CORRELACIÓN")
        logger.info("="*80)
        logger.info(f"  Columnas: {len(corr_cols)}")
        
        fig = viz.plot_correlation_matrix(
            df,
            corr_cols[:30],  # Máximo 30 variables
            save_path=output_dir / f'correlation_matrix.{args.format}'
        )
    
    # 6. DASHBOARD GENERAL
    logger.info("\n" + "="*80)
    logger.info("DASHBOARD GENERAL")
    logger.info("="*80)
    
    topic_col = None
    cluster_col = None
    
    # Buscar columna de topic
    topic_cols = [col for col in df.columns if col.startswith('topic_') and not col.endswith('_prob')]
    if topic_cols:
        topic_col = topic_cols[0]
    
    # Buscar columna de cluster (si existe)
    cluster_cols = [col for col in df.columns if col.startswith('cluster')]
    if cluster_cols:
        cluster_col = cluster_cols[0]
    
    fig = viz.create_analysis_dashboard(
        df,
        topic_col=topic_col,
        cluster_col=cluster_col,
        save_path=output_dir / f'dashboard.{args.format}'
    )
    
    # 7. WORD CLOUD (si hay texto procesado)
    if 'texto_procesado' in df.columns:
        logger.info("\n" + "="*80)
        logger.info("WORD CLOUD")
        logger.info("="*80)
        
        # Tomar muestra del corpus
        sample_size = min(1000, len(df))
        corpus_sample = ' '.join(df['texto_procesado'].head(sample_size).tolist())
        
        try:
            fig = viz.plot_word_cloud(
                corpus_sample,
                save_path=output_dir / f'wordcloud.{args.format}'
            )
        except:
            logger.warning("  Error generando word cloud (wordcloud library no disponible?)")
    
    # RESUMEN FINAL
    logger.info("\n" + "="*80)
    logger.info("VISUALIZACIONES COMPLETADAS")
    logger.info("="*80)
    logger.info(f"Gráficos guardados en: {output_dir}")
    logger.info(f"\nArchivos generados:")
    
    for viz_file in sorted(output_dir.glob(f'*.{args.format}')):
        logger.info(f"  • {viz_file.name}")
    
=======
    viz = Visualizer(style=args.style)
    
    # Identificar columnas
    actor_cols = [col for col in df.columns if col.startswith('actor_')]
    arg_cols = [col for col in df.columns if col.startswith('arg_')]
    frame_cols = [col for col in df.columns if col.startswith('frame_')]
    
    logger.info(f"Columnas encontradas: {len(actor_cols)} actores, "
               f"{len(arg_cols)} argumentos, {len(frame_cols)} frames")
    
    # Generar visualizaciones
    # Matriz de correlación
    if actor_cols and arg_cols:
        logger.info("Generando matriz de correlación...")
        corr_cols = actor_cols + arg_cols + frame_cols
        fig = viz.plot_correlation_matrix(
            df, corr_cols,
            save_path = output_dir / f'correlacion.{args.format}'
        )
    
    # Dashboard
    if actor_cols and arg_cols and frame_cols:
        logger.info("Generando dashboard...")
        fig = viz.create_dashboard(
            df, actor_cols, arg_cols, frame_cols,
            save_path = output_dir / f'dashboard.{args.format}'
        )
    
    # Topics (si existen)
    summary_file = exp_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        if 'topic_modeling' in summary.get('modules_executed', []):
            logger.info("Intentando generar visualizaciones de topics...")
    
    # Embeddings t-SNE
    embedding_cols = [col for col in df.columns if any(
        prefix in col for prefix in ['w2v_', 'd2v_', 'spacy_', 'bert_']
    )]
    
    if embedding_cols:
        logger.info("Generando visualización t-SNE de embeddings...")
        # Tomar una muestra si hay muchos datos
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # Obtener topic dominante si existe
        topic_cols = [col for col in df.columns if col.startswith('topic_')]
        labels = None
        if topic_cols:
            labels = df_sample[topic_cols[0]].astype(str).tolist()
        
        # Probar con diferentes embeddings
        for prefix in ['w2v_', 'd2v_', 'spacy_', 'bert_']:
            cols = [col for col in embedding_cols if col.startswith(prefix)]
            if cols:
                logger.info(f"  Visualizando {prefix[:-1]} embeddings...")
                embeddings = df_sample[cols].values
                fig = viz.plot_embeddings_tsne(
                    embeddings, labels=labels,
                    save_path=output_dir / f'tsne_{prefix[:-1]}.{args.format}'
                )
    
    logger.info("="*80)
    logger.info("VISUALIZACIONES COMPLETADAS")
    logger.info(f"Gráficos guardados en: {output_dir}")
>>>>>>> main
    logger.info("="*80)


if __name__ == '__main__':
    main()
