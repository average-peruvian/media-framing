#!/usr/bin/env python3
"""
Script de visualización de resultados
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import json

# Añadir directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medianalysis.visualization import Visualizer
from medianalysis.utils import setup_logging
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generar visualizaciones de resultados')
    parser.add_argument('experiment_dir', help='Directorio del experimento')
    parser.add_argument('--output-dir', '-o', help='Directorio de salida para gráficos')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'pdf', 'svg'],
                       help='Formato de salida')
    parser.add_argument('--dpi', '-d', type=int, default=300,
                       help='DPI para gráficos')
    parser.add_argument('--style', '-s', default='seaborn',
                       help='Estilo de matplotlib')
    
    args = parser.parse_args()
    
    setup_logging(level='INFO')
    
    exp_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Generando visualizaciones para: {exp_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Cargar datos
    results_file = exp_dir / 'resultados.xlsx'
    if not results_file.exists():
        logger.error(f"No se encontró {results_file}")
        return
    
    df = pd.read_excel(results_file)
    logger.info(f"Datos cargados: {len(df)} registros")
    
    # Crear visualizador
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
    logger.info("="*80)


if __name__ == '__main__':
    main()
