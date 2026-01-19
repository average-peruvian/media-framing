#!/usr/bin/env python3
"""
Script principal para ejecutar experimentos
"""
import argparse
import sys
from pathlib import Path

# Añadir directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medianalysis.experiments import ExperimentConfig, ExperimentRunner, HyperparameterOptimizer
from medianalysis.utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Ejecutar experimento de análisis de conflictos')
    parser.add_argument('data_path', help='Ruta al archivo de datos (.xlsx)')
    parser.add_argument('--config', '-c', default='configs/default.yaml',
                       help='Ruta al archivo de configuración YAML')
    parser.add_argument('--optimize', '-o', action='store_true',
                       help='Ejecutar optimización de hiperparámetros')
    parser.add_argument('--output-dir', '-d', default='experiments',
                       help='Directorio de salida')
    parser.add_argument('--log-level', '-l', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Nivel de logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Cargar configuración
    config = ExperimentConfig(args.config)
    
    if args.optimize:
        # Modo optimización
        print("="*80)
        print("MODO: OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("="*80)
        
        optimizer = HyperparameterOptimizer(config, n_trials=50)
        
        import pandas as pd
        df = pd.read_excel(args.data_path)
        
        # Preprocesar
        from medianalysis.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        df = preprocessor.process_df(df)
        
        # Optimizar
        print("\nOptimizando embeddings...")
        best_embeddings = optimizer.optimize_embeddings(df)
        
        print("\nOptimizando topic modeling...")
        best_topics = optimizer.optimize_topic_modeling(df)
        
        print("\n" + "="*80)
        print("MEJORES HIPERPARÁMETROS")
        print("="*80)
        print("\nEmbeddings:")
        for k, v in best_embeddings.items():
            print(f"  {k}: {v}")
        
        print("\nTopic Modeling:")
        for k, v in best_topics.items():
            print(f"  {k}: {v}")
    
    else:
        # Modo experimento normal
        print("="*80)
        print("EJECUTANDO EXPERIMENTO")
        print("="*80)
        print(f"Datos: {args.data_path}")
        print(f"Config: {args.config}")
        print(f"Output: {args.output_dir}")
        print("="*80)
        
        runner = ExperimentRunner(config, output_dir=args.output_dir)
        results = runner.run(args.data_path)
        
        print("\n" + "="*80)
        print("EXPERIMENTO COMPLETADO")
        print("="*80)
        print(f"ID: {runner.experiment_id}")
        print(f"Resultados en: {runner.output_dir / runner.experiment_id}")


if __name__ == '__main__':
    main()
