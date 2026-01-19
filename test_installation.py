#!/usr/bin/env python3
"""
Script de prueba para verificar instalación
"""
import sys

def test_imports():
    """Prueba que todos los módulos se importen correctamente"""
    print("Probando imports...")
    
    try:
        from medianalysis import (
            TextPreprocessor,
            Embeddings,
            TopicModeller,
            SemanticNetworkAnalyzer,
            ExperimentRunner
        )
        print("✓ Módulos principales OK")
    except Exception as e:
        print(f"✗ Error en módulos principales: {e}")
        return False
    
    try:
        from medianalysis.visualization import Visualizer
        print("✓ Visualización OK")
    except Exception as e:
        print(f"✗ Error en visualización: {e}")
        return False
    
    try:
        from medianalysis.experiments import ExperimentConfig
        print("✓ Experiments OK")
    except Exception as e:
        print(f"✗ Error en experiments: {e}")
        return False
    
    return True

def test_preprocessing():
    """Prueba preprocesamiento básico"""
    print("\nProbando preprocesamiento...")
    
    try:
        from medianalysis import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        text = "Este es un TEXTO de prueba con números 123 y URLs http://example.com"
        clean = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(clean)
        
        assert len(clean) > 0
        assert len(tokens) > 0
        
        print(f"  Input: {text[:50]}...")
        print(f"  Clean: {clean[:50]}...")
        print(f"  Tokens: {tokens}")
        print("✓ Preprocesamiento OK")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("="*80)
    print("PRUEBA DE INSTALACIÓN - medianalysis")
    print("="*80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Preprocesamiento", test_preprocessing()))
    
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_pass = all(r[1] for r in results)
    
    if all_pass:
        print("\n✓ Todas las pruebas pasaron exitosamente")
        print("El framework está listo para usar")
        return 0
    else:
        print("\n✗ Algunas pruebas fallaron")
        print("Revisa los errores arriba")
        return 1

if __name__ == '__main__':
    sys.exit(main())
