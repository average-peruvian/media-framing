#!/bin/bash
# Script de instalación completa

echo "============================================"
echo "INSTALANDO medianalysis FRAMEWORK"
echo "============================================"

# Verificar Python
echo ""
echo "Verificando Python..."
python3 --version || { echo "Error: Python 3 no encontrado"; exit 1; }

# Instalar dependencias
echo ""
echo "Instalando dependencias..."
pip install -r requirements.txt || { echo "Error instalando dependencias"; exit 1; }

# Instalar paquete
echo ""
echo "Instalando paquete..."
pip install -e . || { echo "Error instalando paquete"; exit 1; }

# Descargar modelo spaCy
echo ""
echo "Descargando modelo de spaCy (español)..."
python -m spacy download es_core_news_md || echo "Advertencia: No se pudo descargar modelo de spaCy"

# Verificar instalación
echo ""
echo "============================================"
echo "VERIFICANDO INSTALACIÓN"
echo "============================================"
python test_installation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ INSTALACIÓN COMPLETADA"
    echo "============================================"
    echo ""
    echo "El framework está listo para usar. Ver QUICKSTART.md para ejemplos."
    echo ""
    echo "Comandos rápidos:"
    echo "  python scripts/run_experiment.py datos.xlsx"
    echo "  python scripts/visualize_results.py experiments/<ID>"
    echo ""
else
    echo ""
    echo "Advertencia: Algunas pruebas fallaron"
    echo "El framework puede funcionar parcialmente"
fi
