# 🛠️ Scripts de Análisis

Scripts útiles para analizar y visualizar el entrenamiento.

## `analyze_training.py`

Analiza y visualiza en detalle los logs de un entrenamiento específico.

### Uso

```bash
python scripts/analyze_training.py logs/MAPPO_20251101_143000
```

### Características

✅ **Estadísticas detalladas:**
- Mejor, peor y promedio de recompensas
- Identificación de la mejor iteración
- Análisis de mejora total

✅ **Visualizaciones avanzadas:**
- Recompensa con media móvil y rangos min-max
- Mejora incremental por iteración (barras verdes/rojas)
- Pérdidas en escala logarítmica
- Entropía con umbrales de exploración

✅ **Exportación:**
- Gráficas en alta resolución (300 DPI)
- Análisis guardado en `detailed_analysis.png`

### Ejemplo de Salida

```
📊 Analizando entrenamiento: MAPPO_20251101_143000
============================================================

📈 ESTADÍSTICAS GENERALES
------------------------------------------------------------
Total de iteraciones: 50

🎯 RECOMPENSAS
------------------------------------------------------------
Mejor: -85.32
Peor: -180.45
Promedio: -125.67
Desviación estándar: 25.43
Última: -92.15

🏆 Mejor iteración: 45
   Recompensa: -85.32

📊 Mejor promedio móvil (10 iter): Iteración 43
   Recompensa: -88.54

📈 Mejora total: 58.30 (+32.3%)

============================================================

🎨 Creando visualizaciones...
✅ Visualización guardada en: logs/MAPPO_20251101_143000/detailed_analysis.png
```

## Futuros Scripts

### `compare_experiments.py` (Próximamente)

Compara múltiples entrenamientos lado a lado.

### `find_best_checkpoint.py` (Próximamente)

Identifica automáticamente el mejor checkpoint basado en métricas.

### `export_to_tensorboard.py` (Próximamente)

Convierte los logs a formato TensorBoard para visualización interactiva.

