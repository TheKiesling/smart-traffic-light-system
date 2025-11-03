# 🚦 Sistema de Semáforos Inteligentes con MAPPO

Sistema de control de semáforos inteligentes basado en **Multi-Agent Proximal Policy Optimization (MAPPO)** con integración completa de **SUMO** (Simulation of Urban MObility). Diseñado para entrenar múltiples agentes que controlan semáforos de forma coordinada, optimizando el flujo de tráfico urbano mediante aprendizaje por refuerzo profundo con soporte GPU.

## 🎯 Características

- ✅ **Algoritmo MAPPO**: Implementación con Ray RLlib optimizada para multi-agente
- ✅ **Integración SUMO**: Usando la librería `sumo-rl` oficialmente soportada
- ✅ **Soporte GPU**: Entrenamiento acelerado con PyTorch y CUDA
- ✅ **Arquitectura Modular**: Código limpio y fácilmente extensible
- ✅ **Configuración YAML**: Configuración flexible sin modificar código
- ✅ **Visualización**: Integración con SUMO-GUI para visualizar el comportamiento aprendido
- ✅ **Logging Avanzado**: Sistema automático que guarda métricas por iteración (CSV, JSON, gráficas)
- ✅ **Restricciones de Tiempo**: Respeto obligatorio de min_green y max_green con `fixed_ts`

## 📋 Requisitos Previos

### 1. SUMO (Simulation of Urban MObility)

**Windows:**
```bash
# Descargar e instalar desde:
# https://eclipse.dev/sumo/

# Configurar variable de entorno SUMO_HOME
# Ejemplo: C:\Program Files (x86)\Eclipse\Sumo
```

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-doc

# macOS (Homebrew)
brew install sumo

# Configurar SUMO_HOME
export SUMO_HOME="/usr/share/sumo"
```

### 2. Python 3.8+

### 3. CUDA (para soporte GPU)
- NVIDIA GPU con soporte CUDA
- CUDA Toolkit 11.8 o superior
- cuDNN compatible

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd smart-light-system
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación de GPU (opcional)

```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 📁 Estructura del Proyecto

```
smart-light-system/
├── config/                     # Archivos de configuración
│   ├── sumo_config.yaml       # Configuración de SUMO
│   └── training_config.yaml   # Configuración de entrenamiento MAPPO
├── environments/              # Entornos de RL
│   └── traffic_env.py        # Environment con sumo-rl
├── models/                    # Modelos y entrenadores
│   └── mappo_trainer.py      # Entrenador MAPPO con Ray RLlib
├── scenarios/                 # Escenarios de tráfico SUMO
│   └── simple_grid/          # Intersección simple
│       ├── grid.net.xml      # Red de calles
│       ├── grid.rou.xml      # Rutas de vehículos
│       └── grid.sumocfg      # Configuración SUMO
├── utils/                     # Utilidades
│   └── helpers.py            # Funciones auxiliares
├── results/                   # Resultados de entrenamiento
├── checkpoints/               # Modelos entrenados
├── logs/                      # Logs de TensorBoard
├── main.py                    # Script principal
└── requirements.txt           # Dependencias
```

## 🎮 Uso

### Entrenamiento

```bash
# Entrenamiento básico
python main.py train

# Especificar número de iteraciones
python main.py train --iterations 500

# Con generación de gráficas
python main.py train --iterations 100 --plot

# Configuración personalizada
python main.py train --sumo-config config/sumo_config.yaml --training-config config/training_config.yaml
```

### Evaluación

```bash
# Evaluar modelo entrenado
python main.py evaluate --checkpoint checkpoints/checkpoint_000100

# Evaluar con más episodios
python main.py evaluate --checkpoint checkpoints/checkpoint_000100 --episodes 20
```

### Visualización

```bash
# Visualizar con SUMO-GUI
python main.py visualize --checkpoint checkpoints/checkpoint_000100

# Personalizar duración
python main.py visualize --checkpoint checkpoints/checkpoint_000100 --duration 1800
```

### Análisis de Métricas

```bash
# Analizar un entrenamiento específico
python scripts/analyze_training.py logs/MAPPO_20251101_143000

# Los logs se generan automáticamente durante el entrenamiento en:
# logs/MAPPO_YYYYMMDD_HHMMSS/
#   ├── metrics.csv          # Métricas en formato CSV
#   ├── metrics.json         # Métricas en formato JSON
#   ├── summary.txt          # Resumen estadístico
#   └── training_metrics.png # Gráficas de progreso
```

**Métricas guardadas por iteración:**
- 📊 Recompensa media, máxima y mínima
- ⏱️ Longitud de episodios
- 📉 Policy loss y Value function loss
- 🎲 Entropía de la política
- 🎓 Learning rate actual

Ver documentación completa: [`docs/LOGGING_SYSTEM.md`](docs/LOGGING_SYSTEM.md)

## ⚙️ Configuración

### SUMO Configuration (`config/sumo_config.yaml`)

```yaml
sumo:
  net_file: "scenarios/simple_grid/grid.net.xml"
  route_file: "scenarios/simple_grid/grid.rou.xml"
  use_gui: false              # true para visualización
  num_seconds: 3600           # Duración de la simulación
  delta_time: 5               # Segundos entre decisiones
  yellow_time: 2              # Duración de luz amarilla
  min_green: 5                # Tiempo mínimo en verde
  max_green: 60               # Tiempo máximo en verde
  reward_fn: "diff-waiting-time"  # Función de recompensa
  single_agent: false         # Multi-agente
  sumo_seed: 42               # Semilla para reproducibilidad
```

### Training Configuration (`config/training_config.yaml`)

```yaml
training:
  algorithm: "MAPPO"
  num_workers: 4              # Workers paralelos
  num_gpus: 1                 # GPUs a usar
  framework: "torch"
  
  train_batch_size: 4000
  sgd_minibatch_size: 128
  num_sgd_iter: 10
  
  lr: 0.0003                  # Learning rate
  gamma: 0.99                 # Factor de descuento
  lambda: 0.95                # GAE lambda
  clip_param: 0.2             # PPO clip
  
  model:
    fcnet_hiddens: [256, 256] # Capas ocultas
    fcnet_activation: "relu"
```

## 📊 Monitoreo

### TensorBoard

```bash
tensorboard --logdir results/
```

Accede a `http://localhost:6006` para ver:
- Recompensa por episodio
- Longitud de episodio
- Policy loss
- Value function loss
- Y más métricas

### Gráficas automáticas

El flag `--plot` genera automáticamente gráficas de entrenamiento en `results/training_metrics.png`

## 🔧 Personalización

### Crear nuevos escenarios

1. Diseña tu red con **NETEDIT** (incluido con SUMO)
2. Genera rutas con **SUMO tools**
3. Coloca archivos en `scenarios/tu_escenario/`
4. Actualiza `config/sumo_config.yaml`

### Modificar arquitectura del modelo

Edita `config/training_config.yaml`:

```yaml
model:
  fcnet_hiddens: [512, 512, 256]  # Red más profunda
  fcnet_activation: "tanh"        # Cambiar activación
  use_lstm: true                  # Agregar LSTM
  lstm_cell_size: 256
```

### Función de recompensa personalizada

Opciones disponibles en `sumo-rl`:
- `diff-waiting-time`: Diferencia en tiempo de espera
- `average-speed`: Velocidad promedio
- `queue`: Longitud de cola
- `pressure`: Presión de tráfico

## 🎓 Algoritmo MAPPO

**MAPPO (Multi-Agent PPO)** es una extensión del algoritmo PPO para entornos multi-agente:

- **Centralizado durante entrenamiento**: Usa información global para aprender mejor
- **Descentralizado durante ejecución**: Cada semáforo decide independientemente
- **Value function factorization**: Cada agente tiene su propia función de valor
- **Shared or individual policies**: Políticas compartidas o individuales

### Ventajas para semáforos:

✅ Coordinación implícita entre semáforos  
✅ Estable y sample-efficient  
✅ Escala bien a múltiples agentes  
✅ Soporte GPU para entrenamiento rápido  

## 📈 Resultados Esperados

Después del entrenamiento, deberías observar:

- ⬇️ Reducción del tiempo de espera promedio
- ⬆️ Aumento de la velocidad promedio de vehículos
- ⬇️ Disminución de longitud de colas
- ⬆️ Mejora en throughput de intersecciones

## 🐛 Solución de Problemas

### Error: "SUMO_HOME no configurado"

```bash
# Windows (PowerShell)
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"

# Linux/macOS
export SUMO_HOME="/usr/share/sumo"
```

### Error: "CUDA out of memory"

Reduce en `config/training_config.yaml`:
```yaml
train_batch_size: 2000        # Reducir batch size
num_workers: 2                # Reducir workers
model:
  fcnet_hiddens: [128, 128]   # Red más pequeña
```

### Entrenamiento lento

Ajusta:
```yaml
num_workers: 8                # Más workers si tienes CPUs
rollout_fragment_length: 100  # Fragmentos más cortos
num_gpus: 1                   # Asegurar que usa GPU
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la branch (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📚 Referencias

- **SUMO**: https://eclipse.dev/sumo/
- **sumo-rl**: https://github.com/LucasAlegre/sumo-rl
- **Ray RLlib**: https://docs.ray.io/en/latest/rllib/
- **MAPPO Paper**: https://arxiv.org/abs/2103.01955

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## ✨ Créditos

Desarrollado con:
- Ray RLlib para MAPPO
- sumo-rl para integración con SUMO
- PyTorch para redes neuronales
- SUMO para simulación de tráfico

---

**¡Feliz entrenamiento! 🚦🤖**

Para preguntas o problemas, abre un issue en GitHub.

