# 🧠 De Algoritmista a AI Researcher: Deep Learning & ML From Scratch

> **Un viaje desde los primeros principios computacionales y matemáticos hasta la frontera de la Inteligencia Artificial moderna, pensado por y para programadores.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mcarbonell/algo-to-ai/blob/main/notebooks/00_foundations/00_tensor_thinking.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)

---

## 🎯 Por qué existe este curso

La mayoría de los cursos de Inteligencia Artificial caen en uno de dos extremos:
1. **La trampa de las APIs (`import magic`):** Te enseñan a llamar a librerías de alto nivel o a endpoints de LLMs sin comprender qué pasa debajo. En cuanto algo falla (gradientes que explotan, saturación de memoria, sobreajuste silencioso), estás ciego.
2. **El formalismo desmotivado:** Libros de texto repletos de demostraciones matemáticas de páginas enteras que no conectan con cómo se estructuran las estructuras de datos en memoria, cómo se computan los grafos ni cómo se programa en la práctica.

Este repositorio nace con una premisa diferente: **utilizar la intuición algorítmica y la mentalidad de ingeniería de software como palanca para dominar el Machine Learning y el Deep Learning.**

Si entiendes punteros, grafos dirigidos acíclicos (DAGs), complejidad algorítmica ($O(N)$), layouts de memoria contigua y cómo optimizar bucles, **ya tienes el 70% de la intuición necesaria**. Solo necesitas aprender a proyectar esa intuición hacia el cálculo tensorial, la optimización continua y el diseño de arquitecturas neuronales.

---

## 🧭 Los 4 Principios del Curso

1. **🛠️ Construir para entender (First Principles):**
   Antes de usar `torch.nn.Linear`, `torch.optim.Adam` o `nn.MultiheadAttention`, los construimos desde cero con operaciones tensoriales elementales y derivación manual. Solo después pasamos a la abstracción de producción de PyTorch.
2. **📜 La Historia importa (El proceso de descubrimiento):**
   Ninguna arquitectura moderna apareció de la nada. Cada técnica (desde el Backpropagation hasta Transformers o RoPE) fue la respuesta a un fallo frustrante del método anterior. Comprender la historia te enseña a pensar como un investigador, no solo como un usuario de herramientas.
3. **👁️ Intuición Geométrica y Visual:**
   Los tensores no son solo listas anidadas de números; son transformaciones geométricas en espacios multidimensionales. Priorizamos visualizaciones interactivas de paisajes de pérdida, campos vectoriales y dinámica de activaciones.
4. **⚡ Conciencia de Hardware y Eficiencia:**
   El Deep Learning moderno está limitado por el hardware (ancho de banda de memoria y cómputo). Aprenderás a razonar sobre la memoria (SRAM vs DRAM), el cálculo de strides sin duplicar memoria, la cuantización y la optimización para hardware accesible.

---

## 📚 Mapa de Ruta y Módulos

El curso se estructura en 8 módulos progresivos. Consulta [SYLLABUS.md](SYLLABUS.md) para ver los objetivos detallados, lecturas recomendadas y el estado de desarrollo en espiral.

| Módulo | Título | Enfoque Principal | Estado |
| :--- | :--- | :--- | :---: |
| **00** | [Fundamentos & Pensamiento Tensorial](notebooks/00_foundations/) | Memoria, Strides, Broadcasting, Álgebra Lineal y Grafos Computacionales | ✅ Completado |
| **01** | [Machine Learning Clásico & Optimización](notebooks/01_classical_ml/) | Optimización continua, Regresiones, GBDT (XGBoost) y Teoría de Generalización | 🚀 Activo |
| **02** | [Deep Learning "From Scratch"](notebooks/02_dl_from_scratch/) | Motor Autograd paso a paso, Backpropagation vectorial, Optimizadores y Normalización | 📋 Esqueleto |
| **03** | [Visión y Secuencias](notebooks/03_vision_and_sequences/) | Convolución 2D (`im2col`), ResNets, Autopistas de gradiente, RNNs/LSTMs y sus límites | 📋 Esqueleto |
| **04** | [La Revolución del Transformer](notebooks/04_transformers/) | Atención por producto escalar, Multi-Head, RoPE, KV-Cache y NanoGPT completo | 📋 Esqueleto |
| **05** | [Modelos Generativos](notebooks/05_generative_models/) | Espacios latentes, Autoencoders, VAEs y Fundamentos de Difusión (DDPM) | 📋 Esqueleto |
| **06** | [Post-Training & LLM Engineering](notebooks/06_post_training/) | SFT, Fine-tuning eficiente (LoRA / QLoRA), Alineamiento (DPO) y Razonamiento (CoT) | 📋 Esqueleto |
| **07** | [Sistemas, Eficiencia y Cuantización](notebooks/07_systems_and_efficiency/) | Inferencia local, GGUF/llama.cpp, Cuantización INT4/INT8, FlashAttention y Profiling | 📋 Esqueleto |

---

## 🏛️ Anatomía de cada Capítulo

Cada notebook sigue de manera consistente esta estructura pedagógica:
1. **📜 Contexto Histórico y Descubrimiento:** ¿Qué fallaba en el estado del arte anterior y qué intuición llevó a esta solución?
2. **🧠 Intuición Geométrica para Programadores:** Explicaciones visuales y analogías computacionales.
3. **🛠️ Implementación "From Scratch":** Código puro en Python/NumPy sin cajas negras.
4. **⚡ Transición a PyTorch Moderno:** Cómo se implementa de manera idiomática, tipada y eficiente en PyTorch.
5. **🎯 Retos & Experimentos ("Tinker Time"):** Ejercicios prácticos de código para romper, arreglar y extender el modelo.
6. **📚 Referencias Fundamentales:** Papers seminales, enlaces esenciales y repositorios de referencia comentados.

---

## 🚀 Cómo empezar

### Opción 1: En la nube con Google Colab (Recomendado para empezar rápido)
Cada notebook contiene en la cabecera un botón interactivo `Open in Colab` que te permite ejecutar el código directamente con una GPU T4 o CPU gratuita, sin necesidad de configurar entornos locales.

### Opción 2: En tu máquina local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/mcarbonell/algo-to-ai.git
   cd algo-to-ai
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv .venv
   # En Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # En Linux / macOS:
   source .venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lanzar Jupyter Lab / VS Code:**
   ```bash
   jupyter lab
   ```

---

## 🤝 Contribuciones y Desarrollo en Espiral

Este proyecto sigue una metodología de **desarrollo en espiral**: primero establecemos los fundamentos y esqueletos mínimos viables, para luego iterar profundizando en matemáticas, visualizaciones interactivas y benchmarks.

¡Sugerencias, correcciones y pull requests son siempre bienvenidos!
