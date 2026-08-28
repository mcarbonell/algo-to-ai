# 📋 Syllabus Detallado: De Algoritmista a AI Researcher

Este documento define el temario exhaustivo, los objetivos de aprendizaje por módulo, los proyectos hito y los papers seminales de lectura obligatoria y recomendada.

Sigue una metodología de **desarrollo en espiral**:
* `[x]` Completado (teoría, código from-scratch, notebook interactivo y retos)
* `[/]` En progreso (borrador / estructura activa)
* `[ ]` Planificado (esqueleto definido)

---

## 🗺️ Índice General y Tracker de Progreso

- [x] [Módulo 00: Fundamentos y Pensamiento Tensorial para Algoritmistas](#módulo-00-fundamentos-y-pensamiento-tensorial-para-algoritmistas) `[x]`
- [x] [Módulo 01: Machine Learning Estadístico & Optimización Convexa](#módulo-01-machine-learning-estadístico--optimización-convexa) `[x]`
- [x] [Módulo 02: Deep Learning "From Scratch" & El Grafo Computacional](#módulo-02-deep-learning-from-scratch--el-grafo-computacional) `[x]`
- [ ] [Módulo 03: Arquitecturas Especializadas: Visión y Secuencias](#módulo-03-arquitecturas-especializadas-visión-y-secuencias) `[ ]`
- [ ] [Módulo 04: La Revolución del Transformer y la Atención](#módulo-04-la-revolución-del-transformer-y-la-atención) `[ ]`
- [ ] [Módulo 05: Modelos Generativos y Espacios Latentes](#módulo-05-modelos-generativos-y-espacios-latentes) `[ ]`
- [ ] [Módulo 06: Post-Training, Fine-Tuning y el Arte del LLM](#módulo-06-post-training-fine-tuning-y-el-arte-del-llm) `[ ]`
- [ ] [Módulo 07: Sistemas, Inferencia Eficiente y Cuantización](#módulo-07-sistemas-inferencia-eficiente-y-cuantización) `[ ]`

---

## Módulo 00: Fundamentos y Pensamiento Tensorial para Algoritmistas

> **Objetivo:** Desmitificar los tensores. Dejar de verlos como listas anidadas de Python y entenderlos como buffers continuos de memoria plana indexados por offsets calculados mediante strides y shapes.

* **1. Contexto Histórico:**
  * La evolución del cálculo numérico: de Fortran (BLAS, LAPACK) a NumPy y los aceleradores SIMD/GPU.
  * El cuello de botella de Von Neumann y por qué la memoria plana contigua dicta el rendimiento del Deep Learning.
* **2. Intuición Algorítmica:**
  * Memoria física 1D vs Espacio N-Dimensional lógico.
  * Row-major (C-contiguous) vs Column-major (Fortran).
  * Vistas (`views`) vs Copias (`clone`/`copy`): manipulación de strides a coste $O(1)$ sin asignación de memoria.
  * La semántica de Broadcasting: expansión virtual con stride cero.
* **3. Álgebra Lineal Geométrica:**
  * Vectores como flechas y como estados; matrices como transformaciones lineales del espacio.
  * Producto escalar (dot product) como medida de proyección y alineamiento de similitud.
  * Descomposición en Valores Singulares (SVD) y reducción de rango: la base de la compresión de modelos y LoRA.
* **4. Cálculo Vectorial y Grafos:**
  * Derivadas direccionales, gradientes y matrices Jacobianas.
  * La regla de la cadena multivariable entendida como una propagación de mensajes hacia atrás en un Grafo Dirigido Acíclico (DAG).
* **🎯 Proyecto Hito:** Implementar una clase `MiniTensor` en NumPy/Python puro que soporte almacenamiento plano, cálculo dinámico de strides, transposición por swapping de strides y broadcasting sin duplicar memoria.
* **📚 Papers & Lecturas Seminales:**
  * *The NumPy Array: A Structure for Efficient Numerical Computation* (van der Walt et al., 2011).
  * *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (Williams et al., 2009).

---

## Módulo 01: Machine Learning Estadístico & Optimización Convexa

> **Objetivo:** Comprender la optimización continua como alternativa a la optimización combinatoria clásica, y dominar los fundamentos estadísticos que previenen el sobreajuste.

* **Notebooks del Módulo:**
  * `[x]` `01_gradient_descent_and_linear_models.ipynb`: Perceptrón, Descenso de Gradiente, Regresión Lineal y Logística.
  * `[x]` `02_regularization_geometry.ipynb`: Geometría de $L_1$ Lasso vs $L_2$ Ridge & Weight Decay.
  * `[x]` `03_decision_trees_to_xgboost.ipynb`: Árboles y Gradient Boosted Trees (XGBoost from scratch).
  * `[x]` `04_generalization_and_evaluation.ipynb`: Sesgo-Varianza, PAC Learning y Métricas.

* **1. Contexto Histórico:**
  * El Perceptrón de Rosenblatt (1958) y la prueba de Minsky & Papert (1969) sobre la incapacidad de resolver XOR (el primer invierno de la IA).
  * El auge del Machine Learning estadístico (Vapnik, Breiman, Freund & Schapire en los 90s).
* **2. Modelos Lineales y el Paisaje Convexo:**
  * Regresión lineal: Derivación de la ecuación normal en forma cerrada vs Descenso de Gradiente analítico.
  * Regresión logística: Del odds ratio a la sigmoide y la minimización de la entropía cruzada binaria.
* **3. Regularización y Geometría de Restricciones:**
  * Regularización $L_2$ (Ridge / Weight Decay): contracción esférica de pesos.
  * Regularización $L_1$ (Lasso): esquinas en el poliedro $L_1$ y por qué produce dispersión (*sparsity*).
* **4. Árboles de Decisión y Ensembles:**
  * Árboles como particiones ortogonales del espacio de características (entropía e impureza de Gini).
  * Bagging (Random Forests) para reducir la varianza.
  * Gradient Boosted Trees (XGBoost, LightGBM): optimización en el espacio de funciones mediante gradientes de segundo orden (Taylor).
* **5. Teoría de Generalización:**
  * Trade-off Sesgo vs Varianza.
  * Fundamentos de Aprendizaje Probablemente Aproximadamente Correcto (PAC Learning).
* **🎯 Proyecto Hito:** Implementar un clasificador de Gradient Boosting desde cero con árboles de regresión simples y comparar su frontera de decisión con una regresión logística regularizada.
* **📚 Papers & Lecturas Seminales:**
  * *Random Forests* (Breiman, 2001).
  * *XGBoost: A Scalable Tree Boosting System* (Chen & Guestrin, 2016).

---

## Módulo 02: Deep Learning "From Scratch" & El Grafo Computacional

> **Objetivo:** Construir desde cero un motor de diferenciación automática (Autograd), derivar el backpropagation matricial y entender a nivel de física computacional cómo los optimizadores navegan superficies de pérdida complejas.

* **Notebooks del Módulo:**
  * `[x]` `01_autograd_engine.ipynb`: Construcción de un motor DAG dinámico tipo Micrograd con ordenación topológica y operadores no lineales.
  * `[x]` `02_vectorized_mlp_backprop.ipynb`: Red neuronal densa multicapa con propagación vectorial hacia atrás paso a paso.
  * `[x]` `03_the_optimizer_zoo.ipynb`: De SGD puro a Momentum, RMSProp, Adam y AdamW (física y dinámica de convergencia).
  * `[x]` `04_initialization_and_normalization.ipynb`: Xavier/He initialization, BatchNorm, LayerNorm y RMSNorm.

* **1. Contexto Histórico:**
  * El redescubrimiento de Backpropagation (Rumelhart, Hinton & Williams, 1986).
  * La crisis del gradiente desvaneciente (*vanishing gradient*) y por qué las redes profundas no funcionaban hasta la década de 2010.
* **2. El Grafo de Cómputo (Autograd Engine):**
  * Topología del grafo: nodos de variables y operadores.
  * Ordenamiento topológico dinámico (DFS/Kahn) para la ejecución en reversa.
  * Construcción de un motor escalar tipo Micrograd extendido a tensores.
* **3. Multilayer Perceptron (MLP) y Backpropagation Vectorial:**
  * Derivación formal y paso a paso de $\frac{\partial \mathcal{L}}{\partial W}$ y $\frac{\partial \mathcal{L}}{\partial X}$ para capas densas.
  * Funciones de activación: ReLU, LeakyReLU, GELU, Swish/SiLU (por qué mataron a la sigmoide).
* **4. El Zoo de Optimizadores y sus Dinámicas Físicas:**
  * SGD puro y su problema en valles estrechos (oscilaciones ortogonales).
  * SGD con Momentum y aceleración de Nesterov (simulando una bola con masa física).
  * Métodos adaptativos de tasa de aprendizaje: AdaGrad (acumulador cuadrático), RMSProp (media móvil exponencial) y Adam.
  * AdamW (Loshchilov & Hutter): La corrección crucial del acoplamiento incorrecto entre weight decay y momentos de gradiente.
* **5. Estabilización y Dinámica de Activaciones:**
  * Inicialización de pesos: Xavier/Glorot y He/Kaiming (preservación de la varianza en propagación directa y reversa).
  * Normalización: Batch Normalization vs Layer Normalization vs RMSNorm.
* **🎯 Proyecto Hito:** Construir `NanoAutograd`: un motor con soporte de tensores, backpropagation automático y una suite de optimizadores (SGD, Momentum, AdamW) que entrene un MLP en espirales no lineales sin usar PyTorch.
* **📚 Papers & Lecturas Seminales:**
  * *Learning representations by back-propagating errors* (Rumelhart, Hinton, Williams, 1986).
  * *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet* (He et al., 2015).
  * *Decoupled Weight Decay Regularization (AdamW)* (Loshchilov & Hutter, 2019).

---

## Módulo 03: Arquitecturas Especializadas: Visión y Secuencias

> **Objetivo:** Comprender cómo codificar sesgos inductivos (*inductive biases*) en la arquitectura: equivariancia traslacional en imágenes y causalidad temporal en secuencias.

* **1. Contexto Histórico:**
  * Del Neocognitron de Fukushima (1980) y LeNet-5 (1998) al hito que cambió todo: AlexNet (Krizhevsky et al., 2012).
  * El auge y los dolores de cabeza de las redes recurrentes (Hochreiter & Schmidhuber, 1997 con LSTM).
* **2. Redes Convolucionales (CNNs):**
  * La operación de convolución 2D: filtros, kernels, padding, stride y campos receptivos (*receptive fields*).
  * Implementación eficiente mediante `im2col` (transformar convoluciones en multiplicación matricial GEMM estándar).
  * Pooling, feature maps y jerarquía de características visuales (de bordes a semántica de alto nivel).
* **3. El Problema de la Profundidad y las Conexiones Residuales:**
  * ¿Por qué 50 capas funcionaban peor que 20? El colapso del flujo de gradiente.
  * ResNet (He et al., 2015): La formulación residual $F(x) + x$ como una autopista para la propagación del gradiente.
* **4. Modelado Secuencial Clásico y sus Límites:**
  * RNN básica: paso temporal $h_t = f(W h_{t-1} + U x_t)$.
  * Backpropagation Through Time (BPTT) y gradientes que explotan o se desvanecen.
  * LSTM y GRU: puertas de olvido, entrada y salida para preservar memoria a largo plazo.
  * El cuello de botella insalvable: la naturaleza secuencial $O(T)$ impide la paralelización masiva en GPUs modernas.
* **🎯 Proyecto Hito:** Implementar la convolución 2D vía `im2col` puro y construir un mini-ResNet que clasifique imágenes sin recurrir a las primitivas C++ de convolución de PyTorch.
* **📚 Papers & Lecturas Seminales:**
  * *Gradient-Based Learning Applied to Document Recognition (LeNet)* (LeCun et al., 1998).
  * *Deep Residual Learning for Image Recognition (ResNet)* (He et al., 2015).
  * *Long Short-Term Memory* (Hochreiter & Schmidhuber, 1997).

---

## Módulo 04: La Revolución del Transformer y la Atención

> **Objetivo:** Dominar la arquitectura dominante del estado del arte actual: entender el mecanismo de atención como una memoria asociativa diferenciable, construir un Transformer completo estilo GPT y desentrañar los secretos de la inferencia moderna (KV-Cache, RoPE).

* **1. Contexto Histórico:**
  * El cuello de botella del vector de contexto fijo en Seq2Seq (Sutskever et al., 2014).
  * Atención blanda (*Soft Attention*): Bahdanau et al. (2014) y Luong (2015).
  * El paper que rompió el paradigma: *"Attention Is All You Need"* (Vaswani et al., 2017).
* **2. Desglosando el Mecanismo de Atención:**
  * Queries, Keys, Values: La analogía con una base de datos asociativa o tabla hash diferenciable.
  * Scaled Dot-Product Attention: $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$. Por qué el factor de escala $\sqrt{d_k}$ evita la saturación de gradientes.
  * Multi-Head Attention: Proyecciones en múltiples subespacios de representación.
  * Máscara causal: Garantizar que la posición $i$ no pueda mirar al futuro $j > i$.
* **3. Positional Embeddings y Geometría de Secuencia:**
  * Por qué la atención pura es invariante a permutaciones (*set function*).
  * Encodings sinusoidales absolutos (Vaswani 2017).
  * Rotary Position Embedding (RoPE - Su et al., 2021): Rotación de pares ortogonales en el plano complejo y propiedades de distancia relativa.
* **4. Arquitectura del Decodificador Autoregresivo (Estilo GPT / LLaMA):**
  * Tokenización: De caracteres y palabras a Byte-Pair Encoding (BPE) y SentencePiece.
  * Estructura del bloque Transformer: Pre-LayerNorm / RMSNorm vs Post-LayerNorm.
  * Capa Feed-Forward (FFN): Expansión a $4d$ y variantes modernas (SwiGLU).
* **5. Inferencia Eficiente y KV-Cache:**
  * El problema de la complejidad cuadrática $O(T^2)$ en inferencia autorregresiva naive.
  * La memoria caché de llaves y valores (KV-Cache): almacenamiento y actualización incremental paso a paso.
* **🎯 Proyecto Hito:** Construir `NanoGPT` completo desde cero en un único notebook limpio: tokenizador BPE básico, bloque Transformer con RoPE y KV-Cache, y un bucle de generación de texto interactivo con muestreo de temperatura y Top-p.
* **📚 Papers & Lecturas Seminales:**
  * *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau et al., 2014).
  * *Attention Is All You Need* (Vaswani et al., 2017).
  * *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021).
  * *GLU Variants Improve Transformer (SwiGLU)* (Shazeer, 2020).

---

## Módulo 05: Modelos Generativos y Espacios Latentes

> **Objetivo:** Pasar del modelado discriminativo a la modelización de densidades de probabilidad complejas y síntesis de datos en espacios continuos (imágenes, audio, representaciones multimodales).

* **1. Contexto Histórico:**
  * El dilema generativo: estimación de verosimilitud exacta vs aproximaciones.
  * Autoencoders estándar y por qué su espacio latente discontinuo no sirve para muestrear.
  * VAEs (Kingma & Welling, 2013), GANs (Goodfellow, 2014) y el salto cualitativo hacia la física estadística con modelos de difusión (2015-2020).
* **2. Variational Autoencoders (VAEs):**
  * Inferencia variacional y la cota inferior de evidencia (ELBO).
  * El *Reparameterization Trick*: hacer diferenciable una operación de muestreo estocástico ($z = \mu + \sigma \odot \epsilon$).
  * Divergencia KL como fuerza de regularización hacia una distribución normal estándar $\mathcal{N}(0, I)$.
* **3. Modelos de Difusión (DDPM y Flow Matching):**
  * Intuición física: La termodinámica de no equilibrio y la destrucción de información por difusión de calor/ruido.
  * Proceso hacia adelante (*Forward Process*): Añadir ruido gaussiano progresivo según un schedule $\beta_t$.
  * Proceso inverso (*Reverse Process*): Entrenar una red neuronal para predecir el ruido residual $\epsilon_\theta(x_t, t)$.
  * Muestreo guiado (Classifier-Free Guidance - CFG): Controlar la fidelidad frente a la diversidad mediante interpolación de condicionamiento.
* **🎯 Proyecto Hito:** Implementar un VAE con visualización interactiva 2D del espacio latente (interpolando entre dígitos generados) y un mini-modelo de difusión (DDPM) en juguete 1D/2D para observar visualmente cómo el ruido se desvanece en datos estructurados.
* **📚 Papers & Lecturas Seminales:**
  * *Auto-Encoding Variational Bayes* (Kingma & Welling, 2013).
  * *Denoising Diffusion Probabilistic Models (DDPM)* (Ho, Jain, Abbeel, 2020).
  * *Classifier-Free Diffusion Guidance* (Ho & Salimans, 2022).

---

## Módulo 06: Post-Training, Fine-Tuning y el Arte del LLM

> **Objetivo:** Aprender cómo un modelo base de lenguaje (un predictor de siguiente token) se transforma en un asistente seguro, alineado con preferencias humanas y capaz de razonamiento deliberativo.

* **1. Contexto Histórico:**
  * De GPT-3 a InstructGPT (Ouyang et al., 2022): El descubrimiento de que el pre-entrenamiento da conocimiento, pero el post-entrenamiento da usabilidad y alineamiento.
  * La democratización del código abierto con LLaMA (Touvron et al., 2023) y la evolución hacia modelos de razonamiento (DeepSeek-R1, OpenAI o1).
* **2. Supervised Fine-Tuning (SFT):**
  * Preparación de datasets conversacionales (formato ChatML, roles `system`, `user`, `assistant`).
  * Máscaras de pérdida en el cálculo de Cross-Entropy: entrenar únicamente sobre las respuestas del asistente, ignorando los tokens del prompt.
* **3. Fine-Tuning Paramétrico Eficiente (PEFT):**
  * El problema de costo al actualizar miles de millones de pesos.
  * LoRA (Low-Rank Adaptation - Hu et al., 2021): La hipótesis del rango intrínseco. Descomposición $W + \Delta W = W + \frac{\alpha}{r}(B \cdot A)$ con $r \ll d$.
  * Por qué LoRA añade cero latencia en inferencia (fusión de pesos $W' = W + \Delta W$).
  * QLoRA (Dettmers et al., 2023): Cuantización del modelo base a NormalFloat4 (NF4) y adaptadores en precisión 16-bit.
* **4. Alineamiento de Preferencias:**
  * El pipeline clásico de RLHF: Modelo de recompensa (Reward Model) + PPO (Proximal Policy Optimization).
  * Direct Preference Optimization (DPO - Rafailov et al., 2023): Derivación matemática elegante para optimizar preferencias humanas directamente sobre la política sin entrenar un modelo de recompensa separado ni usar RL complejo.
* **5. Razonamiento y Test-Time Compute:**
  * De Chain-of-Thought (CoT) prompting a modelos entrenados con trazas de pensamiento deliberativo.
  * Búsqueda en inferencia (Best-of-N, Monte Carlo Tree Search básico).
* **🎯 Proyecto Hito:** Implementar una capa LoRA desde cero y un bucle de entrenamiento DPO mínimo para alinear un modelo pequeño hacia respuestas más concisas o estructuradas.
* **📚 Papers & Lecturas Seminales:**
  * *Training language models to follow instructions with human feedback (InstructGPT)* (Ouyang et al., 2022).
  * *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021).
  * *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov et al., 2023).

---

## Módulo 07: Sistemas, Inferencia Eficiente y Cuantización

> **Objetivo:** Cerrar la brecha entre el modelo teórico y la realidad del silicio: entender el muro de la memoria (*memory wall*), la aritmética de baja precisión y cómo ejecutar modelos modernos con latencia mínima en hardware accesible.

* **1. Contexto Histórico:**
  * La Ley de Moore para transistores vs la Ley de amortización del ancho de banda de memoria (HBM/DRAM).
  * La revolución de la inferencia local: Georgi Gerganov y el ecosistema `llama.cpp` / GGUF.
* **2. La Física de la Inferencia: Memory-Bound vs Compute-Bound:**
  * El modelo Roofline aplicado a LLMs: Por qué la generación token a token está limitada por el ancho de banda de lectura de pesos, no por los FLOPS de la GPU.
  * Prefill (Compute-Bound, $O(T^2)$ paralelizable) vs Decode (Memory-Bound, $O(1)$ secuencial).
* **3. Algoritmos de Atención con Conciencia de Hardware:**
  * FlashAttention (Dao et al., 2022): La matemática del tiling y el recomputing en la memoria rápida local (SRAM) para evitar lecturas/escrituras continuas a HBM.
* **4. Cuantización de Pesos y Activaciones:**
  * Representaciones numéricas: FP32, FP16, BF16, INT8, FP8 y tipos sintéticos INT4.
  * Cuantización Post-Entrenamiento (PTQ): Escala ($s$) y Zero-point ($z$) simétrico vs asimétrico.
  * Técnicas avanzadas: GPTQ (cuantización de segundo orden con la inversa de la Hessiana) y AWQ (Activation-aware Weight Quantization: proteger el 1% de pesos sobresalientes).
* **5. Formatos y Ecosistemas de Inferencia:**
  * Exportación a ONNX Runtime y optimización de grafos (fusión de operadores).
  * El formato GGUF: estructura de archivos unificada, metadatos y mapeo de memoria directa (`mmap`).
* **🎯 Proyecto Hito:** Implementar un cuantizador simétrico y asimétrico INT8/INT4 desde cero con análisis de error de reconstrucción (degradación de perplejidad) y medir el ahorro de memoria y aceleración en CPU.
* **📚 Papers & Lecturas Seminales:**
  * *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., 2022).
  * *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (Lin et al., 2023).
  * *The llama.cpp Architecture & GGUF Specification* (Gerganov et al.).
