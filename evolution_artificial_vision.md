Aquí tienes un seminario estructurado sobre la evolución de la **Visión Artificial moderna**: desde sus raíces biológicas y los primeros filtros manuales hasta la era de los Vision Transformers (ViT) y los modelos multimodales contemporáneos.

---

# Seminario: De los Píxeles a los Conceptos
### *Historia, Arquitectura y Estado del Arte en Visión Artificial*

---

## Prólogo: La pesadilla del *Feature Engineering* (Pre-2012)

Durante décadas, el problema de la visión artificial no fue la falta de algoritmos de clasificación, sino **cómo describir una imagen numéricamente**.

1. **La inspiración biológica (Hubel & Wiesel, 1959):**
   Descubrieron que la corteza visual de los mamíferos procesa información en capas jerárquicas: neuronas simples detectan bordes con orientaciones específicas; neuronas complejas combinan bordes en formas; áreas superiores reconocen objetos.
2. **Los métodos clásicos (años 90 y 2000):**
   Los ingenieros diseñaban a mano extractores de características (*hand-crafted features*):
   * **SIFT** (*Scale-Invariant Feature Transform*) y **SURF**: para encontrar puntos clave invariantes a escala y rotación.
   * **HOG** (*Histogram of Oriented Gradients*): para capturar siluetas (muy usado para peatones).
   * **Viola-Jones** (filtros tipo Haar): el mítico algoritmo que permitía a las cámaras digitales detectar caras en tiempo real (2001).
3. **El cuello de botella:**
   Tras extraer esas características a mano, se entrenaba un clasificador clásico (como un SVM o *Random Forest*). Si cambiaba la iluminación, la postura o el ángulo de cámara, el descriptor fallaba. El sistema no aprendía **qué** mirar, dependía del ingenio del matemático.

---

## Acto I: Yann LeCun y las Convoluciones (1989 – 1998)

Efectivamente, **Yann LeCun** (investigador francés formado en París y discípulo de Hinton en Toronto) sentó las bases de la visión moderna en los laboratorios Bell de AT&T.

Inspirado por el *Neocognitron* de Kunihiko Fukushima (1980), LeCun combinó tres ideas fundamentales en su arquitectura **LeNet-5 (1998)**:

* **Campos receptivos locales (*Local Receptive Fields*):** En lugar de conectar cada píxel a cada neurona (lo que provocaría millones de pesos y destruiría la geometría 2D), una neurona solo mira una pequeña ventana local (por ejemplo, $5\times 5$).
* **Pesos compartidos (*Weight Sharing*):** El mismo filtro (o *kernel*) se desliza por toda la imagen. Esto aporta **invarianza a la traslación**: un gato en la esquina superior izquierda se detecta con el mismo filtro que un gato en el centro.
* **Submuestreo (*Pooling*):** Reduce la resolución espacial conservando las características dominantes, haciendo al modelo más robusto a pequeñas variaciones y reduciendo el cómputo.

> **¿Por qué se estancó durante 14 años?**  
> LeNet-5 funcionó de maravilla para leer cheques y códigos postales (el dataset MNIST). Sin embargo, no escalaba a imágenes del mundo real en alta resolución. Faltaban dos cosas críticas: **datos a escala masiva** y **potencia de cálculo masiva**.

---

## Acto II: El Big Bang de 2012 — AlexNet y la revolución profunda

En 2012 se produjo el punto de inflexión de toda la inteligencia artificial moderna en el reto anual **ImageNet (ILSVRC)** (un dataset de 1,2 millones de imágenes y 1.000 clases creado por Fei-Fei Li).

**Alex Krizhevsky, Ilya Sutskever y Geoffrey Hinton** presentaron **AlexNet**:
* Redujeron el error de clasificación del **26%** (mejor método clásico) al **15.3%**, una brecha histórica que dejó obsoletos los métodos tradicionales en una sola tarde.

### Las tres claves de AlexNet:
1. **GPUs de consumo (NVIDIA GeForce GTX 580):** Alex Krizhevsky escribió código CUDA personalizado para entrenar la red en dos tarjetas gráficas en paralelo.
2. **Funciones de activación ReLU ($f(x) = \max(0, x)$):** Reemplazaron a la sigmoide y tanh, evitando el problema del desvanecimiento del gradiente (*vanishing gradient*) y acelerando el entrenamiento 6 veces.
3. **Regularización por Dropout:** Evitó que una red de 60 millones de parámetros memorizara el dataset (*overfitting*).

### La era dorada de las CNNs (2014 – 2017):
* **VGGNet (2014):** Demostró que apilar filtros muy pequeños ($3\times 3$) uno detrás de otro es mucho más expresivo y eficiente que usar filtros grandes ($7\times 7$ u $11\times 11$).
* **GoogLeNet / Inception (2014):** Introdujo convoluciones multiescala en paralelo y las convoluciones $1\times 1$ para reducir la dimensionalidad de canales.
* **ResNet (Kaiming He et al., 2015):** **El hito arquitectónico clave.** Hasta entonces, a partir de 20-30 capas las redes dejaban de aprender debido a la degradación. He introdujo las **conexiones residuales (*skip connections*)**:
  $$y = \mathcal{F}(x) + x$$
  Permitiendo que el gradiente fluya directamente hacia atrás. Con esto lograron entrenar redes de **152 capas**, superando por primera vez la precisión humana en ImageNet (error top-5 por debajo del 3.6%).

---

## Acto III: Más allá de clasificar — Detección y Segmentación

Clasificar ("hay un perro en la foto") era solo el primer paso. El mundo real requería localizarlo espacialmente.

```
+-------------------+--------------------+------------------------+
| Clasificación     | Detección          | Segmentación           |
| "¿Qué hay?"       | "¿Qué y dónde?"    | "¿A qué píxel pertenece?"|
|   [ Gato ]        |  [ Bounding Box ]  |    [ Máscara píxel ]   |
+-------------------+--------------------+------------------------+
```

### 1. Detección de Objetos (Object Detection):
* **El paradigma en dos etapas (*Two-Stage*):**
  * **R-CNN $\to$ Fast R-CNN $\to$ Faster R-CNN (2015):** Primero una red propone regiones de interés (*Region Proposal Network, RPN*), y luego otra las clasifica y ajusta la caja. Muy preciso, pero lento (pocos fotogramas por segundo).
* **El paradigma en una etapa (*One-Stage / Real-Time*):**
  * **YOLO (*You Only Look Once*, Joseph Redmon, 2015):** Planteó la detección no como un pipeline por fases, sino como un **único problema de regresión directo**. Divide la imagen en una cuadrícula ($S \times S$) y predice cajas y probabilidades simultáneamente en una sola pasada.
  * Alcanzó más de 45 FPS en GPU, permitiendo visión en tiempo real en robótica, coches autónomos y cámaras de seguridad. La saga YOLO continúa evolucionando hoy (v8, v9, v10, v11) refinando velocidad y precisión.

### 2. Segmentación:
* **Segmentación Semántica:** Clasificar cada píxel con una etiqueta (cielo, carretera, peatón), sin distinguir individuos.
  * **FCN** (*Fully Convolutional Networks*) y **U-Net** (2015, Ronneberger): Su estructura en forma de "U" con conexiones de salto entre capas de codificación y decodificación se convirtió en el estándar médico... y años después en el núcleo de los modelos de difusión (Stable Diffusion).
* **Segmentación por Instancias:** Distinguir cada individuo por separado (Peatón 1 vs Peatón 2).
  * **Mask R-CNN** (2017): Extendió Faster R-CNN añadiendo una rama paralela que predice la máscara binaria a nivel de píxel para cada objeto detectado.

---

## Acto IV: La invasión de los Transformers — De NLP a ViT (2020)

En 2017 nació el mecanismo de autoatención (*Attention is All You Need*) y arrasó en NLP (BERT, GPT). Pero la comunidad de visión se preguntaba: *¿Puede un Transformer procesar una imagen sin convoluciones?*

El problema computacional era que la autoatención tiene complejidad cuadrática $\mathcal{O}(N^2)$. Si una imagen tiene $1000 \times 1000$ píxeles, son $10^6$ elementos; calcular la matriz de atención entre cada par de píxeles exigiría un billón de operaciones por capa, inviable.

### El salto: Vision Transformer (ViT, Dosovitskiy et al. / Google Brain, 2020)
El artículo se tituló elocuentemente: *"An Image is Worth 16x16 Words"*.

1. **La receta de ViT:**
   * Trocea la imagen en parches fijos (típicamente de $16 \times 16$ píxeles).
   * Aplana cada parche y lo proyecta linealmente a un vector (*patch embedding*).
   * Añade un vector de posición (*positional embedding*) para que la red sepa dónde estaba cada parche.
   * Pasa esa secuencia de vectores directamente por un Transformer Encoder estándar, exactamente igual que si fueran tokens de palabras.

```
  Imagen (224x224) 
       │
  [ Troceado en parches de 16x16 ] ──► 196 "parches"
       │
  [ Proyección lineal + Positional Embeddings ]
       │
  [ Transformer Encoder (Auto-atención global) ]
       │
  [ Clasificación final / Representación latente ]
```

### El choque conceptual: CNN vs ViT
| Propiedad | Convoluciones (CNN) | Vision Transformers (ViT) |
| :--- | :--- | :--- |
| **Sesgo inductivo (*Inductive bias*)** | **Muy alto:** Asume de partida que píxeles cercanos están relacionados y que los patrones se repiten en el espacio. | **Casi nulo:** No sabe nada de geometría ni 2D; debe aprender las relaciones espaciales desde cero. |
| **Comportamiento con pocos datos** | Rinde muy bien gracias a su sesgo inductivo. | Rinde peor / sobreajusta con facilidad. |
| **Comportamiento con datos masivos** | Tiende a saturar (*plateau* de rendimiento). | **Escala continuamente**; a mayor cómputo y datos, mayor precisión. |
| **Campo receptivo** | Crece gradualmente capa a capa. | **Global desde la capa 1:** El píxel superior izquierdo puede interactuar con el inferior derecho inmediatamente. |

*Para solucionar la ineficiencia de ViT en tareas de alta resolución (detección y segmentación denso), surgieron variantes como **Swin Transformer** (2021), que aplica atención en ventanas locales desplazadas (*shifted windows*).*

---

## Acto V: El Estado del Arte Actual (2023 – Presente)

Hoy en día, la visión artificial ya no se entrena únicamente clasificando etiquetas supervisadas de ImageNet. El paradigma ha cambiado hacia tres pilares:

### 1. Modelos Fundacionales de Segmentación (SAM)
* **SAM (*Segment Anything Model*, Meta AI, 2023):**
  Un modelo entrenado con más de 1.000 millones de máscaras. Permite segmentar cualquier objeto en tiempo cero mediante *prompts*: hacer clic en un punto, arrastrar una caja delimitadora o pedirle que segmente toda la imagen en modo *zero-shot*.

### 2. Autoaprendizaje Autosupervisado (DINOv2)
* **DINO / DINOv2 (Meta AI):**
  Entrena ViTs sin etiquetas humanas (mediante destilación entre una red estudiante y una red profesora). Como resultado, la red aprende mapas de características con una comprensión semántica y geométrica tan rica que puede transferirse a estimación de profundidad 3D, correspondencia entre partes del cuerpo o segmentación sin reentrenar.

### 3. Fusión Multimodal (Vision-Language Models - VLMs)
La visión y el lenguaje se han unificado:
* **CLIP (OpenAI, 2021):** Alineó texto e imagen en el mismo espacio vectorial mediante aprendizaje contrastivo (*contrastive learning*). Es el motor que permite la búsqueda semántica de imágenes y el condicionamiento de los modelos generativos.
* **LLMs Multimodales Nativos (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5, LLaVA):**
  Ya no se trata solo de decir "gato", sino de razonar: *"Explícame qué está fallando en este diagrama de circuito"* o *"Lee esta factura manuscrita y calcula el total"*. Las arquitecturas proyectan los tokens de visión (extraídos con ViT) al espacio de embedding del modelo de lenguaje.
* **Modelos Generativos Visuales (Diffusion Transformers - DiT):**
  Herramientas como Flux o Sora abandonaron las U-Nets clásicas y adoptaron transformers de difusión sobre representaciones latentes de parches de imagen y vídeo.

---

## Resumen en Perspectiva: La Gran Lección

Si tuvieras que resumir la historia de la visión artificial en una sola idea clave (a menudo llamada *"The Bitter Lesson"* por Rich Sutton):

1. **Años 90–2000:** Intentamos programar las reglas visuales a mano (SIFT, HOG).
2. **Años 2010:** Dejamos que la red aprendiera las características, pero le impusimos una estructura espacial rígida (las Convoluciones).
3. **Años 2020:** Eliminamos casi cualquier restricción previa y dejamos que arquitecturas de propósito general (Transformers con atención pura) aprendan directamente del flujo masivo de datos y píxeles.