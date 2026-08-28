# Módulo 07: Sistemas, Inferencia Eficiente y Cuantización

Optimización de hardware, cuellos de botella de ancho de banda y despliegue local de modelos.

## Notebooks Planificados
* `01_memory_wall_and_roofline.ipynb`: El modelo Roofline, prefill (compute-bound) vs decode (memory-bound).
* `02_quantization_from_scratch.ipynb`: Cuantización simétrica y asimétrica INT8 / INT4, calibración y AWQ.
* `03_hardware_aware_attention.ipynb`: FlashAttention: conceptos de SRAM vs HBM y tiling en bloques.
* `04_local_inference_and_onnx.ipynb`: Exportación a ONNX Runtime, GGUF/llama.cpp e inferencia local eficiente en CPU/GPU integrada.
