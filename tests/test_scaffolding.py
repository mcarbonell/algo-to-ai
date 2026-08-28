"""
Tests de verificación estructural para el repositorio algo-to-ai.
Valida la integridad de los notebooks (JSON válido), imports y lógica fundamental de MiniTensor.
"""

import json
from pathlib import Path
import pytest
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_notebooks_are_valid_json():
    """Verifica que todos los notebooks .ipynb sean archivos JSON bien formados."""
    notebooks = list(REPO_ROOT.glob("**/*.ipynb"))
    assert len(notebooks) >= 2, "Deberían existir al menos template_notebook.ipynb y 00_tensor_thinking.ipynb"

    for nb_path in notebooks:
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "cells" in data, f"El notebook {nb_path.name} no contiene el campo 'cells'"
        assert "metadata" in data, f"El notebook {nb_path.name} no contiene el campo 'metadata'"
        assert data.get("nbformat") == 4, f"El notebook {nb_path.name} debe tener nbformat == 4"


def test_package_import():
    """Verifica que el paquete base algo_to_ai se pueda importar correctamente."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    
    import algo_to_ai
    from algo_to_ai.utils import plot_loss_landscape, plot_tensor_strides
    
    assert algo_to_ai.__version__ == "0.1.0"
    assert callable(plot_loss_landscape)
    assert callable(plot_tensor_strides)


def test_minitensor_logic():
    """Valida la lógica de strides y vistas de MiniTensor."""
    # Extraer la implementación lógica para test unitario
    class MiniTensor:
        def __init__(self, data, shape, strides=None, offset=0):
            self.storage = data
            self.shape = tuple(shape)
            self.offset = offset
            if strides is None:
                strides = [1] * len(self.shape)
                for k in range(len(self.shape) - 2, -1, -1):
                    strides[k] = strides[k + 1] * self.shape[k + 1]
                self.strides = tuple(strides)
            else:
                self.strides = tuple(strides)

        def __getitem__(self, indices):
            if isinstance(indices, int):
                indices = (indices,)
            linear_idx = self.offset
            for idx, stride in zip(indices, self.strides):
                linear_idx += idx * stride
            return self.storage[linear_idx]

        def transpose(self, dim0=0, dim1=1):
            new_shape = list(self.shape)
            new_strides = list(self.strides)
            new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
            new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
            return MiniTensor(self.storage, tuple(new_shape), tuple(new_strides), self.offset)

        def broadcast_to(self, new_shape):
            diff = len(new_shape) - len(self.shape)
            current_shape = (1,) * diff + self.shape
            new_strides = [0] * diff + list(self.strides)
            final_strides = []
            for old_dim, new_dim, st in zip(current_shape, new_shape, new_strides):
                if old_dim == new_dim:
                    final_strides.append(st)
                elif old_dim == 1:
                    final_strides.append(0)
                else:
                    raise ValueError("Cannot broadcast")
            return MiniTensor(self.storage, new_shape, tuple(final_strides), self.offset)

    # 1. Matriz 2x3 básica
    data = [10, 20, 30, 40, 50, 60]
    t = MiniTensor(data, shape=(2, 3))
    assert t.shape == (2, 3)
    assert t.strides == (3, 1)
    assert t[0, 0] == 10
    assert t[1, 2] == 60

    # 2. Transposición en O(1)
    t_trans = t.transpose(0, 1)
    assert t_trans.shape == (3, 2)
    assert t_trans.strides == (1, 3)
    assert t_trans[2, 1] == 60
    assert t_trans.storage is t.storage  # Mismo buffer

    # 3. Broadcasting en O(1) con stride 0
    col = MiniTensor([100, 200], shape=(2, 1))
    broad = col.broadcast_to((2, 4))
    assert broad.shape == (2, 4)
    assert broad.strides == (1, 0)
    assert broad[0, 0] == 100
    assert broad[0, 3] == 100
    assert broad[1, 0] == 200
    assert broad[1, 3] == 200
