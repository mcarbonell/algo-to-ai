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


def test_dagnode_reverse_autograd():
    """Valida la acumulación de gradientes en reversa sobre un DAG."""
    class Node:
        def __init__(self, val, children=()):
            self.val = float(val)
            self.grad = 0.0
            self.children = set(children)
            self._backward = lambda: None

        def __add__(self, other):
            other = other if isinstance(other, Node) else Node(other)
            out = Node(self.val + other.val, (self, other))
            def _b():
                self.grad += 1.0 * out.grad
                other.grad += 1.0 * out.grad
            out._backward = _b
            return out

        def __mul__(self, other):
            other = other if isinstance(other, Node) else Node(other)
            out = Node(self.val * other.val, (self, other))
            def _b():
                self.grad += other.val * out.grad
                other.grad += self.val * out.grad
            out._backward = _b
            return out

        def backward(self):
            topo = []
            visited = set()
            def dfs(v):
                if v not in visited:
                    visited.add(v)
                    for c in v.children:
                        dfs(c)
                    topo.append(v)
            dfs(self)
            self.grad = 1.0
            for n in reversed(topo):
                n._backward()

    # L = (a * b) + a
    # dL/da = b + 1, dL/db = a
    a = Node(3.0)
    b = Node(4.0)
    prod = a * b
    L = prod + a
    L.backward()

    assert L.val == 15.0
    assert a.grad == 5.0  # b (4) + 1
    assert b.grad == 3.0  # a (3)


def test_linear_and_logistic_regression():
    """Valida la convergencia matemática de las regresiones from-scratch."""
    # 1. Regresión lineal simple
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * X[:, 0] + 1.0  # y = 2x + 1
    
    # Ecuación normal
    X_aug = np.hstack([np.ones((4, 1)), X])
    w = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
    assert np.isclose(w[0], 1.0)  # bias
    assert np.isclose(w[1], 2.0)  # slope

    # 2. Regresión logística separable
    X_log = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_log = np.array([0.0, 0.0, 1.0, 1.0])
    
    weights = np.array([0.0])
    bias = 0.0
    lr = 0.5
    for _ in range(300):
        z = X_log @ weights + bias
        y_hat = 1.0 / (1.0 + np.exp(-z))
        err = y_hat - y_log
        weights -= lr * (1.0 / len(y_log)) * (X_log.T @ err)
        bias -= lr * np.mean(err)
        
    preds = (1.0 / (1.0 + np.exp(-(X_log @ weights + bias))) >= 0.5).astype(float)
    assert np.array_equal(preds, y_log)


def test_ridge_and_lasso_regularization():
    """Valida que Ridge contraiga pesos y Lasso produzca ceros exactos (sparsity)."""
    np.random.seed(42)
    m, d = 50, 10
    X = np.random.randn(m, d)
    # Solo las 2 primeras variables tienen peso real
    w_true = np.zeros(d)
    w_true[0] = 4.0
    w_true[1] = -2.0
    y = X @ w_true + np.random.randn(m) * 0.1

    # 1. Ridge
    I = np.eye(d)
    w_ridge = np.linalg.solve(X.T @ X + 10.0 * I, X.T @ y)
    # Ridge no debe poner ceros exactos
    assert not any(np.isclose(w_ridge[2:], 0.0, atol=1e-6))
    # Pero los pesos verdaderos deben ser los dominantes
    assert abs(w_ridge[0]) > abs(w_ridge[2])

    # 2. Lasso con Coordinate Descent y Soft-Thresholding
    def soft_threshold(rho, lam):
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        return 0.0

    z = np.sum(X ** 2, axis=0)
    w_lasso = np.zeros(d)
    for _ in range(500):
        for j in range(d):
            res_j = y - (X @ w_lasso - X[:, j] * w_lasso[j])
            rho_j = np.dot(X[:, j], res_j)
            w_lasso[j] = soft_threshold(rho_j, 5.0) / z[j]

    # Lasso debe anular la mayoría de las columnas de ruido a exactamente 0.0
    num_ceros = np.sum(np.isclose(w_lasso[2:], 0.0, atol=1e-6))
    assert num_ceros >= 6, f"Se esperaban al menos 6 ceros en las 8 columnas de ruido, obtenidos {num_ceros}"
    assert w_lasso[0] > 0
    assert w_lasso[1] < 0


def test_xgboost_from_scratch():
    """Valida la fórmula de pesos y ganancia de división de Tianqi Chen (XGBoost)."""
    # 1. Validación de la fórmula de peso óptimo de hoja w* = -G / (H + lambda)
    g = np.array([-2.0, -3.0, -1.0])
    h = np.array([1.0, 1.0, 1.0])
    reg_lambda = 1.0
    G = np.sum(g)  # -6.0
    H = np.sum(h)  # 3.0
    w_star = - G / (H + reg_lambda)
    assert np.isclose(w_star, 1.5)  # -(-6) / (3 + 1) = 6 / 4 = 1.5

    # 2. Validación de fórmula de ganancia de división
    # Nodo padre
    gain_parent = (G ** 2) / (H + reg_lambda)
    # Split izquierdo: 2 muestras (-2, -3) -> G_L = -5, H_L = 2
    G_L, H_L = -5.0, 2.0
    gain_L = (G_L ** 2) / (H_L + reg_lambda)
    # Split derecho: 1 muestra (-1) -> G_R = -1, H_R = 1
    G_R, H_R = -1.0, 1.0
    gain_R = (G_R ** 2) / (H_R + reg_lambda)

    split_gain = 0.5 * (gain_L + gain_R - gain_parent)
    # gain_parent = 36 / 4 = 9.0
    # gain_L = 25 / 3 = 8.333
    # gain_R = 1 / 2 = 0.5
    # split_gain = 0.5 * (8.3333 + 0.5 - 9.0) = 0.5 * (-0.1666) < 0 (no vale la pena dividir)
    assert split_gain < 0

    # 3. Mini ensemble ajustando un escalón
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    
    # Modelo simple boosting
    y_pred = np.full(len(y), np.mean(y))  # 5.0
    for _ in range(5):
        g = y_pred - y
        h = np.ones(len(y))
        # Mejor corte en x = 5.0
        mask_left = X[:, 0] <= 5.0
        mask_right = ~mask_left
        w_L = - np.sum(g[mask_left]) / (np.sum(h[mask_left]) + 0.1)
        w_R = - np.sum(g[mask_right]) / (np.sum(h[mask_right]) + 0.1)
        y_pred[mask_left] += 0.5 * w_L
        y_pred[mask_right] += 0.5 * w_R

    # Tras 5 pasos el error cuadrático debe haber colapsado
    mse = np.mean((y_pred - y) ** 2)
    assert mse < 1.0




