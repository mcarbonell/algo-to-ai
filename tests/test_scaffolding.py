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


def test_stratified_kfold_and_roc_auc():
    """Valida la preservación de proporciones de clases en K-Fold y el cálculo de AUC."""
    # 1. Stratified K-Fold
    y = np.array([0] * 80 + [1] * 20)  # 20% clase positiva
    k = 4
    
    clases = np.unique(y)
    folds = [[] for _ in range(k)]
    for c in clases:
        idx_c = np.where(y == c)[0]
        for i, s_idx in enumerate(idx_c):
            folds[i % k].append(s_idx)
            
    # Cada uno de los 4 folds debe tener exactamente 5 muestras de la clase 1 y 20 de la clase 0
    for f in range(k):
        val_idx = np.array(folds[f])
        assert np.sum(y[val_idx] == 1) == 5
        assert np.sum(y[val_idx] == 0) == 20

    # 2. Curva ROC y AUC con clasificador perfecto
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    
    # Clasificador perfecto debe tener AUC = 1.0
    desc = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    tpr_list, fpr_list = [0.0], [0.0]
    tp, fp = 0, 0
    for l in y_sorted:
        if l == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    
    fpr_arr, tpr_arr = np.array(fpr_list), np.array(tpr_list)
    auc = float(np.sum((fpr_arr[1:] - fpr_arr[:-1]) * (tpr_arr[1:] + tpr_arr[:-1]) / 2.0))
    assert np.isclose(auc, 1.0)


def test_value_autograd_engine():
    """Valida el motor de autograd escalar Value con DFS y regla de la cadena multivariable."""
    import math

    class V:
        def __init__(self, data, children=()):
            self.data = float(data)
            self.grad = 0.0
            self._prev = set(children)
            self._backward = lambda: None

        def __add__(self, other):
            other = other if isinstance(other, V) else V(other)
            out = V(self.data + other.data, (self, other))
            def _b():
                self.grad += 1.0 * out.grad
                other.grad += 1.0 * out.grad
            out._backward = _b
            return out

        def __mul__(self, other):
            other = other if isinstance(other, V) else V(other)
            out = V(self.data * other.data, (self, other))
            def _b():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _b
            return out

        def relu(self):
            out = V(max(0.0, self.data), (self,))
            def _b():
                self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad
            out._backward = _b
            return out

        def backward(self):
            topo = []
            visited = set()
            def dfs(v):
                if v not in visited:
                    visited.add(v)
                    for c in v._prev:
                        dfs(c)
                    topo.append(v)
            dfs(self)
            self.grad = 1.0
            for n in reversed(topo):
                n._backward()

    # Expresión no lineal: L = relu(x * y) + x
    # Si x = 2.0, y = 3.0 -> prod = 6.0 > 0 -> relu = 6.0
    # L = 6.0 + 2.0 = 8.0
    # dL/dx = y + 1 = 3 + 1 = 4.0
    # dL/dy = x = 2.0
    x = V(2.0)
    y = V(3.0)
    L = (x * y).relu() + x
    L.backward()

    assert L.data == 8.0
    assert x.grad == 4.0
    assert y.grad == 2.0


def test_vectorized_linear_and_relu():
    """Valida la matemática y correspondencia dimensional de backpropagation matricial."""
    B, d_in, d_out = 4, 3, 2
    np.random.seed(42)
    X = np.random.randn(B, d_in)
    W = np.random.randn(d_in, d_out)
    b = np.zeros((1, d_out))

    # Forward
    Z = X @ W + b
    A = np.maximum(0.0, Z)

    # Simular gradiente entrante delta: (B, d_out)
    delta = np.random.randn(B, d_out)

    # Backward ReLU
    delta_Z = delta * (Z > 0.0)

    # Backward Linear
    dW = X.T @ delta_Z
    db = np.sum(delta_Z, axis=0, keepdims=True)
    dX = delta_Z @ W.T

    # Comprobar consistencia dimensional
    assert dW.shape == (d_in, d_out)
    assert db.shape == (1, d_out)
    assert dX.shape == (B, d_in)

    # Comprobación numérica: dW debe ser la suma de X[i].T @ delta_Z[i] muestra a muestra
    dW_loop = np.zeros_like(W)
    for i in range(B):
        dW_loop += np.outer(X[i], delta_Z[i])
    assert np.allclose(dW, dW_loop)


def test_optimizers_adam_and_adamw():
    """Valida la mecánica de momentos y desacoplamiento de weight decay en Adam y AdamW."""
    # 1. AdamW en 1 dimensión
    w = np.array([5.0])
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    wd = 0.05

    m = np.zeros(1)
    v = np.zeros(1)

    # Simular 10 pasos en f(w) = w^2 (gradiente = 2w)
    for t in range(1, 11):
        g = 2.0 * w
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        w = w * (1.0 - lr * wd) - (lr / (np.sqrt(v_hat) + eps)) * m_hat

    # El peso debe haber convergido significativamente hacia 0
    assert abs(w[0]) < 4.0


def test_initialization_and_normalization():
    """Valida la conservación de varianza de He/Kaiming y las estadísticas de LayerNorm y RMSNorm."""
    np.random.seed(42)
    # 1. He/Kaiming init
    n_in = 1000
    n_out = 1000
    W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    expected_var = 2.0 / n_in
    actual_var = float(np.var(W))
    assert np.isclose(actual_var, expected_var, rtol=0.1)

    # 2. LayerNorm
    X = np.random.randn(10, 50) * 5.0 + 3.0
    mu = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    X_norm = (X - mu) / np.sqrt(var + 1e-5)
    assert np.allclose(np.mean(X_norm, axis=-1), 0.0, atol=1e-5)
    assert np.allclose(np.var(X_norm, axis=-1), 1.0, atol=1e-3)

    # 3. RMSNorm
    mean_sq = np.mean(X ** 2, axis=-1, keepdims=True)
    X_rms = X / np.sqrt(mean_sq + 1e-6)
    rms_val = np.sqrt(np.mean(X_rms ** 2, axis=-1))
    assert np.allclose(rms_val, 1.0, atol=1e-3)


def test_conv2d_im2col():
    """Valida la mecánica de desenrollado im2col y la equivalencia matricial GEMM de Conv2D."""
    # 1. Imagen pequeña de 1 canal y 4x4
    X = np.array([[[[1.0, 2.0, 0.0, 1.0],
                    [0.0, 1.0, 2.0, 0.0],
                    [3.0, 0.0, 1.0, 2.0],
                    [1.0, 2.0, 0.0, 1.0]]]])  # (1, 1, 4, 4)
    # Filtro 2x2
    W = np.array([[[[1.0, 0.0],
                    [0.0, 1.0]]]])  # (1, 1, 2, 2)
    
    # Con stride=1 y padding=0, out_h = 4 - 2 + 1 = 3, out_w = 3
    # Valor en (0, 0) debe ser 1*1 + 2*0 + 0*0 + 1*1 = 2.0
    kh, kw = 2, 2
    out_h, out_w = 3, 3
    
    # im2col manual
    cols = np.zeros((1 * kh * kw, 1 * out_h * out_w))
    col_idx = 0
    for h in range(out_h):
        for w in range(out_w):
            patch = X[0, 0, h:h+kh, w:w+kw]
            cols[:, col_idx] = patch.ravel()
            col_idx += 1
            
    out_gemm = (W.reshape(1, -1) @ cols).reshape(1, 1, out_h, out_w)
    
    # Comprobar la primera ventana (0, 0): X[0:2, 0:2] = [[1, 2], [0, 1]] * [[1, 0], [0, 1]] = 1 + 1 = 2.0
    assert out_gemm[0, 0, 0, 0] == 2.0
    # Comprobar la ventana (0, 1): X[0:2, 1:3] = [[2, 0], [1, 2]] * [[1, 0], [0, 1]] = 2 + 2 = 4.0
    assert out_gemm[0, 0, 0, 1] == 4.0


def test_residual_block_gradient_highway():
    """Valida que la conexion residual actua como una autopista de gradiente matematico."""
    dim = 8
    B = 4
    x = np.random.randn(B, dim)
    
    # Bloque con pesos iniciales en cero
    W1 = np.zeros((dim, dim))
    W2 = np.zeros((dim, dim))
    
    # Forward: y = ReLU(x @ W1) @ W2 + x = 0 + x = x
    h = np.maximum(0.0, x @ W1)
    f_x = h @ W2
    y = f_x + x
    assert np.allclose(y, x)
    
    # Backward: dL/dx = dL/dy @ dF/dx + dL/dy = 0 + dL/dy = dL/dy
    grad_output = np.ones((B, dim))
    d_fx = (grad_output @ W2.T) * (h > 0) @ W1.T
    grad_input = d_fx + grad_output
    
    # El gradiente debe pasar exactamente intacto (+I)
    assert np.allclose(grad_input, grad_output)


def test_lstm_cell_constant_error_carousel():
    """Valida la autopista de memoria aditiva de la celda LSTM (Constant Error Carousel)."""
    # Si f_t = 1.0 (recordar todo) e i_t = 0.0 (no escribir nada nuevo)
    c_prev = np.array([[1.5, -2.0, 3.0]])
    f_gate = np.array([[1.0, 1.0, 1.0]])
    i_gate = np.array([[0.0, 0.0, 0.0]])
    c_tilde = np.array([[0.5, 0.5, 0.5]])
    o_gate = np.array([[1.0, 1.0, 1.0]])
    
    # C_t = f_t * C_{t-1} + i_t * C_tilde
    c_t = f_gate * c_prev + i_gate * c_tilde
    assert np.allclose(c_t, c_prev), "La memoria debe mantenerse 100% intacta a traves del tiempo"
    
    # h_t = o_t * tanh(C_t)
    h_t = o_gate * np.tanh(c_t)
    assert np.allclose(h_t, np.tanh(c_prev))


def test_scaled_dot_product_attention():
    """Valida la mecánica de atención por producto escalar y la máscara causal."""
    B, T, d_k = 2, 4, 8
    np.random.seed(42)
    Q = np.random.randn(B, T, d_k)
    K = np.random.randn(B, T, d_k)
    V = np.random.randn(B, T, d_k)

    # 1. Sin máscara
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_k)
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    out = weights @ V
    assert out.shape == (B, T, d_k)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)

    # 2. Con máscara causal
    mask = np.tril(np.ones((T, T), dtype=bool))
    scores_causal = np.where(mask, scores, -1e9)
    exp_sc = np.exp(scores_causal - np.max(scores_causal, axis=-1, keepdims=True))
    weights_causal = exp_sc / np.sum(exp_sc, axis=-1, keepdims=True)
    
    # La parte estrictamente superior de weights_causal debe ser exactamente 0
    for i in range(T):
        for j in range(i + 1, T):
            assert np.all(weights_causal[:, i, j] == 0.0)













