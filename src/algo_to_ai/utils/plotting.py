"""
Utilidades de visualización gráfica para el curso algo-to-ai.
"""

from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_landscape(
    loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    w1_range: Tuple[float, float] = (-3.0, 3.0),
    w2_range: Tuple[float, float] = (-3.0, 3.0),
    num_points: int = 100,
    trajectory: Optional[np.ndarray] = None,
    title: str = "Paisaje de Pérdida 2D (Loss Landscape)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Genera una vista en mapa de contornos de una función de pérdida 2D,
    opcionalmente sobreponiendo la trayectoria de optimización.
    """
    w1 = np.linspace(w1_range[0], w1_range[1], num_points)
    w2 = np.linspace(w2_range[0], w2_range[1], num_points)
    W1, W2 = np.meshgrid(w1, w2)
    Z = loss_fn(W1, W2)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(W1, W2, Z, levels=50, cmap="viridis", alpha=0.85)
    ax.contour(W1, W2, Z, levels=20, colors="white", alpha=0.3, linewidths=0.5)
    fig.colorbar(contour, ax=ax, label="Pérdida L(w1, w2)")

    if trajectory is not None and len(trajectory) > 0:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="red",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Trayectoria de optimización",
        )
        ax.plot(
            trajectory[0, 0],
            trajectory[0, 1],
            "go",
            markersize=8,
            label="Inicio (w0)",
        )
        ax.plot(
            trajectory[-1, 0],
            trajectory[-1, 1],
            "r*",
            markersize=12,
            label="Fin (w*)",
        )
        ax.legend()

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Parámetro $w_1$")
    ax.set_ylabel("Parámetro $w_2$")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig, ax


def plot_tensor_strides(
    shape: Tuple[int, int],
    strides: Tuple[int, int],
    title: str = "Mapeo de Layout de Memoria y Strides",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualiza una matriz lógica 2D y el offset en el buffer plano de memoria física
    calculado según sus strides: offset = i * stride[0] + j * stride[1].
    """
    rows, cols = shape
    grid = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            grid[i, j] = i * strides[0] + j * strides[1]

    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(grid, cmap="Blues", aspect="auto")

    for i in range(rows):
        for j in range(cols):
            val = grid[i, j]
            ax.text(
                j,
                i,
                f"[{i},{j}]\nidx: {val}",
                ha="center",
                va="center",
                color="white" if val > grid.max() / 2 else "black",
                fontweight="bold",
                fontsize=9,
            )

    fig.colorbar(cax, ax=ax, label="Offset en buffer plano (1D)")
    ax.set_title(f"{title}\nShape={shape}, Strides={strides}", fontsize=11)
    ax.set_xlabel("Dimensión 1 (Columnas)")
    ax.set_ylabel("Dimensión 0 (Filas)")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    return fig, ax
