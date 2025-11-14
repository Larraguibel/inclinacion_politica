import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def mae_por_class(y_true, y_pred):
    """
    Calcula el MAE por clase (L=-1, C=0, R=1).
    y_true : array-like de tamaño N con valores -1, 0, 1
    y_pred : array-like de tamaño N con valores continuos
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_map = {
        -1: "L",
         0: "C",
         1: "R"
    }

    results = {}

    for cls in [-1, 0, 1]:
        mask = (y_true == cls)
        if mask.sum() == 0:
            results[class_map[cls]] = None
        else:
            mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))
            results[class_map[cls]] = mae

    return results


def plot_1d_predictions(
    y_pred,
    true_labels,
    colors={"L": "red", "C": "green", "R": "blue"},
    figsize=(10, 5),
    alpha=0.6,
    s=45,
    thresholds=(-0.33, 0, 0.33),
    title="Predicciones de ideología (1D) coloreadas por clase real",
    ax=None,
):
    """
    Produce un scatter 1D de predicciones continuas, coloreado por clase real.

    Parámetros:
    - y_pred: array-like de predicciones continuas
    - true_labels: array-like de etiquetas reales ("L", "C", "R" o -1, 0, 1)
    - colors: dict opcional {"L": color, "C": color, "R": color}
    - figsize: tamaño de la figura (si ax es None)
    - alpha: transparencia de los puntos
    - s: tamaño de los puntos
    - thresholds: tupla de líneas horizontales guía
    - title: título del gráfico
    - ax: (opcional) axes de matplotlib para dibujar dentro de un subplot
    """

    # Normalizar etiquetas a "L", "C", "R"
    labels_arr = np.array(true_labels)
    num_to_str = {-1: "L", 0: "C", 1: "R"}

    if np.issubdtype(labels_arr.dtype, np.number):
        # Etiquetas numéricas (-1, 0, 1)
        labels_str = [num_to_str[int(c)] for c in labels_arr]
    else:
        # Ya son strings u objetos; forzamos a str
        labels_str = [str(c) for c in labels_arr]

    y_pred = np.array(y_pred)
    point_colors = [colors[c] for c in labels_str]

    # Si no recibimos ax, creamos la figura y el axis
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_figure = True

    # Scatter 1D
    ax.scatter(
        np.arange(len(y_pred)),
        y_pred,
        c=point_colors,
        alpha=alpha,
        s=s
    )

    # Umbrales horizontales
    for t in thresholds:
        if t == 0:
            ax.axhline(t, color="black", linestyle="--", linewidth=1)
        else:
            ax.axhline(t, color="gray", linestyle=":", linewidth=1)

    # Labels
    ax.set_title(title)
    ax.set_xlabel("Índice del sample")
    ax.set_ylabel("Score ideológico (predicción continua)")

    # Leyenda manual
    for label, color in colors.items():
        ax.scatter([], [], c=color, label=label)
    ax.legend(title="Clase real")

    if created_figure:
        plt.tight_layout()
        plt.show()
