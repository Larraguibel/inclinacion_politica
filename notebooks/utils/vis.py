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
    colors={"L": "blue", "C": "green", "R": "red"},
    figsize=(10, 5),
    alpha=0.6,
    s=45,
    thresholds=(-0.33, 0, 0.33),
    title="Predicciones de ideología (1D) coloreadas por clase real"
):
    """
    Produce un scatter 1D de predicciones continuas, coloreado por clase real.

    Parámetros:
    - y_pred: array-like de predicciones continuas
    - true_labels: array-like de etiquetas reales ("L", "C", "R")
    - colors: dict opcional {"L": color, "C": color, "R": color}
    - figsize: tamaño de la figura (default=(10, 5))
    - alpha: transparencia de los puntos
    - s: tamaño de los puntos
    - thresholds: tuplas con valores guía (default: (-0.33, 0, 0.33))
    - title: título del gráfico
    """

    point_colors = [colors[c] for c in true_labels]
    y_pred = np.array(y_pred)

    plt.figure(figsize=figsize)

    # Plot en 1D
    plt.scatter(
        np.arange(len(y_pred)),
        y_pred,
        c=point_colors,
        alpha=alpha,
        s=s
    )

    # Líneas horizontales (umbrales)
    for t in thresholds:
        if t == 0:
            plt.axhline(t, color="black", linestyle="--", linewidth=1)
        else:
            plt.axhline(t, color="gray", linestyle=":", linewidth=1)

    # Etiquetas del plot
    plt.title(title)
    plt.xlabel("Índice del sample en test")
    plt.ylabel("Score ideológico (predicción continua)")

    # Leyenda manual
    for label, color in colors.items():
        plt.scatter([], [], c=color, label=label)
    plt.legend(title="Clase real")