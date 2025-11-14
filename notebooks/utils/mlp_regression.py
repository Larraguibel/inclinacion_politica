# mlp_regression.py

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import Dataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_dataloaders(X_train, y_train, X_test, y_test, dataset_cls,
                       batch_size=32, shuffle_train=True):
    """
    Crea los DataLoader a partir de arrays y una clase Dataset.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : np.ndarray o tensores
        Datos de entrada y etiquetas.
    dataset_cls : clase
        Clase de Dataset que recibe (X, y), por ejemplo EmbeddingDataset.
    batch_size : int, opcional
    shuffle_train : bool, opcional

    Returns
    -------
    train_loader, test_loader
    """
    train_ds = dataset_cls(X_train, y_train)
    test_ds = dataset_cls(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate_regression(model, data_loader, device):
    """
    Evalúa el modelo en un DataLoader de regresión y devuelve (MAE, R^2).
    """
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(yb.cpu().numpy())

    preds_arr = np.array(preds_list)
    labels_arr = np.array(labels_list)

    mae = mean_absolute_error(labels_arr, preds_arr)
    r2 = r2_score(labels_arr, preds_arr)
    return mae, r2


def train_mlp_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    dataset_cls,
    input_dim,
    MLP=None,
    device=None,
    batch_size=32,
    lr=1e-3,
    epochs=100,
    eval_every=10,
):
    """
    Entrena un MLP para regresión replicando tu loop original.

    Parámetros
    ----------
    X_train, y_train, X_test, y_test : arrays o tensores
    dataset_cls : clase Dataset (por ej. EmbeddingDataset)
    input_dim : int, dimensión de entrada del MLP
    MLP : clase del modelo (obligatoria)
    device : torch.device, opcional
    batch_size : int
    lr : float
    epochs : int
    eval_every : int
        Cada cuántas épocas evaluar en el set de test.

    Devuelve
    --------
    model : nn.Module
    history : dict con listas 'train_loss', 'test_mae', 'test_r2'
    """
    if device is None:
        device = get_device()

    if MLP is None:
        raise ValueError("Debes proporcionar una clase MLP para el modelo.")
    
    # Dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, dataset_cls,
        batch_size=batch_size, shuffle_train=True
    )

    # Modelo, criterio y optimizador
    model = MLP(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_mae": [],
        "test_r2": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Evaluación periódica en test
        if (epoch + 1) % eval_every == 0:
            mae, r2 = evaluate_regression(model, test_loader, device)

            # Guardar en history
            history["test_mae"].append(mae)
            history["test_r2"].append(r2)

            first_str = f"Epoch {epoch+1}/{epochs}  "
            print(f"{first_str}|  Train Loss: {avg_train_loss:.4f}")
            print(f'{" " * len(first_str)}|  Test MAE: {mae:.4f}  |  Test R^2: {r2:.4f}')

    return model, history


class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]