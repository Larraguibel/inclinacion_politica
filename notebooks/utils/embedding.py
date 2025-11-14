import time
from tqdm import tqdm
import os
from google import genai
import numpy as np
import torch
from torch.utils.data import Dataset


BATCH_SIZE = 100
MODEL_NAME = "text-embedding-004"


class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def embed_texts_in_batches(
    texts,
    model=MODEL_NAME,
    batch_size=BATCH_SIZE,
    client=None,
    max_retries=3,
    backoff_base=1.0,
):
    """
    Genera embeddings por lotes con el modelo de Google AI.
    Devuelve un np.array shape (N, dim) con dtype float32.
    Hace reintentos con backoff SOLO cuando hay error.
    No duerme entre batches exitosos -> máximo rendimiento.
    """
    
    if client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("La variable de entorno 'GOOGLE_API_KEY' no está configurada.")
        client = genai.Client(api_key=api_key)
    
    all_embeddings = []

    for start in tqdm(range(0, len(texts), batch_size)):
        batch = texts[start:start + batch_size]

        # reintentos si la API rate-limitea o algo temporal falla
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch
                )
                batch_embeddings = [e.values for e in response.embeddings]
                all_embeddings.extend(batch_embeddings)
                break  # salimos del loop de reintento si salió bien

            except Exception as e:
                print(f"⚠️ Error en batch {start}-{start+batch_size} (intento {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # backoff exponencial antes de reintentar
                    sleep_time = backoff_base ** attempt
                    print(f"   Esperando {sleep_time:.1f}s antes de reintentar…")
                    time.sleep(sleep_time)
                else:
                    # Ya no quedan intentos -> seguimos adelante pero marcamos el fallo
                    print(f"Batch {start}-{start+batch_size} falló definitivamente, continuo con el siguiente batch.")
                    # Para mantener longitud consistente, opcionalmente puedes
                    # agregar un vector de NaNs para cada texto fallido.
                    fail_vec = [np.full_like(batch_embeddings[0], np.nan)] * len(batch) if 'batch_embeddings' in locals() and len(batch_embeddings) > 0 else []
                    all_embeddings.extend(fail_vec)

    return np.array(all_embeddings, dtype=np.float32)
