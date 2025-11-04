import re
from langdetect import detect, DetectorFactory
import pandas as pd


DetectorFactory.seed = 0


def clean_text(text):
    if not isinstance(text, str):
        return text
    
    # 1️⃣ Eliminar URLs (http, https, www)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # 2️⃣ Eliminar menciones de Twitter o handles tipo @usuario
    text = re.sub(r"@\w+", "", text)
    
    # 3️⃣ Eliminar hashtags (#hashtag)
    text = re.sub(r"#\w+", "", text)
    
    # 4️⃣ Eliminar emojis y símbolos fuera del rango básico de Unicode
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticones
                               u"\U0001F300-\U0001F5FF"  # símbolos y pictogramas
                               u"\U0001F680-\U0001F6FF"  # transporte y mapas
                               u"\U0001F1E0-\U0001F1FF"  # banderas
                               u"\U00002700-\U000027BF"  # otros símbolos
                               u"\U000024C2-\U0001F251"  # adicional
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r"", text)
    
    # 5️⃣ Quitar espacios múltiples o finales
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False  # por si hay textos vacíos o emojis
