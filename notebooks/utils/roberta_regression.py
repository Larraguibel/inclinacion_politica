# utils/roberta_regression.py

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch


# ------------------------------------------------------------
# 1. Preprocesamiento: convertir tweets → Dataset HF
# ------------------------------------------------------------
def prepare_dataset(df, text_col="text", label_col="label"):
    df = df[[text_col, label_col]].copy()
    df[label_col] = df[label_col].astype(float)     # <-- IMPORTANT
    return Dataset.from_pandas(df)

# ------------------------------------------------------------
# 2. Tokenización
# ------------------------------------------------------------
def tokenize_dataset(dataset, tokenizer, max_length=128):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    return dataset.map(tokenize_fn, batched=True)


# ------------------------------------------------------------
# 3. Cargar modelo base + LoRA
# ------------------------------------------------------------
def load_roberta_lora_regression(model_name="roberta-base",
                                 r=16, alpha=32, dropout=0.1):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # IMPORTANT: num_labels=1 => regression head
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,        # <-- ONE neuron output
        problem_type="regression"
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",    # <-- REGRESSION LoRA
    )

    model = get_peft_model(base_model, lora_config)
    return tokenizer, model


# ------------------------------------------------------------
# 4. Entrenar modelo con Trainer
# ------------------------------------------------------------
def train_roberta_lora_regression(
    model,
    tokenizer,
    dataset,
    output_dir="roberta-lora-regression",
    num_epochs=5,
    batch_size=16,
    lr=2e-4,
    weight_decay=0.01,
):
    # Split train/test
    train_test = dataset.train_test_split(test_size=0.2)

    # Versión minimalista, compatible con versiones antiguas de transformers
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_steps=50,
        weight_decay=weight_decay,   # regularización L2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer



# ------------------------------------------------------------
# 5. Inferencia
# ------------------------------------------------------------
def predict_value(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.squeeze().item()     # return scalar float


# ------------------------------------------------------------
# 6. Predicción por lotes
# ------------------------------------------------------------
def predict_batch_values(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.squeeze(-1).cpu().numpy()   # shape: (N,)


# ------------------------------------------------------------
# 7. Guardar modelo + tokenizer (LoRA)
# ------------------------------------------------------------
def save_lora_model(trainer, tokenizer, path="roberta_lora_regression_saved"):
    trainer.save_model(path)
    tokenizer.save_pretrained(path)
    print(f"[✔] Regression LoRA model and tokenizer saved at: {path}")

# ------------------------------------------------------------
# 8. Cargar modelo + tokenizer (LoRA)
# ------------------------------------------------------------
def load_lora_regression_model(base_model_name, saved_path):
    tokenizer = AutoTokenizer.from_pretrained(saved_path)

    # Load base backbone
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        problem_type="regression"
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, saved_path)
    model.eval()

    return tokenizer, model

