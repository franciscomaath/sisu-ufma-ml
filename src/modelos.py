"""Construção dos modelos e grade de hiperparâmetros registrada na D5, antes das execuções."""
from itertools import product

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

PESO_BALANCEADO = "balanceado"

# Grade da D5. "peso" é traduzido para cada família em criar_modelo.
GRADE = {
    "logistica": [{"C": c, "peso": peso} for c, peso in product([0.1, 1.0, 10.0], [None, PESO_BALANCEADO])],
    "lightgbm": [
        {"num_leaves": folhas, "min_child_samples": minimo, "peso": peso}
        for folhas, minimo, peso in product([15, 63], [20, 200], [None, PESO_BALANCEADO])
    ],
    "mlp": [
        {"hidden_layer_sizes": camadas, "alpha": alpha, "peso": peso}
        for camadas, alpha, peso in product([(64,), (128, 64)], [1e-4, 1e-3], [None, PESO_BALANCEADO])
    ],
}

# Ensaio de custo da MLP (D5)
TAMANHO_ENSAIO_MLP = 50_000
LIMITE_MINUTOS_MLP = 40
TAMANHO_SUBAMOSTRA_MLP = 200_000
CONFIG_ENSAIO_MLP = {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "peso": None}


def criar_preprocessamento() -> ColumnTransformer:
    """Mediana com indicador e padronização nos numéricos; one-hot nos categóricos (como os baselines)."""
    numericos = Pipeline([
        ("imputacao", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("escala", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numericos", numericos, config.ATRIBUTOS_NUMERICOS),
        ("categoricos", OneHotEncoder(handle_unknown="ignore"), config.ATRIBUTOS_CATEGORICOS),
    ])


def razao_negativos_positivos(y: pd.Series) -> float:
    positivos = int(y.sum())
    if positivos == 0 or positivos == len(y):
        raise ValueError("O treino precisa das duas classes.")
    return (len(y) - positivos) / positivos


def pesos_balanceados(y: pd.Series) -> np.ndarray:
    """Peso n / (2 · contagem da classe), o mesmo de class_weight="balanced"."""
    y = np.asarray(y).astype(int)
    contagem = np.bincount(y, minlength=2)
    if (contagem == 0).any():
        raise ValueError("O treino precisa das duas classes.")
    return (len(y) / (2 * contagem))[y]


def criar_classificador(familia: str, parametros: dict, y_treino: pd.Series):
    """Classificador da família com os parâmetros da grade e os fixos da D5."""
    peso = parametros.get("peso")
    if peso not in (None, PESO_BALANCEADO):
        raise ValueError(f"Peso desconhecido: {peso}")
    if familia == "logistica":
        return LogisticRegression(
            C=parametros["C"], class_weight="balanced" if peso else None, solver="liblinear",
            max_iter=1000, random_state=config.SEMENTE)
    if familia == "lightgbm":
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=parametros["num_leaves"],
            min_child_samples=parametros["min_child_samples"],
            scale_pos_weight=razao_negativos_positivos(y_treino) if peso else 1.0,
            random_state=config.SEMENTE, n_jobs=8, verbose=-1)
    if familia == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=parametros["hidden_layer_sizes"], alpha=parametros["alpha"],
            solver="adam", learning_rate_init=1e-3, batch_size=512, max_iter=50,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
            random_state=config.SEMENTE)
    raise ValueError(f"Família desconhecida: {familia}")


def criar_modelo(familia: str, parametros: dict, y_treino: pd.Series) -> Pipeline:
    return Pipeline([
        ("preprocessamento", criar_preprocessamento()),
        ("classificador", criar_classificador(familia, parametros, y_treino)),
    ])


def argumentos_de_ajuste(familia: str, parametros: dict, y_treino: pd.Series) -> dict:
    """Na MLP, o peso balanceado vai como sample_weight, porque ela não tem class_weight."""
    if familia == "mlp" and parametros.get("peso") == PESO_BALANCEADO:
        return {"classificador__sample_weight": pesos_balanceados(y_treino)}
    return {}


def descrever(parametros: dict) -> str:
    """Texto curto e estável de uma configuração, para tabelas."""
    return ", ".join(f"{chave}={valor}" for chave, valor in parametros.items())
