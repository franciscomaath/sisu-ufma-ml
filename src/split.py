"""Divisão temporal. Nunca aleatória: o objetivo é prever anos futuros."""
import pandas as pd

from src import config


def dividir_temporal(df: pd.DataFrame):
    """Retorna (treino, validação, teste). 2017 fica de fora (só histórico)."""
    treino = df[df["ANO"].isin(config.ANOS_TREINO)]
    validacao = df[df["ANO"].isin(config.ANOS_VALIDACAO)]
    teste = df[df["ANO"].isin(config.ANOS_TESTE)]
    return treino, validacao, teste
