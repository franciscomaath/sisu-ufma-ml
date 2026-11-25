import pandas as pd
import pytest

from src import config
from src.features import adicionar_corte_anterior, colunas_atributos, construir_historico
from src.limpeza import limpar


def test_nenhuma_coluna_de_vazamento_nos_atributos(df_bruto):
    df = limpar(df_bruto)
    df = adicionar_corte_anterior(df, construir_historico(df))
    atributos = colunas_atributos(df)
    for c in config.COLUNAS_VAZAMENTO + [config.ALVO, "APROVADO"]:
        assert c not in atributos
    assert "CORTE_ANO_ANTERIOR" in atributos and "DIF_CORTE_ANTERIOR" in atributos


def test_atributos_no_parquet_real():
    if not config.PARQUET_LIMPO.exists():
        pytest.skip("parquet limpo não gerado")
    df = pd.read_parquet(config.PARQUET_LIMPO)
    assert not set(colunas_atributos(df)) & set(config.COLUNAS_VAZAMENTO)
