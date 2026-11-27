import pandas as pd
import pytest

from src import config
from src.features import adicionar_corte_anterior, colunas_atributos, construir_historico
from src.limpeza import limpar


def test_nenhuma_coluna_de_vazamento_nos_atributos(df_bruto):
    df = limpar(df_bruto)
    df = adicionar_corte_anterior(df, construir_historico(df))
    df["COLUNA_EXTRA"] = 1
    atributos = colunas_atributos(df)
    assert atributos == config.ATRIBUTOS
    assert len(set(atributos)) == 17
    excluidas = config.COLUNAS_VAZAMENTO + [
        config.ALVO, "APROVADO", "ANO", "SEXO", "IDADE", "UF_CANDIDATO",
        "MUNICIPIO_CANDIDATO", "MUNICIPIO_CAMPUS", "GRUPO_CONCORRENCIA", "COLUNA_EXTRA",
    ]
    assert set(atributos).isdisjoint(excluidas)


def test_atributos_informam_colunas_ausentes():
    df = pd.DataFrame(columns=[c for c in config.ATRIBUTOS if c not in {"NOTA_R", "OPCAO"}])
    with pytest.raises(ValueError, match="Atributos obrigatórios ausentes: NOTA_R, OPCAO"):
        colunas_atributos(df)


def test_atributos_no_parquet_real():
    if not config.PARQUET_LIMPO.exists():
        pytest.skip("parquet limpo não gerado")
    anos = config.ANOS_HISTORICO + config.ANOS_TREINO
    df = pd.read_parquet(config.PARQUET_LIMPO, filters=[("ANO", "in", anos)])
    df = adicionar_corte_anterior(df, construir_historico(df))
    assert colunas_atributos(df) == config.ATRIBUTOS
