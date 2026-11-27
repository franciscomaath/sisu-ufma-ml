"""Garantias sobre o parquet real (2017–2023). Pulados se ele não existir.

Só conferem a preparação das entradas: nenhum modelo é treinado ou avaliado.
"""
import pandas as pd
import pytest

from src import config
from src import verificar_dados as vd
from src.features import construir_historico, montar_entrada_treino


@pytest.fixture(scope="module")
def df():
    if not config.PARQUET_LIMPO.exists():
        pytest.skip("parquet limpo não gerado")
    return pd.read_parquet(config.PARQUET_LIMPO)


@pytest.fixture(scope="module")
def versionado():
    if not config.HISTORICO_CORTES.exists():
        pytest.skip("histórico de cortes ausente")
    return pd.read_parquet(config.HISTORICO_CORTES)


def test_nota_recalculada_igual_a_original(df):
    t = vd.equivalencia_da_nota(df)
    assert t["diferentes"].sum() == 0
    assert t["alvo_diferente"].sum() == 0


def test_historico_versionado_igual_ao_reconstruido(df, versionado):
    pd.testing.assert_frame_equal(construir_historico(df), versionado, check_exact=True)


def test_entradas_de_treino_iguais_as_de_inferencia(df, versionado):
    t = vd.consistencia_treino_inferencia(df, versionado, tamanho_amostra=100)
    assert t["treino_x_inferencia"].sum() == 0
    assert set(t["tipos_e_indice_iguais"]) == {"sim"}
    assert t["treino_x_parquet"].iloc[:-1].sum() == 0


def test_so_o_corte_do_ano_anterior_e_usado(df):
    t = vd.uso_do_ano_anterior(df).drop(columns=["ANO", "linhas"])
    assert (t == 0).all().all()


def test_sem_atributos_proibidos(df):
    entradas = montar_entrada_treino(df, construir_historico(df))
    assert list(entradas.columns) == config.ATRIBUTOS
    assert set(vd.atributos_nas_entradas(df, entradas)["nas_entradas"]) == {"não"}


def test_sem_historico_fica_ausente(df, versionado):
    t = vd.sem_historico(df)
    assert t["corte_e_dif_incoerentes"].sum() == 0
    assert t["outros_atributos_ausentes"].sum() == 0
    assert (t["sem_corte"] == t["chave_inexistente_em_n_1"] + t["chave_sem_valor_em_n_1"]).all()
    assert vd.cenarios_sem_historico(df, versionado)["com_corte"].iloc[1:].sum() == 0


def test_entradas_dos_baselines_publicados_nao_mudaram(df):
    t = vd.entradas_dos_baselines(df)
    assert t["linhas_diferentes"].sum() == 0
    assert set(t["tipos_iguais"]) == {"sim"}
