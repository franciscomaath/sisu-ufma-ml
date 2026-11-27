"""D12/D13: montagem compartilhada da entrada (treino e inferência), com dados sintéticos."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import (
    adicionar_corte_anterior, calcular_nota_candidato, construir_historico,
    diferenca_nota_recalculada, montar_entrada_inferencia, montar_entrada_treino,
)
from src.limpeza import limpar


def _oferta(**kw):
    base = {"CODIGO_CAMPUS": 1, "NOME_CURSO_NORM": "DIREITO", "GRAU": "Bacharelado",
            "TURNO": "Noturno", "MODALIDADE": "AC", "EDICAO": 1, "OPCAO": 1,
            "PERCENTUAL_BONUS": 0.0, "NOTA_L": 700.0, "NOTA_CH": 700.0, "NOTA_CN": 700.0,
            "NOTA_M": 700.0, "NOTA_R": 700.0}
    base.update(kw)
    return base


def _historico():
    """Cortes por ano. 2023 e 2024 têm valores distintos para detectar uso do ano errado."""
    linhas = []
    for ano, corte in [(2021, 600.0), (2022, 650.0), (2023, 680.0), (2024, 999.0)]:
        linhas.append({**{k: v for k, v in _oferta().items() if k in config.CHAVE_HISTORICO},
                       "ANO": ano, "CORTE_ANO_ANTERIOR": corte})
    return pd.DataFrame(linhas)


def test_nota_candidato_media_das_cinco_com_bonus():
    df = pd.DataFrame([_oferta(NOTA_L=800.0, NOTA_CH=700.0, NOTA_CN=600.0, NOTA_M=500.0,
                               NOTA_R=900.0, PERCENTUAL_BONUS=10.0),
                       _oferta(NOTA_L=500.0, NOTA_CH=500.0, NOTA_CN=500.0, NOTA_M=500.0, NOTA_R=600.0)])
    nota = calcular_nota_candidato(df)
    assert nota.iloc[0] == pytest.approx(700.0 * 1.10)
    assert nota.iloc[1] == pytest.approx(520.0)


def test_diferenca_nota_recalculada_detecta_divergencia():
    df = pd.DataFrame([_oferta(), _oferta()])
    df["NOTA_CANDIDATO"] = [700.0, 705.0]
    assert diferenca_nota_recalculada(df).tolist() == pytest.approx([0.0, 5.0])


def test_inferencia_usa_ano_anterior_ao_ano_alvo():
    df = pd.DataFrame([_oferta()])
    x24 = montar_entrada_inferencia(df, _historico(), ano_alvo=2024)
    x23 = montar_entrada_inferencia(df, _historico(), ano_alvo=2023)
    assert x24["CORTE_ANO_ANTERIOR"].iloc[0] == 680.0  # corte de 2023; o 999 (2024) nunca é usado
    assert x23["CORTE_ANO_ANTERIOR"].iloc[0] == 650.0  # corte de 2022; o 680 (2023) nunca é usado
    assert x24["DIF_CORTE_ANTERIOR"].iloc[0] == pytest.approx(700.0 - 680.0)


def test_sem_correspondencia_fica_ausente():
    df = pd.DataFrame([_oferta(), _oferta(MODALIDADE="PP_RENDA"), _oferta(EDICAO=2)])
    x = montar_entrada_inferencia(df, _historico(), ano_alvo=2024)
    assert x["CORTE_ANO_ANTERIOR"].notna().tolist() == [True, False, False]
    assert x["DIF_CORTE_ANTERIOR"].notna().tolist() == [True, False, False]
    # ano sem N-1 no histórico: tudo ausente
    y = montar_entrada_inferencia(df, _historico(), ano_alvo=2030)
    assert y["CORTE_ANO_ANTERIOR"].isna().all()


def test_saida_so_com_atributos_d2_na_ordem():
    df = pd.DataFrame([_oferta(NOTA_CORTE=1.0, CLASSIFICACAO=1, MATRICULA="S", SEXO="F", IDADE=20,
                               NOTA_SUFICIENTE=1, APROVADO=1, ANO=2022)])
    for x in (montar_entrada_inferencia(df, _historico(), 2024), montar_entrada_treino(df, _historico())):
        assert list(x.columns) == config.ATRIBUTOS


def test_tem_nota_zero():
    df = pd.DataFrame([_oferta(), _oferta(NOTA_M=0.0)])
    x = montar_entrada_inferencia(df, _historico(), 2024)
    assert x["TEM_NOTA_ZERO"].tolist() == [0, 1]


def test_treino_e_inferencia_produzem_as_mesmas_entradas():
    df = pd.DataFrame([_oferta(NOTA_L=800.0, PERCENTUAL_BONUS=10.0), _oferta(MODALIDADE="PP_RENDA"),
                       _oferta(EDICAO=2, NOTA_CN=0.0)], index=[10, 11, 12])
    treino = montar_entrada_treino(df.assign(ANO=2023), _historico())
    infer = montar_entrada_inferencia(df, _historico(), ano_alvo=2023)
    pd.testing.assert_frame_equal(treino, infer)


def test_treino_usa_o_ano_de_cada_linha_e_preserva_indice():
    df = pd.DataFrame([_oferta(ANO=2022), _oferta(ANO=2023), _oferta(ANO=2024)], index=[5, 3, 9])
    x = montar_entrada_treino(df, _historico())
    assert x.index.tolist() == [5, 3, 9]
    assert x["CORTE_ANO_ANTERIOR"].tolist() == [600.0, 650.0, 680.0]


def test_entrada_com_colunas_obrigatorias_ausentes():
    df = pd.DataFrame([_oferta()]).drop(columns=["NOTA_M"])
    with pytest.raises(ValueError, match="NOTA_M"):
        montar_entrada_inferencia(df, _historico(), 2024)
    with pytest.raises(ValueError, match="ANO"):
        montar_entrada_treino(pd.DataFrame([_oferta()]), _historico())


def test_nao_modifica_o_dataframe_de_entrada():
    df = pd.DataFrame([_oferta()])
    antes = df.copy()
    montar_entrada_inferencia(df, _historico(), 2024)
    pd.testing.assert_frame_equal(df, antes)


def test_integracao_com_limpeza_e_historico(df_bruto):
    """Saída da limpeza → histórico → montagem; reaproveita adicionar_corte_anterior."""
    limpo = limpar(df_bruto)
    hist = construir_historico(limpo)
    x = montar_entrada_treino(limpo, hist)
    ref = adicionar_corte_anterior(limpo, hist)
    assert list(x.columns) == config.ATRIBUTOS and len(x) == len(limpo)
    np.testing.assert_array_equal(x["CORTE_ANO_ANTERIOR"].to_numpy(), ref["CORTE_ANO_ANTERIOR"].to_numpy())
    np.testing.assert_array_equal(x["TEM_NOTA_ZERO"].to_numpy(), limpo["TEM_NOTA_ZERO"].to_numpy())


def test_inferencia_com_historico_real_versionado():
    if not config.HISTORICO_CORTES.exists():
        pytest.skip("historico_cortes.parquet ausente")
    hist = pd.read_parquet(config.HISTORICO_CORTES)
    ref = hist[(hist.ANO == hist.ANO.max()) & hist.CORTE_ANO_ANTERIOR.notna()].iloc[0]
    linha = {**{c: ref[c] for c in config.CHAVE_HISTORICO}, "OPCAO": 1, "PERCENTUAL_BONUS": 0.0,
             "NOTA_L": 700.0, "NOTA_CH": 700.0, "NOTA_CN": 700.0, "NOTA_M": 700.0, "NOTA_R": 700.0}
    x = montar_entrada_inferencia(pd.DataFrame([linha]), hist, ano_alvo=int(ref["ANO"]) + 1)
    assert x["CORTE_ANO_ANTERIOR"].iloc[0] == ref["CORTE_ANO_ANTERIOR"]
