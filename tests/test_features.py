import pandas as pd
import pytest

from src import config
from src.features import adicionar_corte_anterior, construir_historico
from src.limpeza import limpar


def test_corte_anterior_vem_do_ano_anterior(df_bruto):
    df = limpar(df_bruto)
    out = adicionar_corte_anterior(df, construir_historico(df))
    esperado = {2017: None, 2018: 600.0, 2019: 610.0}  # NOTA_CORTE do ano N-1
    for ano, corte in esperado.items():
        vals = out.loc[out.ANO == ano, "CORTE_ANO_ANTERIOR"]
        if corte is None:
            assert vals.isna().all()
        else:
            assert (vals == corte).all()  # nunca o corte do próprio ano


def test_dif_corte_anterior(df_bruto):
    df = limpar(df_bruto)
    out = adicionar_corte_anterior(df, construir_historico(df))
    linha = out[(out.ANO == 2019) & (out.NOTA_CANDIDATO == 700.0)].iloc[0]
    assert linha.DIF_CORTE_ANTERIOR == 700.0 - 610.0


def test_sem_duplicar_linhas(df_bruto):
    df = limpar(df_bruto)
    assert len(adicionar_corte_anterior(df, construir_historico(df))) == len(df)


def _linhas_corte(cortes, ano=2022):
    """Inscrições sintéticas da mesma chave e ano, com os cortes dados."""
    base = {"CODIGO_CAMPUS": 1, "NOME_CURSO_NORM": "DIREITO", "GRAU": "B", "TURNO": "N",
            "MODALIDADE": "AC", "EDICAO": 1, "ANO": ano}
    return pd.DataFrame([{**base, "NOTA_CORTE": c} for c in cortes])


def test_historico_aceita_cortes_repetidos_iguais():
    hist = construir_historico(_linhas_corte([650.0, 650.0, 650.0]))
    assert len(hist) == 1 and hist["CORTE_ANO_ANTERIOR"].iloc[0] == 650.0


def test_historico_erro_com_cortes_conflitantes():
    with pytest.raises(ValueError, match="mais de um corte"):
        construir_historico(_linhas_corte([650.0, 651.0]))


def test_historico_conflito_ignora_ausentes_mas_detecta_dois_valores():
    # um valor + ausente: sem conflito, vale o valor
    hist = construir_historico(_linhas_corte([650.0, float("nan")]))
    assert hist["CORTE_ANO_ANTERIOR"].iloc[0] == 650.0
    # dois valores + ausente: conflito
    with pytest.raises(ValueError, match="mais de um corte"):
        construir_historico(_linhas_corte([650.0, float("nan"), 700.0]))


def test_historico_todos_ausentes_fica_ausente():
    hist = construir_historico(_linhas_corte([float("nan"), float("nan")]))
    assert len(hist) == 1 and pd.isna(hist["CORTE_ANO_ANTERIOR"].iloc[0])


def test_historico_conflito_em_um_ano_nao_afeta_outro_erro_aponta_a_chave():
    df = pd.concat([_linhas_corte([650.0, 650.0], 2021), _linhas_corte([600.0, 610.0], 2022)])
    with pytest.raises(ValueError, match="2022"):
        construir_historico(df)
