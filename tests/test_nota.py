"""D12: nota com 2 casas, meio centésimo para cima, e comportamento perto do corte."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import calcular_nota_candidato, calcular_nota_centesimos, montar_entrada_inferencia
from src.limpeza import NOTAS_PROVAS


def _provas(notas: list[float], bonus: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame([dict(zip(NOTAS_PROVAS, notas), PERCENTUAL_BONUS=bonus)])


@pytest.mark.parametrize("notas, esperado", [
    ([500.1] * 5, 52511),                            # 525,105
    ([600.0, 600.0, 600.0, 600.0, 600.5], 63011),    # 630,105
    ([500.0, 500.0, 500.0, 500.0, 500.5], 52511),    # mesma soma, provas diferentes
])
def test_empate_de_meio_centesimo_sobe(notas, esperado):
    centesimos, empate = calcular_nota_centesimos(_provas(notas, 5.0))
    assert centesimos.tolist() == [esperado]
    assert empate.tolist() == [True]
    assert calcular_nota_candidato(_provas(notas, 5.0)).tolist() == [esperado / 100]


@pytest.mark.parametrize("ultima, bonus, esperado", [
    (500.1, 5.0, 525.02),    # 525,021: abaixo da metade
    (500.4, 5.0, 525.08),    # 525,084
    (500.6, 5.0, 525.13),    # 525,126: acima da metade
    (500.9, 5.0, 525.19),    # 525,189
    (500.1, 20.0, 600.02),   # 600,024
    (500.3, 20.0, 600.07),   # 600,072
    (500.3, 10.0, 550.07),   # 550,066
    (500.1, 0.0, 500.02),    # sem bônus a conta já tem 2 casas
])
def test_arredondamento_fora_do_empate(ultima, bonus, esperado):
    df = _provas([500.0, 500.0, 500.0, 500.0, ultima], bonus)
    assert calcular_nota_candidato(df).tolist() == [esperado]
    assert not calcular_nota_centesimos(df)[1].iloc[0]


def test_empate_nao_depende_de_arredondamento_de_ponto_flutuante():
    # A nota exata é 782,145; em ponto flutuante fica logo abaixo e os arredondamentos descem
    df = _provas([640.2, 727.0, 690.5, 846.8, 820.0], 5.0)
    media = df[NOTAS_PROVAS].mean(axis=1) * 1.05
    assert round(float(media.iloc[0]), 2) == 782.14
    assert media.round(2).iloc[0] == 782.14
    assert calcular_nota_candidato(df).iloc[0] == 782.15


@pytest.mark.parametrize("notas, bonus", [
    ([np.nan, 500.0, 500.0, 500.0, 500.0], 0.0),
    ([None, 500.0, 500.0, 500.0, 500.0], 0.0),
    ([np.inf, 500.0, 500.0, 500.0, 500.0], 0.0),
    ([500.0] * 5, np.nan),
])
def test_ausentes_e_nao_finitos_sao_recusados(notas, bonus):
    # uma prova ausente não pode virar média de quatro
    with pytest.raises(ValueError, match="ausentes ou não finitos"):
        calcular_nota_candidato(_provas(notas, bonus))


def test_prova_com_duas_casas_e_bonus_fracionado_sao_recusados():
    with pytest.raises(ValueError, match="1 casa decimal"):
        calcular_nota_candidato(_provas([500.15, 500.0, 500.0, 500.0, 500.0]))
    with pytest.raises(ValueError, match="inteiro"):
        calcular_nota_candidato(_provas([500.0] * 5, 2.5))


def _inferir(notas: list[float], bonus: float, corte: float) -> pd.Series:
    oferta = {"CODIGO_CAMPUS": 1, "NOME_CURSO_NORM": "DIREITO", "GRAU": "Bacharelado",
              "TURNO": "Noturno", "MODALIDADE": "AC", "EDICAO": 1}
    entrada = _provas(notas, bonus).assign(**oferta, OPCAO=1)
    historico = pd.DataFrame([{**oferta, "ANO": 2022, "CORTE_ANO_ANTERIOR": corte}])
    saida = montar_entrada_inferencia(entrada, historico, ano_alvo=2023)
    assert list(saida.columns) == config.ATRIBUTOS
    return saida.iloc[0]


@pytest.mark.parametrize("corte, diferenca, suficiente", [
    (525.10, 0.01, True),
    (525.11, 0.0, True),     # nota igual ao corte
    (525.12, -0.01, False),
])
def test_empate_perto_do_corte(corte, diferenca, suficiente):
    linha = _inferir([500.1] * 5, 5.0, corte)
    assert linha["DIF_CORTE_ANTERIOR"] == pytest.approx(diferenca, abs=1e-9)
    assert (linha["NOTA_CANDIDATO"] >= linha["CORTE_ANO_ANTERIOR"]) == suficiente


def test_nota_igual_ao_corte_sem_bonus_tem_diferenca_zero():
    # A média direta destas provas dá 625,9399999999999, abaixo do corte 625,94
    notas = [627.5, 615.2, 420.3, 646.7, 820.0]
    assert np.mean(notas) < 625.94
    linha = _inferir(notas, 0.0, 625.94)
    assert linha["NOTA_CANDIDATO"] == 625.94
    assert linha["DIF_CORTE_ANTERIOR"] == 0.0
