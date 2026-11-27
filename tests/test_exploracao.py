import numpy as np
import pandas as pd

from src import config
from src import exploracao as ep
from src.features import construir_historico, montar_entrada_treino


def _treino(df):
    df = df.dropna(subset=[config.ALVO])
    treino = df[df["ANO"].isin(config.ANOS_TREINO)]
    return treino, montar_entrada_treino(treino, construir_historico(df[df["ANO"] <= 2021]))


def test_resumo_e_ausencia(df_limpo_sintetico):
    treino, x = _treino(df_limpo_sintetico)
    resumo = ep.resumo_alvo(treino)
    assert resumo["ANO"].tolist() == config.ANOS_TREINO
    assert resumo["rotuladas"].sum() == len(treino)
    ausencia = ep.ausencia_corte(treino, x)
    assert ausencia["linhas"].sum() == len(treino)
    assert ausencia["sem_corte"].sum() == x["CORTE_ANO_ANTERIOR"].isna().sum()
    assert set(ep.grupo_modalidade(pd.Series(["AC", "BONUS_MA", "PP_RENDA"]))) == {
        "Ampla concorrência", "Bônus Maranhão", "Cotas"}


def test_taxa_por_diferenca_agrega_extremos_e_sem_historico():
    x = pd.DataFrame({"DIF_CORTE_ANTERIOR": [-900.0, -5.0, 3.0, 500.0, np.nan]})
    y = pd.Series([0, 0, 1, 1, 1])
    t = ep.taxa_por_diferenca(x, y, largura=10, faixa=(-20, 20))
    assert t["linhas"].sum() == 5
    assert t.loc[t["inicio"] == -20, "linhas"].item() == 1   # -900 vai para a primeira faixa
    assert t.loc[t["inicio"] == 10, "linhas"].item() == 1    # 500 vai para a última
    assert t.loc[t["inicio"].isna(), "taxa_%"].item() == 100.0


def test_extremos(df_limpo_sintetico):
    _, x = _treino(df_limpo_sintetico)
    t = ep.extremos(x).set_index("atributo")
    assert list(t.index) == config.ATRIBUTOS_NUMERICOS
    assert t.loc["NOTA_L", "minimo"] <= t.loc["NOTA_L", "mediana"] <= t.loc["NOTA_L", "maximo"]


def test_figuras_sao_gravadas(df_limpo_sintetico, tmp_path):
    treino, x = _treino(df_limpo_sintetico)
    y = treino[config.ALVO].astype(int)
    ep.figura_alvo(ep.resumo_alvo(treino), tmp_path / "a.png")
    ep.figura_ausencia(ep.ausencia_corte(treino, x), tmp_path / "b.png")
    ep.figura_diferenca(ep.taxa_por_diferenca(x, y), tmp_path / "c.png")
    assert all((tmp_path / n).stat().st_size > 0 for n in ["a.png", "b.png", "c.png"])
