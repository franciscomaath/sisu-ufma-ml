"""Análise exploratória com os dados de treino (2018–2021), sem tocar na validação nem no teste.

Gera figuras em reports/figures/ e tabelas em reports/experimentos/.
Uso: python -m src.exploracao
"""
import matplotlib
import numpy as np
import pandas as pd

from src import config
from src.features import construir_historico, montar_entrada_treino
from src.particoes import DIR_EXPERIMENTOS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AZUL, LARANJA, VERDE_AGUA = "#257de0", "#f47230", "#00b894"
TINTA, TINTA_SECUNDARIA, GRADE = "#191919", "#666666", "#e6e6e6"
GRUPOS_MODALIDADE = {"AC": "Ampla concorrência", "BONUS_MA": "Bônus Maranhão"}
LARGURA_FAIXA = 10
FAIXA_DIFERENCA = (-150, 100)


def dados_de_treino() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inscrições rotuladas de treino e suas entradas montadas (histórico de 2017–2021)."""
    anos = config.ANOS_HISTORICO + config.ANOS_TREINO
    df = pd.read_parquet(config.PARQUET_LIMPO, filters=[("ANO", "in", anos)])
    treino = df[df["ANO"].isin(config.ANOS_TREINO)].dropna(subset=[config.ALVO])
    return treino, montar_entrada_treino(treino, construir_historico(df))


def grupo_modalidade(modalidade: pd.Series) -> pd.Series:
    return modalidade.map(GRUPOS_MODALIDADE).fillna("Cotas")


def resumo_alvo(treino: pd.DataFrame) -> pd.DataFrame:
    y = treino[config.ALVO].astype(int)
    t = y.groupby(treino["ANO"]).agg(rotuladas="size", positivos="sum").reset_index()
    t["taxa_%"] = (100 * t["positivos"] / t["rotuladas"]).round(2)
    return t


def ausencia_corte(treino: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """% de linhas sem corte anterior por ano, edição e grupo de modalidade."""
    base = pd.DataFrame({"ANO": treino["ANO"], "EDICAO": treino["EDICAO"],
                         "grupo": grupo_modalidade(treino["MODALIDADE"]),
                         "sem_corte": x["CORTE_ANO_ANTERIOR"].isna()})
    t = base.groupby(["ANO", "EDICAO", "grupo"])["sem_corte"].agg(linhas="size", sem_corte="sum").reset_index()
    t["sem_corte_%"] = (100 * t["sem_corte"] / t["linhas"]).round(1)
    return t


def taxa_por_diferenca(x: pd.DataFrame, y: pd.Series, largura: int = LARGURA_FAIXA,
                       faixa: tuple[int, int] = FAIXA_DIFERENCA) -> pd.DataFrame:
    """Taxa de nota suficiente por faixa de DIF_CORTE_ANTERIOR; extremos agregados nas pontas."""
    dif = x["DIF_CORTE_ANTERIOR"]
    com = dif.notna()
    bordas = np.arange(faixa[0], faixa[1] + largura, largura)
    limitada = dif[com].clip(faixa[0], faixa[1] - 1e-9)
    faixas = pd.cut(limitada, bordas, right=False)
    t = y[com].astype(int).groupby(faixas, observed=True).agg(linhas="size", positivos="sum").reset_index()
    t["inicio"] = [intervalo.left for intervalo in t.iloc[:, 0]]
    t = t.drop(columns=t.columns[0])
    t["taxa_%"] = (100 * t["positivos"] / t["linhas"]).round(2)
    sem = y[~com].astype(int)
    linha_sem = pd.DataFrame([{"linhas": len(sem), "positivos": int(sem.sum()), "inicio": np.nan,
                               "taxa_%": round(100 * sem.mean(), 2) if len(sem) else np.nan}])
    return pd.concat([t, linha_sem], ignore_index=True)[["inicio", "linhas", "positivos", "taxa_%"]]


def extremos(x: pd.DataFrame) -> pd.DataFrame:
    """Mínimo, quantis, máximo, ausentes e zeros de cada atributo numérico."""
    linhas = []
    for coluna in config.ATRIBUTOS_NUMERICOS:
        s = x[coluna].astype("float64")
        q = s.quantile([0.01, 0.5, 0.99])
        linhas.append({"atributo": coluna, "minimo": s.min(), "p01": q[0.01], "mediana": q[0.5],
                       "p99": q[0.99], "maximo": s.max(), "ausentes": int(s.isna().sum()),
                       "zeros": int((s == 0).sum())})
    return pd.DataFrame(linhas)


def _eixo_limpo(eixo) -> None:
    eixo.grid(axis="y", color=GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    eixo.tick_params(colors=TINTA_SECUNDARIA)


def figura_alvo(resumo: pd.DataFrame, caminho) -> None:
    fig, (esq, dir_) = plt.subplots(1, 2, figsize=(10, 4.2))
    anos = resumo["ANO"].astype(str)
    esq.bar(anos, resumo["rotuladas"] / 1000, color=AZUL, width=0.6)
    for i, v in enumerate(resumo["rotuladas"]):
        esq.text(i, v / 1000, f"{v / 1000:.1f} mil".replace(".", ","), ha="center", va="bottom", color=TINTA)
    esq.set(title="Inscrições com alvo", ylabel="Milhares de inscrições", xlabel="Ano")
    dir_.bar(anos, resumo["taxa_%"], color=AZUL, width=0.6)
    for i, v in enumerate(resumo["taxa_%"]):
        dir_.text(i, v, f"{v:.2f}%".replace(".", ","), ha="center", va="bottom", color=TINTA)
    dir_.set(title="Taxa de nota suficiente", ylabel="% das inscrições com alvo", xlabel="Ano")
    for eixo in (esq, dir_):
        _eixo_limpo(eixo)
    fig.suptitle("Distribuição do alvo no treino (2018–2021)", color=TINTA)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300)
    plt.close(fig)


def figura_ausencia(ausencia: pd.DataFrame, caminho) -> None:
    por_ano = ausencia.groupby(["ANO", "grupo"])[["linhas", "sem_corte"]].sum().reset_index()
    por_ano["pct"] = 100 * por_ano["sem_corte"] / por_ano["linhas"]
    grupos = [("Ampla concorrência", AZUL), ("Bônus Maranhão", LARANJA), ("Cotas", VERDE_AGUA)]
    anos = sorted(por_ano["ANO"].unique())
    largura = 0.26
    fig, eixo = plt.subplots(figsize=(8.5, 4.6))
    for k, (grupo, cor) in enumerate(grupos):
        valores = [por_ano.loc[(por_ano["ANO"] == a) & (por_ano["grupo"] == grupo), "pct"] for a in anos]
        valores = [v.iloc[0] if len(v) else np.nan for v in valores]
        posicoes = np.arange(len(anos)) + (k - 1) * (largura + 0.02)
        eixo.bar(posicoes, valores, width=largura, color=cor, label=grupo)
        for p, v in zip(posicoes, valores):
            if not np.isnan(v):
                eixo.text(p, v, f"{v:.0f}%", ha="center", va="bottom", color=TINTA, fontsize=8)
            else:
                eixo.text(p, 1, "não\nexistia", ha="center", va="bottom", color=TINTA_SECUNDARIA, fontsize=7)
    eixo.set_xticks(np.arange(len(anos)), [str(a) for a in anos])
    eixo.set(ylim=(0, 105), xlabel="Ano", ylabel="% das inscrições sem corte do ano anterior",
             title="Ausência do corte do ano anterior no treino, por grupo de modalidade")
    eixo.legend(frameon=False, loc="upper right")
    _eixo_limpo(eixo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300)
    plt.close(fig)


def figura_diferenca(taxa: pd.DataFrame, caminho) -> None:
    com = taxa.dropna(subset=["inicio"])
    centro = com["inicio"] + LARGURA_FAIXA / 2
    fig, eixo = plt.subplots(figsize=(8.5, 4.6))
    eixo.plot(centro, com["taxa_%"], color=AZUL, linewidth=2, marker="o", markersize=6)
    eixo.axvline(0, color=TINTA_SECUNDARIA, linestyle="--", linewidth=1)
    eixo.text(2, 8, "nota igual ao\ncorte anterior", color=TINTA_SECUNDARIA, fontsize=8, va="center")
    sem = taxa[taxa["inicio"].isna()].iloc[0]
    eixo.set(xlabel=f"Nota do candidato − corte do ano anterior (faixas de {LARGURA_FAIXA} pontos; "
                    "pontas agregam os extremos)",
             ylabel="% com nota suficiente", ylim=(0, 102),
             title="Taxa de nota suficiente pela distância ao corte anterior (treino)")
    texto = (f"Sem corte anterior: {sem['taxa_%']:.2f}%".replace(".", ",")
             + f" de {int(sem['linhas']):,} inscrições".replace(",", "."))
    eixo.annotate(texto,
                  xy=(0.01, 0.95), xycoords="axes fraction", color=TINTA, fontsize=9)
    _eixo_limpo(eixo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300)
    plt.close(fig)


def main() -> None:
    treino, x = dados_de_treino()
    y = treino[config.ALVO].astype(int)
    resumo, ausencia = resumo_alvo(treino), ausencia_corte(treino, x)
    taxa, tabela_extremos = taxa_por_diferenca(x, y), extremos(x)
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    DIR_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
    figura_alvo(resumo, config.DIR_FIGURAS / "alvo_treino_por_ano.png")
    figura_ausencia(ausencia, config.DIR_FIGURAS / "ausencia_corte_anterior_treino.png")
    figura_diferenca(taxa, config.DIR_FIGURAS / "taxa_por_diferenca_corte_treino.png")
    for nome, tabela in [("alvo", resumo), ("ausencia_corte", ausencia),
                         ("taxa_por_diferenca", taxa), ("extremos", tabela_extremos)]:
        tabela.to_csv(DIR_EXPERIMENTOS / f"exploracao_{nome}.csv", index=False, lineterminator="\n")
        print(f"\n== {nome}\n{tabela.to_string(index=False)}")


if __name__ == "__main__":
    main()
