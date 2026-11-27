"""Atributos do modelo: corte do ano anterior (legítimo, já era público antes da inscrição),
nota do candidato e montagem da entrada compartilhada entre treino e aplicação (D2, D12, D13)."""
import joblib
import numpy as np
import pandas as pd

from src import config
from src.limpeza import NOTAS_PROVAS, marcar_nota_zero

# Colunas que o chamador precisa fornecer para montar a entrada (treino ou inferência).
COLUNAS_ENTRADA = config.CHAVE_HISTORICO + ["OPCAO", "PERCENTUAL_BONUS"] + NOTAS_PROVAS


def construir_historico(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela de cortes por (curso, modalidade, edição, ano). Um valor por chave.

    O corte deve ser único por chave e ano: com mais de um valor não nulo diferente, levanta
    ValueError. Repetições iguais são aceitas; chave sem nenhum corte fica com NaN.
    """
    chave = config.CHAVE_HISTORICO + ["ANO"]
    distintos = df.groupby(chave)["NOTA_CORTE"].nunique(dropna=True)
    conflitos = distintos[distintos > 1]
    if len(conflitos):
        raise ValueError(
            f"{len(conflitos)} chave(s) (curso, modalidade, edição, ano) com mais de um corte "
            f"não nulo diferente. Exemplos: {conflitos.head(5).index.tolist()}"
        )
    hist = df.groupby(chave, as_index=False)["NOTA_CORTE"].max()  # constante por chave
    return hist.rename(columns={"NOTA_CORTE": "CORTE_ANO_ANTERIOR"})


def adicionar_corte_anterior(df: pd.DataFrame, historico: pd.DataFrame) -> pd.DataFrame:
    """Junta o corte do ano N-1 (mesma edição) em cada inscrição do ano N."""
    h = historico.copy()
    h["ANO"] = h["ANO"] + 1  # o corte do ano N-1 passa a valer para o ano N
    chave = config.CHAVE_HISTORICO + ["ANO"]
    saida = df.drop(columns=["CORTE_ANO_ANTERIOR", "DIF_CORTE_ANTERIOR"], errors="ignore")
    saida = saida.merge(h[chave + ["CORTE_ANO_ANTERIOR"]], on=chave, how="left", validate="m:1")
    saida["DIF_CORTE_ANTERIOR"] = saida["NOTA_CANDIDATO"] - saida["CORTE_ANO_ANTERIOR"]
    return saida


def cobertura_por_ano(df: pd.DataFrame) -> pd.DataFrame:
    """% de linhas com corte anterior encontrado, por ano."""
    return (df.groupby("ANO")["CORTE_ANO_ANTERIOR"].apply(lambda s: 100 * s.notna().mean())
            .round(1).rename("cobertura_%").to_frame())


def colunas_atributos(df: pd.DataFrame) -> list[str]:
    """Valida e retorna as colunas de entrada na ordem definida pela D2."""
    faltantes = [coluna for coluna in config.ATRIBUTOS if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Atributos obrigatórios ausentes: {', '.join(faltantes)}")
    return config.ATRIBUTOS.copy()


def selecionar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve só as colunas da D2, na ordem da lista. Falta de coluna é erro."""
    return df[colunas_atributos(df)].copy()


def calcular_nota_centesimos(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Nota do candidato em centésimos (D12), em aritmética inteira. Retorna (centésimos, empate).

    Ordem das operações: (1) cada prova vira um inteiro em décimos; (2) somam-se as cinco;
    (3) soma × (100 + PERCENTUAL_BONUS); (4) um único arredondamento, no fim:
    centésimos = (produto + 25) // 50, que leva o meio centésimo para cima.
    `empate` marca as notas exatamente no meio de dois centésimos.

    Exige provas com no máximo 1 casa decimal e bônus inteiro, sem ausentes (ValueError).
    Não valida a faixa das notas nem os valores de bônus permitidos.
    """
    decimos = df[NOTAS_PROVAS].astype("float64") * 10
    bonus = df["PERCENTUAL_BONUS"].astype("float64")
    if not (np.isfinite(decimos.to_numpy()).all() and np.isfinite(bonus.to_numpy()).all()):
        raise ValueError("Há notas ou bônus ausentes ou não finitos.")
    if not np.allclose(decimos, decimos.round(), rtol=0, atol=1e-6):
        raise ValueError("Cada prova deve ter no máximo 1 casa decimal.")
    if not (bonus == bonus.round()).all():
        raise ValueError("PERCENTUAL_BONUS deve ser um número inteiro.")
    soma = decimos.round().astype("int64").sum(axis=1)
    # nota = soma/10/5 × (100 + bônus)/100, portanto centésimos = soma × (100 + bônus) / 50
    produto = soma * (100 + bonus.round().astype("int64"))
    return (produto + 25) // 50, (produto % 50) == 25


def calcular_nota_candidato(df: pd.DataFrame) -> pd.Series:
    """D12: média das 5 provas × (1 + bônus/100), com 2 casas e meio centésimo para cima.

    Única função da nota no treino e na inferência. A regra reproduz a NOTA_CANDIDATO dos
    dados históricos de 2017–2023; não foi conferida em fonte oficial do SISU.
    """
    centesimos, _ = calcular_nota_centesimos(df)
    return centesimos / 100


def diferenca_nota_recalculada(df: pd.DataFrame) -> pd.Series:
    """|NOTA_CANDIDATO do dado − nota recalculada|, por linha."""
    return (df["NOTA_CANDIDATO"] - calcular_nota_candidato(df)).abs()


def _montar(df: pd.DataFrame, historico: pd.DataFrame, ano_inscricao: pd.Series) -> pd.DataFrame:
    faltantes = [coluna for coluna in COLUNAS_ENTRADA if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes para montar a entrada: {', '.join(faltantes)}")
    base = df[COLUNAS_ENTRADA].copy()
    base["ANO"] = ano_inscricao.to_numpy()  # só para consultar o ano N-1
    base["NOTA_CANDIDATO"] = calcular_nota_candidato(base)
    base["TEM_NOTA_ZERO"] = marcar_nota_zero(base)
    saida = adicionar_corte_anterior(base, historico)  # NaN sem correspondência (D3)
    saida.index = df.index
    return selecionar_atributos(saida)


def montar_entrada_treino(df: pd.DataFrame, historico: pd.DataFrame) -> pd.DataFrame:
    """Atributos (D2) para dados históricos: cada linha usa o corte do ano N-1 da coluna ANO.

    `df` precisa de ANO e das COLUNAS_ENTRADA. O índice é preservado, para alinhar o alvo.
    NOTA_CANDIDATO é recalculada; NOTA_CORTE, alvo, ANO e outras colunas não chegam à saída.
    """
    if "ANO" not in df.columns:
        raise ValueError("Coluna ANO obrigatória na montagem para treinamento.")
    return _montar(df, historico, df["ANO"])


def montar_entrada_inferencia(df: pd.DataFrame, historico: pd.DataFrame, ano_alvo: int) -> pd.DataFrame:
    """Atributos (D2) para prever o ano `ano_alvo`, consultando só o histórico de `ano_alvo - 1`.

    Mesmo código do treino (D13). `df` não precisa de ANO; o índice é preservado.
    """
    return _montar(df, historico, pd.Series(int(ano_alvo), index=df.index))


def main() -> None:
    df = pd.read_parquet(config.PARQUET_LIMPO)
    hist = construir_historico(df)
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(config.HISTORICO_CORTES, index=False)
    joblib.dump(hist, config.HISTORICO_CORTES_JOBLIB)
    df = adicionar_corte_anterior(df, hist)
    print(cobertura_por_ano(df))
    print(df.groupby("ANO")["CORTE_ANO_ANTERIOR"].apply(lambda s: s.notna().sum()))


if __name__ == "__main__":
    main()
