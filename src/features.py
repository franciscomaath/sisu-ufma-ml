"""Atributo-chave: corte do ano anterior (legítimo, já era público antes da inscrição)."""
import joblib
import pandas as pd

from src import config


def construir_historico(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela de cortes por (curso, modalidade, edição, ano). Um valor por chave."""
    chave = config.CHAVE_HISTORICO + ["ANO"]
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
    """Colunas permitidas como atributos (exclui alvo, vazamento e auxiliares)."""
    return [c for c in df.columns if c not in config.COLUNAS_NAO_ATRIBUTO]


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
