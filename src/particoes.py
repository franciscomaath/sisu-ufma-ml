"""Divisão interna da validação (2022) em A, B e C, por grupo de oferta e modalidade (D4, D8).

Só as inscrições rotuladas de 2022 entram. As cinco partições do StratifiedGroupKFold apenas
constroem A (0, 1, 2), B (3) e C (4); não são rodadas de validação cruzada.
Uso: python -m src.particoes
"""
import hashlib

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config

N_PARTICOES = 5
PARTES = {0: "A", 1: "A", 2: "A", 3: "B", 4: "C"}
DIR_EXPERIMENTOS = config.RAIZ / "reports" / "experimentos"
ARQUIVO_MAPA = DIR_EXPERIMENTOS / "particoes_2022.csv"


def rotuladas_validacao(df: pd.DataFrame) -> pd.DataFrame:
    """Inscrições de config.ANOS_VALIDACAO com alvo."""
    return df[df["ANO"].isin(config.ANOS_VALIDACAO)].dropna(subset=[config.ALVO])


def chave_grupo(df: pd.DataFrame) -> pd.Series:
    """Identificador textual do grupo: os campos de config.CHAVE_HISTORICO unidos por '|'."""
    return df[config.CHAVE_HISTORICO].astype(str).agg("|".join, axis=1)


def particionar(df: pd.DataFrame) -> pd.Series:
    """Partição (0 a 4) de cada linha, estratificando pelo alvo e agrupando pela chave."""
    if df[config.ALVO].isna().any():
        raise ValueError("Particione só inscrições rotuladas.")
    grupos = pd.factorize(chave_grupo(df))[0]
    divisor = StratifiedGroupKFold(n_splits=N_PARTICOES, shuffle=True, random_state=config.SEMENTE)
    particao = pd.Series(-1, index=df.index, dtype="int64")
    for numero, (_, posicoes) in enumerate(divisor.split(df, df[config.ALVO].astype(int), grupos)):
        particao.iloc[posicoes] = numero
    return particao


def mapa_grupos(df: pd.DataFrame, particao: pd.Series) -> pd.DataFrame:
    """Um registro por grupo: chave, partição, parte, linhas e positivos."""
    base = df[config.CHAVE_HISTORICO].assign(
        particao=particao, positivos=df[config.ALVO].astype(int))
    mapa = base.groupby(config.CHAVE_HISTORICO + ["particao"], as_index=False).agg(
        linhas=("positivos", "size"), positivos=("positivos", "sum"))
    if mapa.duplicated(config.CHAVE_HISTORICO).any():
        raise ValueError("Um grupo caiu em mais de uma partição.")
    mapa.insert(len(config.CHAVE_HISTORICO) + 1, "parte", mapa["particao"].map(PARTES))
    return mapa.sort_values(config.CHAVE_HISTORICO, ignore_index=True)


def aplicar_mapa(df: pd.DataFrame, mapa: pd.DataFrame) -> pd.Series:
    """Parte (A, B ou C) de cada linha a partir de um mapa salvo. Grupo ausente do mapa é erro."""
    chaves = mapa.assign(_chave=chave_grupo(mapa)).set_index("_chave")["parte"]
    parte = chave_grupo(df).map(chaves)
    if parte.isna().any():
        raise ValueError(f"{int(parte.isna().sum())} linha(s) de grupos ausentes do mapa.")
    return parte.rename("parte")


def resumo_partes(df: pd.DataFrame, parte: pd.Series) -> pd.DataFrame:
    """Grupos, linhas, positivos, negativos, taxa e fração das linhas em cada parte."""
    base = pd.DataFrame({"parte": parte, "grupo": chave_grupo(df), "y": df[config.ALVO].astype(int)})
    t = base.groupby("parte").agg(grupos=("grupo", "nunique"), linhas=("y", "size"), positivos=("y", "sum"))
    t["negativos"] = t["linhas"] - t["positivos"]
    t["taxa_%"] = (100 * t["positivos"] / t["linhas"]).round(2)
    t["fracao_linhas_%"] = (100 * t["linhas"] / t["linhas"].sum()).round(1)
    return t.reset_index()


def verificar_partes(df: pd.DataFrame, parte: pd.Series) -> dict[str, int]:
    """Confere disjunção de grupos e linhas e presença das duas classes. Falha com ValueError."""
    grupos_por_chave = pd.DataFrame({"grupo": chave_grupo(df), "parte": parte}).groupby("grupo")["parte"].nunique()
    checagens = {
        "linhas sem parte": int(parte.isna().sum()),
        "linhas duplicadas no índice": int(df.index.duplicated().sum()),
        "grupos em mais de uma parte": int((grupos_por_chave > 1).sum()),
        "partes sem as duas classes": int(
            (df[config.ALVO].astype(int).groupby(parte).nunique() < 2).sum()),
        "partes ausentes": len({"A", "B", "C"} - set(parte.dropna())),
    }
    problemas = {nome: valor for nome, valor in checagens.items() if valor}
    if problemas:
        raise ValueError(f"Divisão inválida: {problemas}")
    return checagens


def sha256(caminho) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def main() -> None:
    df = pd.read_parquet(config.PARQUET_LIMPO, filters=[("ANO", "in", config.ANOS_VALIDACAO)])
    rotuladas = rotuladas_validacao(df)
    particao = particionar(rotuladas)
    mapa = mapa_grupos(rotuladas, particao)
    parte = aplicar_mapa(rotuladas, mapa)
    verificar_partes(rotuladas, parte)

    DIR_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
    mapa.to_csv(ARQUIVO_MAPA, index=False, lineterminator="\n")
    por_particao = mapa.groupby(["particao", "parte"], as_index=False).agg(
        grupos=("linhas", "size"), linhas=("linhas", "sum"), positivos=("positivos", "sum"))
    por_particao["negativos"] = por_particao["linhas"] - por_particao["positivos"]
    resumo = resumo_partes(rotuladas, parte)
    por_particao.to_csv(DIR_EXPERIMENTOS / "particoes_2022_resumo_particoes.csv", index=False, lineterminator="\n")
    resumo.to_csv(DIR_EXPERIMENTOS / "particoes_2022_resumo_partes.csv", index=False, lineterminator="\n")
    print(f"Rotuladas de {config.ANOS_VALIDACAO}: {len(rotuladas)} linhas, {mapa.shape[0]} grupos.")
    print(por_particao.to_string(index=False))
    print(resumo.to_string(index=False))
    print(f"Mapa: {ARQUIVO_MAPA} (sha256 {sha256(ARQUIVO_MAPA)})")


if __name__ == "__main__":
    main()
