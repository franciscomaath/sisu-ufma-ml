"""Limpeza determinística do CSV bruto do SISU/UFMA (sem aprender nada dos dados)."""
import re
import unicodedata

import numpy as np
import pandas as pd

from src import config

NOTAS_PROVAS = ["NOTA_L", "NOTA_CH", "NOTA_CN", "NOTA_M", "NOTA_R"]

# Colunas descartadas e motivo (usado no relatório).
COLUNAS_DESCARTADAS = {
    "TP_COTA": "89% de nulos",
    "QT_VAGAS_CONCORRENCIA": "79% de nulos",
    "NOME_CAMPUS": "redundante com CODIGO_CAMPUS/MUNICIPIO_CAMPUS",
    "CLASSIFICACAO": "vazamento (só existe após o resultado)",
    "MATRICULA": "vazamento (só existe após o resultado)",
    "DT_NASCIMENTO": "substituída por IDADE",
    "MOD_CONCORRENCIA": "texto instável entre anos; substituído por MODALIDADE",
    "TIPO_MOD_CONCORRENCIA": "código inconsistente entre anos; coberto por MODALIDADE",
    "SUBGRUPO_COTA": "mudou de significado em 2021 (SOCIAL virou D); coberto por MODALIDADE",
    "CODIGO_CURSO": "numeração instável (2018/1 usa outra); curso identificado por CODIGO_CAMPUS+NOME+GRAU+TURNO",
    "NOME_CURSO": "mantido só como NOME_CURSO_NORM",
}


def normalizar_texto(s) -> str:
    """Maiúsculas, sem acentos e sem espaços repetidos."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.upper()).strip()


def modalidade_canonica(texto: str) -> str:
    """Deriva uma modalidade estável a partir do texto de MOD_CONCORRENCIA.

    Componentes: etnia (PP/I), deficiência (D) e renda (RENDA/INDEP). Os códigos de
    SUBGRUPO_COTA e TIPO_MOD_CONCORRENCIA mudaram de significado ao longo dos anos.
    """
    t = normalizar_texto(texto)
    if t.startswith("AMPLA"):
        return "AC"
    if "ENSINO FUNDAMENTAL (9" in t:
        return "BONUS_MA"
    if "CONCORRENTES" in t and "AMPLA" in t:
        return "AC_MEDICINA_REGIONAL"  # recorte regional de Medicina (só 2023)
    partes = []
    if "PRETOS OU PARDOS" in t:
        partes.append("PP")
    elif "INDIGENAS" in t:
        partes.append("I")
    if "DEFICIENCIA" in t:
        partes.append("D")
    com_renda = "RENDA FAMILIAR" in t and "INDEPENDENTEMENTE DA RENDA" not in t
    partes.append("RENDA" if com_renda else "INDEP")
    return "_".join(partes)


def marcar_nota_zero(df: pd.DataFrame) -> pd.Series:
    """TEM_NOTA_ZERO: 1 se alguma das 5 provas for 0. Vale para a limpeza e a montagem (D13)."""
    return (df[NOTAS_PROVAS] == 0).any(axis=1).astype("int8")


def limpar(df: pd.DataFrame, log: dict | None = None) -> pd.DataFrame:
    """Aplica a limpeza. Se `log` for um dict, registra as linhas afetadas por decisão."""
    log = log if log is not None else {}
    df = df.copy()

    # 1) Construir o alvo e manter a aprovação oficial para comparação.
    notas_disponiveis = df[["NOTA_CANDIDATO", "NOTA_CORTE"]].notna().all(axis=1)
    nota_suficiente = df["NOTA_CANDIDATO"] >= df["NOTA_CORTE"]
    df[config.ALVO] = nota_suficiente.astype("Int8").where(notas_disponiveis)
    log["linhas_sem_alvo"] = int((~notas_disponiveis).sum())
    df["APROVADO"] = (df["APROVADO"].astype(str).str.strip().str.upper() == "S").astype("int8")

    # 2) Idade a partir do ano de nascimento; fora de [14, 80] vira NaN
    idade = df["ANO"] - df["DT_NASCIMENTO"]
    fora = (idade < 14) | (idade > 80)
    log["idade_fora_14_80_viram_nan"] = int(fora.sum())
    log["idade_ja_nula"] = int(idade.isna().sum())
    df["IDADE"] = idade.mask(fora)

    # 3) Notas 0 em provas: marcar (não remover) para não distorcer a população real
    zero = marcar_nota_zero(df)
    log["linhas_com_nota_zero_marcadas"] = int(zero.sum())
    df["TEM_NOTA_ZERO"] = zero

    # 4) Chaves estáveis de curso e modalidade
    df["NOME_CURSO_NORM"] = df["NOME_CURSO"].map(normalizar_texto)
    mapa = {t: modalidade_canonica(t) for t in df["MOD_CONCORRENCIA"].unique()}
    df["MODALIDADE"] = df["MOD_CONCORRENCIA"].map(mapa)

    # 5) Descartes (NOTA_CORTE fica para construir o alvo e o histórico)
    log["colunas_descartadas"] = dict(COLUNAS_DESCARTADAS)
    df = df.drop(columns=[c for c in COLUNAS_DESCARTADAS if c in df.columns])
    return df.reset_index(drop=True)


def main() -> None:
    log: dict = {}
    bruto = pd.read_csv(config.CSV_BRUTO, low_memory=False)
    log["linhas_entrada"] = len(bruto)
    limpo = limpar(bruto, log)
    config.DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    limpo.to_parquet(config.PARQUET_LIMPO, index=False)
    for k, v in log.items():
        print(k, "->", v)
    print(limpo.groupby("ANO")[config.ALVO].agg(
        inscricoes="size", rotuladas="count", suficientes="sum", taxa="mean"
    ))


if __name__ == "__main__":
    main()
