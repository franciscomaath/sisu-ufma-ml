"""Verifica a preparação dos dados reais, sem alterar dados, alvo ou histórico.

Não treina modelos nem calcula métricas de desempenho. Gera tabelas em `reports/`.
Uso: python -m src.verificar_dados (depois de python -m src.limpeza).
"""
import numpy as np
import pandas as pd

from src import config
from src.features import adicionar_corte_anterior, construir_historico
from src.limpeza import NOTAS_PROVAS

DIR_RELATORIOS = config.RAIZ / "reports"
COLUNAS_CORTE = ["CORTE_ANO_ANTERIOR", "DIF_CORTE_ANTERIOR"]
TAMANHO_AMOSTRA = 1000
TOLERANCIA = 1e-9  # abaixo disso, a diferença é ruído de ponto flutuante


def nome_conjunto(ano: int) -> str:
    """Conjunto da divisão temporal (D4) a que o ano pertence."""
    for nome, anos in [("Histórico", config.ANOS_HISTORICO), ("Treino", config.ANOS_TREINO),
                       ("Validação", config.ANOS_VALIDACAO), ("Teste", config.ANOS_TESTE)]:
        if ano in anos:
            return nome
    return "Fora da divisão"


def contagens_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """Inscrições, linhas sem alvo e positivos por ano."""
    t = df.groupby("ANO")[config.ALVO].agg(inscricoes="size", rotuladas="count", suficientes="sum")
    t = t.astype("int64").reset_index()
    t.insert(1, "conjunto", t["ANO"].map(nome_conjunto))
    t.insert(4, "sem_alvo", t["inscricoes"] - t["rotuladas"])
    t["taxa_%"] = (100 * t["suficientes"] / t["rotuladas"]).round(2)
    return t


def contagens_por_conjunto(por_ano: pd.DataFrame) -> pd.DataFrame:
    """Soma a tabela por ano em cada conjunto, na ordem temporal."""
    colunas = ["inscricoes", "rotuladas", "sem_alvo", "suficientes"]
    t = por_ano.groupby("conjunto", sort=False)[colunas].sum().reset_index()
    t["taxa_%"] = (100 * t["suficientes"] / t["rotuladas"]).round(2)
    return t


def resumo_sem_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """Motivo da falta de alvo e aprovação oficial dessas linhas, por ano."""
    sem = df[df[config.ALVO].isna()]
    sem = sem.assign(
        sem_corte=sem["NOTA_CORTE"].isna(),
        sem_nota=sem["NOTA_CANDIDATO"].isna(),
        aprovado=sem["APROVADO"].eq(1),
    )
    t = sem.groupby("ANO").agg(
        sem_alvo=("sem_corte", "size"), sem_corte=("sem_corte", "sum"),
        sem_nota=("sem_nota", "sum"), aprovado_1=("aprovado", "sum"),
    )
    return t.astype("int64").reset_index()


def calcular_nota_centesimos(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Nota em centésimos, em aritmética inteira, com arredondamento "meio para cima".

    Retorna (centésimos, empate). `empate` marca as notas exatamente no meio de dois
    centésimos. Exige notas com no máximo 1 casa decimal e bônus inteiro, sem ausentes.
    """
    decimos = df[NOTAS_PROVAS] * 10
    bonus = df["PERCENTUAL_BONUS"]
    if decimos.isna().any().any() or bonus.isna().any():
        raise ValueError("Há notas ou bônus ausentes.")
    if not np.allclose(decimos, decimos.round(), rtol=0, atol=1e-6) or not (bonus == bonus.round()).all():
        raise ValueError("A conta exata exige notas com 1 casa decimal e bônus inteiro.")
    soma = decimos.round().astype("int64").sum(axis=1)
    # nota = soma/10/5 * (100 + bônus)/100  ->  centésimos = soma * (100 + bônus) / 50
    numerador = soma * (100 + bonus.round().astype("int64"))
    return (numerador + 25) // 50, (numerador % 50) == 25


def regras_de_calculo(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Variantes da fórmula da D12, para comparar com a nota original."""
    sem_arredondar = df[NOTAS_PROVAS].mean(axis=1, skipna=False) * (1 + df["PERCENTUAL_BONUS"] / 100)
    centesimos, _ = calcular_nota_centesimos(df)
    return {
        "Sem arredondar": sem_arredondar,
        "Meio para cima, 2 casas (conta inteira)": centesimos / 100,
        "round(2) do pandas": sem_arredondar.round(2),
        "Truncar em 2 casas": np.floor(sem_arredondar * 100 + 1e-7) / 100,
    }


def comparar_formula(df: pd.DataFrame) -> pd.DataFrame:
    """Divergências de cada regra contra NOTA_CANDIDATO e efeito que teriam no alvo."""
    original = df["NOTA_CANDIDATO"]
    com_alvo = original.notna() & df["NOTA_CORTE"].notna()
    alvo = original >= df["NOTA_CORTE"]
    linhas = []
    for nome, calculada in regras_de_calculo(df).items():
        dif = (calculada - original).abs()
        alvo_alternativo = calculada >= df["NOTA_CORTE"]
        linhas.append({
            "regra": nome,
            "linhas": len(df),
            "divergentes": int((dif > TOLERANCIA).sum()),
            "maior_diferenca": f"{dif.max():.6f}",
            "alvo_1_para_0": int((com_alvo & alvo & ~alvo_alternativo).sum()),
            "so_ponto_flutuante": int(
                (com_alvo & (alvo != alvo_alternativo) & (dif <= TOLERANCIA)).sum()),
            "alvo_0_para_1": int((com_alvo & ~alvo & alvo_alternativo).sum()),
        })
    return pd.DataFrame(linhas)


def formula_por_ano(df: pd.DataFrame) -> pd.DataFrame:
    """Por ano: efeito de não arredondar e empates no meio do centésimo."""
    regras = regras_de_calculo(df)
    sem = regras["Sem arredondar"]
    _, empate = calcular_nota_centesimos(df)
    base = pd.DataFrame({
        "ANO": df["ANO"],
        "divergentes_sem_arredondar": (sem - df["NOTA_CANDIDATO"]).abs() > TOLERANCIA,
        "alvo_mudaria_sem_arredondar": (df["NOTA_CANDIDATO"] >= df["NOTA_CORTE"]) & (sem < df["NOTA_CORTE"]),
        "empates_meio_centesimo": empate,
        "divergentes_round2": (regras["round(2) do pandas"] - df["NOTA_CANDIDATO"]).abs() > TOLERANCIA,
    })
    return base.groupby("ANO").sum().astype("int64").reset_index()


def conflitos_de_corte(df: pd.DataFrame) -> dict[str, int]:
    """Confere se cada chave (curso, modalidade, edição, ano) tem um único corte."""
    chave = config.CHAVE_HISTORICO + ["ANO"]
    g = df.groupby(chave, dropna=False)["NOTA_CORTE"].agg(
        linhas="size", distintos="nunique", preenchidos="count")
    mistas = (g["preenchidos"] > 0) & (g["preenchidos"] < g["linhas"])
    return {
        "Linhas com algum campo da chave ausente": int(df[chave].isna().any(axis=1).sum()),
        "Chaves (curso, modalidade, edição, ano)": len(g),
        "Chaves com mais de um corte": int((g["distintos"] > 1).sum()),
        "Chaves que misturam corte preenchido e ausente": int(mistas.sum()),
        "Chaves sem nenhum corte": int((g["preenchidos"] == 0).sum()),
        "Linhas nas chaves sem nenhum corte": int(g.loc[g["preenchidos"] == 0, "linhas"].sum()),
    }


def cobertura_detalhada(montado: pd.DataFrame) -> pd.DataFrame:
    """Cobertura do corte anterior por ano: geral, por edição e nas linhas com alvo."""
    linhas = []
    for ano, g in montado.groupby("ANO"):
        if ano in config.ANOS_HISTORICO:
            continue
        tem = g["CORTE_ANO_ANTERIOR"].notna()
        linha = {"ANO": ano, "conjunto": nome_conjunto(ano), "linhas": len(g),
                 "com_corte": int(tem.sum()), "cobertura_%": round(100 * tem.mean(), 1)}
        for edicao in sorted(montado["EDICAO"].unique()):
            na_edicao = tem[g["EDICAO"] == edicao]
            linha[f"cobertura_ed{edicao}_%"] = round(100 * na_edicao.mean(), 1) if len(na_edicao) else np.nan
        rotulada = g[config.ALVO].notna()
        linha["rotuladas_com_corte"] = int((tem & rotulada).sum())
        linhas.append(linha)
    return pd.DataFrame(linhas)


def _linhas_diferentes(a: pd.DataFrame, b: pd.DataFrame, tolerancia: float = 0.0) -> int:
    """Número de linhas em que alguma coluna de corte difere (NaN igual a NaN)."""
    x = a[COLUNAS_CORTE].to_numpy(dtype=float)
    y = b[COLUNAS_CORTE].to_numpy(dtype=float)
    iguais = (np.abs(x - y) <= tolerancia) | (np.isnan(x) & np.isnan(y))
    return int((~iguais.all(axis=1)).sum())


def consistencia_montagem(df: pd.DataFrame, versionado: pd.DataFrame,
                          tamanho_amostra: int = TAMANHO_AMOSTRA) -> pd.DataFrame:
    """Compara a montagem em lote (treino) com cenários que imitam a inferência.

    Usa só `adicionar_corte_anterior`: não há função de montagem de inferência a testar.
    """
    historico = construir_historico(df)
    lote = adicionar_corte_anterior(df, historico)
    resultados = []

    def registrar(nome: str, comparadas: int, diferentes: int, acima: int | None = None) -> None:
        resultados.append({"verificacao": nome, "linhas_comparadas": comparadas,
                           "linhas_diferentes": diferentes,
                           "acima_de_1e-9": diferentes if acima is None else acima})

    registrar("Histórico versionado x reconstruído (montagem em lote)",
              len(df), _linhas_diferentes(lote, adicionar_corte_anterior(df, versionado)))

    comparadas = diferentes = 0
    for ano in sorted(df["ANO"].unique()):
        mascara = df["ANO"] == ano
        so_passado = adicionar_corte_anterior(df[mascara], historico[historico["ANO"] < ano])
        comparadas += int(mascara.sum())
        diferentes += _linhas_diferentes(lote[mascara.to_numpy()], so_passado)
    registrar("Histórico completo x só anos anteriores", comparadas, diferentes)

    posicoes = pd.Series(np.arange(len(df))).sample(
        n=min(tamanho_amostra, len(df)), random_state=config.SEMENTE).to_numpy()
    uma_a_uma = pd.concat(
        [adicionar_corte_anterior(df.iloc[[i]], versionado) for i in posicoes], ignore_index=True)
    registrar("Lote x uma inscrição por vez (amostra)",
              len(posicoes), _linhas_diferentes(lote.iloc[posicoes], uma_a_uma))

    regras = regras_de_calculo(df)
    for nome in ["Meio para cima, 2 casas (conta inteira)", "Sem arredondar"]:
        recalculado = df.assign(NOTA_CANDIDATO=regras[nome])
        montado = adicionar_corte_anterior(recalculado, versionado)
        registrar(f"Nota original x recalculada: {nome.lower()}", len(df),
                  _linhas_diferentes(lote, montado),
                  _linhas_diferentes(lote, montado, TOLERANCIA))
    return pd.DataFrame(resultados)


def tabela_markdown(t: pd.DataFrame) -> str:
    """Tabela em Markdown, sem depender do pacote tabulate."""
    def texto(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:g}"
        return str(v)

    linhas = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    for registro in t.itertuples(index=False, name=None):
        linhas.append("| " + " | ".join(texto(v) for v in registro) + " |")
    return "\n".join(linhas)


def main() -> None:
    for caminho, comando in [(config.PARQUET_LIMPO, "src.limpeza"), (config.HISTORICO_CORTES, "src.features")]:
        if not caminho.is_file():
            raise SystemExit(f"Arquivo ausente: {caminho}. Execute python -m {comando}.")
    df = pd.read_parquet(config.PARQUET_LIMPO)
    versionado = pd.read_parquet(config.HISTORICO_CORTES)
    historico = construir_historico(df)

    por_ano = contagens_alvo(df)
    conflitos = pd.DataFrame(conflitos_de_corte(df).items(), columns=["verificacao", "quantidade"])
    igual = "sim" if historico.equals(versionado) else "não"

    secoes = [
        ("Alvo por ano", "alvo_por_ano", por_ano),
        ("Alvo por conjunto", "alvo_por_conjunto", contagens_por_conjunto(por_ano)),
        ("Linhas sem alvo", "sem_alvo", resumo_sem_alvo(df)),
        ("Fórmula da nota: regras comparadas", "formula_regras", comparar_formula(df)),
        ("Fórmula da nota: efeito por ano", "formula_por_ano", formula_por_ano(df)),
        ("Cortes por chave", "conflitos_corte", conflitos),
        ("Cobertura do corte anterior", "cobertura", cobertura_detalhada(
            adicionar_corte_anterior(df, historico))),
        ("Montagem em lote x cenários de inferência", "montagem", consistencia_montagem(df, versionado)),
    ]

    DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    texto = [
        "# Verificação da preparação dos dados",
        "",
        "Arquivo gerado por `python -m src.verificar_dados`. Não editar à mão.",
        "Não treina modelos nem calcula métricas de desempenho.",
        "",
        f"Linhas no parquet limpo: {len(df)}.",
        f"Histórico versionado idêntico ao reconstruído: {igual} ({len(versionado)} linhas).",
    ]
    for titulo, arquivo, tabela in secoes:
        tabela.to_csv(DIR_RELATORIOS / f"verificacao_{arquivo}.csv", index=False, lineterminator="\n")
        texto += ["", f"## {titulo}", "", tabela_markdown(tabela)]
    saida = "\n".join(texto) + "\n"
    (DIR_RELATORIOS / "verificacao_dados.md").write_text(saida, encoding="utf-8", newline="\n")
    print(saida)


if __name__ == "__main__":
    main()
