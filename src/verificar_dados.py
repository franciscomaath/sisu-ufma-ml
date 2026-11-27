"""Verifica a preparação dos dados reais, sem alterar dados, alvo ou histórico.

Não treina modelos nem calcula métricas de desempenho. Gera tabelas em `reports/conciliacao/`.
Uso: python -m src.verificar_dados (depois de python -m src.limpeza).
"""
import numpy as np
import pandas as pd

from src import config
from src.features import (
    COLUNAS_ENTRADA, adicionar_corte_anterior, calcular_nota_candidato, calcular_nota_centesimos,
    colunas_atributos, construir_historico, montar_entrada_inferencia, montar_entrada_treino,
)
from src.limpeza import NOTAS_PROVAS

DIR_RELATORIOS = config.RAIZ / "reports" / "conciliacao"
COLUNAS_CORTE = ["CORTE_ANO_ANTERIOR", "DIF_CORTE_ANTERIOR"]
COLUNAS_PROIBIDAS = config.COLUNAS_VAZAMENTO + [
    config.ALVO, "APROVADO", "ANO", "SEXO", "IDADE", "UF_CANDIDATO", "MUNICIPIO_CANDIDATO",
    "MUNICIPIO_CAMPUS", "GRUPO_CONCORRENCIA",
]
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


def regras_de_calculo(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Variantes da fórmula da D12, para comparar com a nota original."""
    sem_arredondar = df[NOTAS_PROVAS].mean(axis=1, skipna=False) * (1 + df["PERCENTUAL_BONUS"] / 100)
    return {
        "Sem arredondar": sem_arredondar,
        "Meio para cima, 2 casas (conta inteira)": calcular_nota_candidato(df),
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


def equivalencia_da_nota(df: pd.DataFrame) -> pd.DataFrame:
    """Nota da função compartilhada contra a original, com igualdade exata, por ano."""
    recalculada = calcular_nota_candidato(df)
    _, empate = calcular_nota_centesimos(df)
    base = pd.DataFrame({
        "ANO": df["ANO"],
        "linhas": 1,
        "diferentes": recalculada.ne(df["NOTA_CANDIDATO"]),
        "empates_meio_centesimo": empate,
        "alvo_diferente": df[config.ALVO].notna() & (
            recalculada.ge(df["NOTA_CORTE"]) != df["NOTA_CANDIDATO"].ge(df["NOTA_CORTE"])),
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


def linhas_diferentes(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Linhas em que alguma coluna difere, comparando por posição (ausente igual a ausente)."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise ValueError("As tabelas comparadas têm colunas ou tamanhos diferentes.")
    x, y = a.reset_index(drop=True), b.reset_index(drop=True)
    iguais = x.eq(y) | (x.isna() & y.isna())
    return int((~iguais).any(axis=1).sum())


def consistencia_treino_inferencia(df: pd.DataFrame, versionado: pd.DataFrame,
                                   tamanho_amostra: int = TAMANHO_AMOSTRA) -> pd.DataFrame:
    """Compara as entradas de treino com as de inferência, nas 17 colunas.

    A inferência recebe só COLUNAS_ENTRADA (o que a aplicação teria) e usa o histórico
    versionado; o treino usa o histórico reconstruído do parquet.
    """
    treino = montar_entrada_treino(df, construir_historico(df))
    do_parquet = [c for c in config.ATRIBUTOS if c not in COLUNAS_CORTE]
    linhas = []
    for ano, posicoes in df.groupby("ANO").indices.items():
        referencia = treino.iloc[posicoes]
        inferida = montar_entrada_inferencia(df.iloc[posicoes][COLUNAS_ENTRADA], versionado, int(ano))
        linhas.append({
            "cenario": str(ano),
            "linhas": len(posicoes),
            "treino_x_inferencia": linhas_diferentes(referencia, inferida),
            "tipos_e_indice_iguais": "sim" if referencia.dtypes.equals(inferida.dtypes)
            and referencia.index.equals(inferida.index) else "não",
            "treino_x_parquet": linhas_diferentes(df.iloc[posicoes][do_parquet], referencia[do_parquet]),
        })

    posicoes = pd.Series(np.arange(len(df))).sample(
        n=min(tamanho_amostra, len(df)), random_state=config.SEMENTE).to_numpy()
    diferentes = tipos_diferentes = 0
    for i in posicoes:
        # reconstrói a inscrição a partir de valores soltos, como faria um formulário
        registro = df.iloc[[i]][COLUNAS_ENTRADA].to_dict("records")[0]
        inferida = montar_entrada_inferencia(
            pd.DataFrame([registro], index=[df.index[i]]), versionado, int(df["ANO"].iat[i]))
        diferentes += linhas_diferentes(treino.iloc[[i]], inferida)
        tipos_diferentes += not treino.iloc[[i]].dtypes.equals(inferida.dtypes)
    linhas.append({"cenario": "Amostra, uma inscrição por vez", "linhas": len(posicoes),
                   "treino_x_inferencia": diferentes,
                   "tipos_e_indice_iguais": "sim" if tipos_diferentes == 0 else "não",
                   "treino_x_parquet": "-"})
    return pd.DataFrame(linhas)


def corte_por_consulta_direta(df: pd.DataFrame) -> pd.Series:
    """Segunda implementação do corte N−1, sem `merge` nem `construir_historico`."""
    chave = config.CHAVE_HISTORICO + ["ANO"]
    preenchidos = df[chave + ["NOTA_CORTE"]].dropna(subset=["NOTA_CORTE"]).drop_duplicates()
    cortes = {linha[:-1]: linha[-1] for linha in preenchidos.itertuples(index=False, name=None)}
    valores = [cortes.get((*linha[:-1], linha[-1] - 1), np.nan)
               for linha in df[chave].itertuples(index=False, name=None)]
    return pd.Series(valores, index=df.index, dtype="float64")


def uso_do_ano_anterior(df: pd.DataFrame) -> pd.DataFrame:
    """Evidências de que só o corte de N−1 influencia as entradas de cada ano N."""
    historico = construir_historico(df)
    treino = montar_entrada_treino(df, historico)
    direta = corte_por_consulta_direta(df)
    linhas = []
    for ano, posicoes in df.groupby("ANO").indices.items():
        entrada, referencia = df.iloc[posicoes], treino.iloc[posicoes]
        e_anterior = historico["ANO"] == ano - 1
        adulterado = historico.assign(
            CORTE_ANO_ANTERIOR=historico["CORTE_ANO_ANTERIOR"].where(e_anterior, -1.0))
        sem_anterior = montar_entrada_inferencia(entrada, historico[~e_anterior], int(ano))
        linhas.append({
            "ANO": ano,
            "linhas": len(posicoes),
            "so_historico_n_1": linhas_diferentes(
                referencia, montar_entrada_inferencia(entrada, historico[e_anterior], int(ano))),
            "outros_anos_adulterados": linhas_diferentes(
                referencia, montar_entrada_inferencia(entrada, adulterado, int(ano))),
            "corte_do_proprio_ano_adulterado": linhas_diferentes(
                referencia, montar_entrada_treino(entrada.assign(NOTA_CORTE=-1.0), historico)),
            "consulta_direta": linhas_diferentes(
                referencia[["CORTE_ANO_ANTERIOR"]],
                direta.iloc[posicoes].to_frame("CORTE_ANO_ANTERIOR")),
            "com_corte_sem_historico_n_1": int(sem_anterior["CORTE_ANO_ANTERIOR"].notna().sum()),
        })
    return pd.DataFrame(linhas)


def atributos_nas_entradas(df: pd.DataFrame, entradas: pd.DataFrame) -> pd.DataFrame:
    """Presença de cada coluna proibida no parquet e nas entradas montadas."""
    if list(entradas.columns) != colunas_atributos(entradas):
        raise ValueError("As entradas não têm exatamente os atributos da D2, na ordem.")
    return pd.DataFrame([{
        "coluna_proibida": coluna,
        "no_parquet": "sim" if coluna in df.columns else "não",
        "nas_entradas": "sim" if coluna in entradas.columns else "não",
    } for coluna in COLUNAS_PROIBIDAS])


def sem_historico(df: pd.DataFrame) -> pd.DataFrame:
    """Por ano: linhas sem corte anterior, o motivo e como as entradas ficam."""
    historico = construir_historico(df)
    entradas = montar_entrada_treino(df, historico)
    chave = config.CHAVE_HISTORICO + ["ANO"]
    seguinte = historico.assign(ANO=historico["ANO"] + 1)
    existe = pd.Series(pd.MultiIndex.from_frame(df[chave]).isin(pd.MultiIndex.from_frame(seguinte[chave])),
                       index=df.index)
    sem_corte = entradas["CORTE_ANO_ANTERIOR"].isna()
    base = pd.DataFrame({
        "ANO": df["ANO"],
        "linhas": 1,
        "sem_corte": sem_corte,
        "chave_inexistente_em_n_1": ~existe,
        "chave_sem_valor_em_n_1": existe & sem_corte,
        "corte_e_dif_incoerentes": sem_corte != entradas["DIF_CORTE_ANTERIOR"].isna(),
        "outros_atributos_ausentes": entradas.drop(columns=COLUNAS_CORTE).isna().any(axis=1),
    })
    return base.groupby("ANO").sum().astype("int64").reset_index()


def cenarios_sem_historico(df: pd.DataFrame, versionado: pd.DataFrame) -> pd.DataFrame:
    """Inferência com as inscrições do último ano em situações sem histórico (e no ano seguinte)."""
    ultimo = int(df["ANO"].max())
    entrada = df[df["ANO"] == ultimo][COLUNAS_ENTRADA]
    cenarios = [
        (f"Ano-alvo {ultimo + 1} (usa o corte de {ultimo})", entrada, ultimo + 1),
        (f"Ano-alvo {ultimo + 7}, sem histórico de N−1", entrada, ultimo + 7),
        ("Campus que não existe no histórico", entrada.assign(CODIGO_CAMPUS=-1), ultimo + 1),
    ]
    linhas = []
    for nome, tabela, ano_alvo in cenarios:
        saida = montar_entrada_inferencia(tabela, versionado, ano_alvo)
        linhas.append({"cenario": nome, "linhas": len(saida),
                       "com_corte": int(saida["CORTE_ANO_ANTERIOR"].notna().sum()),
                       "sem_corte": int(saida["CORTE_ANO_ANTERIOR"].isna().sum())})
    return pd.DataFrame(linhas)


def entradas_dos_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compara a entrada dos baselines publicados (179593a) com a montagem compartilhada."""
    historico = construir_historico(df[df["ANO"].isin(config.ANOS_HISTORICO + config.ANOS_TREINO)])
    linhas = []
    for nome, anos in [("Treino", config.ANOS_TREINO), ("Validação", config.ANOS_VALIDACAO)]:
        inscricoes = df[df["ANO"].isin(anos)].dropna(subset=[config.ALVO])
        # caminho anterior: junção direta, usando a NOTA_CANDIDATO do CSV
        anterior = adicionar_corte_anterior(inscricoes, historico)
        antiga = anterior[colunas_atributos(anterior)]
        nova = montar_entrada_treino(inscricoes, historico)
        linhas.append({
            "conjunto": nome,
            "linhas": len(inscricoes),
            "linhas_diferentes": linhas_diferentes(antiga, nova),
            "tipos_iguais": "sim" if antiga.dtypes.equals(nova.dtypes) else "não",
        })
    return pd.DataFrame(linhas)


def empates_com_corte_anterior(df: pd.DataFrame) -> pd.DataFrame:
    """Linhas de treino e validação em que a regra `>` e a `>=` discordam (nota igual ao corte)."""
    anos = config.ANOS_TREINO + config.ANOS_VALIDACAO
    rotuladas = df[df["ANO"].isin(anos)].dropna(subset=[config.ALVO])
    entradas = montar_entrada_treino(rotuladas, construir_historico(df[df["ANO"] < max(anos)]))
    base = pd.DataFrame({
        "ANO": rotuladas["ANO"],
        "com_corte": entradas["CORTE_ANO_ANTERIOR"].notna(),
        "nota_igual_ao_corte_anterior": entradas["NOTA_CANDIDATO"].eq(entradas["CORTE_ANO_ANTERIOR"]),
    })
    return base.groupby("ANO").sum().astype("int64").reset_index()


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
    entradas = montar_entrada_treino(df, historico)

    por_ano = contagens_alvo(df)
    conflitos = pd.DataFrame(conflitos_de_corte(df).items(), columns=["verificacao", "quantidade"])
    igual = "sim" if historico.equals(versionado) else "não"
    com_corte = df.assign(CORTE_ANO_ANTERIOR=entradas["CORTE_ANO_ANTERIOR"])

    secoes = [
        ("Alvo por ano", "alvo_por_ano", por_ano),
        ("Alvo por conjunto", "alvo_por_conjunto", contagens_por_conjunto(por_ano)),
        ("Linhas sem alvo", "sem_alvo", resumo_sem_alvo(df)),
        ("Fórmula da nota: regras comparadas", "formula_regras", comparar_formula(df)),
        ("Nota da função compartilhada x original", "nota_equivalencia", equivalencia_da_nota(df)),
        ("Cortes por chave", "conflitos_corte", conflitos),
        ("Cobertura do corte anterior", "cobertura", cobertura_detalhada(com_corte)),
        ("Entradas de treino x inferência", "treino_inferencia",
         consistencia_treino_inferencia(df, versionado)),
        ("Uso exclusivo do corte de N−1", "uso_n_menos_1", uso_do_ano_anterior(df)),
        ("Colunas proibidas", "colunas_proibidas", atributos_nas_entradas(df, entradas)),
        ("Linhas sem corte anterior", "sem_historico", sem_historico(df)),
        ("Cenários de inferência", "cenarios_inferencia", cenarios_sem_historico(df, versionado)),
        ("Entradas dos baselines publicados x montagem compartilhada", "entradas_baselines",
         entradas_dos_baselines(df)),
        ("Empates entre nota e corte anterior (treino e validação)", "empates_regra",
         empates_com_corte_anterior(df)),
    ]

    DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    texto = [
        "# Verificação da preparação dos dados",
        "",
        "Arquivo gerado por `python -m src.verificar_dados`. Não editar à mão.",
        "Não treina modelos nem calcula métricas de desempenho.",
        "Nas tabelas de comparação, os números contam linhas diferentes (0 = iguais).",
        "",
        f"Linhas no parquet limpo: {len(df)}.",
        f"Histórico versionado idêntico ao reconstruído: {igual} ({len(versionado)} linhas).",
        f"Entradas montadas: {entradas.shape[1]} colunas, iguais à lista da D2 e na mesma ordem.",
    ]
    for titulo, arquivo, tabela in secoes:
        tabela.to_csv(DIR_RELATORIOS / f"verificacao_{arquivo}.csv", index=False, lineterminator="\n")
        texto += ["", f"## {titulo}", "", tabela_markdown(tabela)]
    saida = "\n".join(texto) + "\n"
    (DIR_RELATORIOS / "verificacao_dados.md").write_text(saida, encoding="utf-8", newline="\n")
    print(saida)


if __name__ == "__main__":
    main()
