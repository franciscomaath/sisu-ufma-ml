import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df_limpo_sintetico():
    """Parquet limpo sintético de 2017 a 2023: 8 cursos × 2 modalidades × 2 edições, 30 inscrições cada.

    A nota é a mesma nas 5 provas e sem bônus, para coincidir com a fórmula da D12.
    """
    rng = np.random.default_rng(0)
    linhas = []
    for ano in range(2017, 2024):
        for curso in range(8):
            for modalidade in ["AC", "PP_RENDA"]:
                for edicao in [1, 2]:
                    corte = round(600 + 15 * curso + rng.normal(0, 10), 2)
                    for _ in range(30):
                        nota = round(corte + rng.normal(-40, 40), 1)
                        linhas.append({
                            "ANO": ano, "EDICAO": edicao, "CODIGO_CAMPUS": 1,
                            "NOME_CURSO_NORM": f"CURSO {curso}", "GRAU": "Bacharelado",
                            "TURNO": "Integral", "MODALIDADE": modalidade, "OPCAO": int(rng.integers(1, 3)),
                            "PERCENTUAL_BONUS": 0.0, "NOTA_L": nota, "NOTA_CH": nota, "NOTA_CN": nota,
                            "NOTA_M": nota, "NOTA_R": nota, "NOTA_CANDIDATO": nota, "NOTA_CORTE": corte,
                            "TEM_NOTA_ZERO": 0, "APROVADO": int(nota >= corte),
                            "NOTA_SUFICIENTE": int(nota >= corte),
                        })
    df = pd.DataFrame(linhas)
    df["NOTA_SUFICIENTE"] = df["NOTA_SUFICIENTE"].astype("Int8")
    df.loc[df.index[5], ["NOTA_CORTE", "NOTA_SUFICIENTE"]] = [np.nan, pd.NA]  # uma linha sem alvo
    return df


@pytest.fixture
def df_bruto():
    """Mini-conjunto sintético com as colunas do CSV bruto."""
    linhas = []
    for ano in [2017, 2018, 2019]:
        for i, (nota, ap) in enumerate([(700.0, "S"), (500.0, "N"), (0.0, "N")]):
            linhas.append({
                "ANO": ano, "EDICAO": 1, "CODIGO_CAMPUS": 1, "NOME_CAMPUS": "X",
                "MUNICIPIO_CAMPUS": "São Luís", "CODIGO_CURSO": 10, "NOME_CURSO": " Direito ",
                "GRAU": "Bacharelado", "TURNO": "Noturno", "TP_COTA": None,
                "TIPO_MOD_CONCORRENCIA": "A", "MOD_CONCORRENCIA": "Ampla concorrência",
                "QT_VAGAS_CONCORRENCIA": None, "PERCENTUAL_BONUS": 0.0, "SEXO": "F",
                "DT_NASCIMENTO": [2000.0, 2068.0, 1990.0][i], "UF_CANDIDATO": "MA",
                "MUNICIPIO_CANDIDATO": "X", "OPCAO": 1, "NOTA_L": nota, "NOTA_CH": nota,
                "NOTA_CN": nota, "NOTA_M": nota, "NOTA_R": 600, "NOTA_CANDIDATO": nota,
                "NOTA_CORTE": 600.0 + (ano - 2017) * 10, "CLASSIFICACAO": float(i + 1),
                "APROVADO": ap, "MATRICULA": "PENDENTE", "GRUPO_CONCORRENCIA": "AC",
                "SUBGRUPO_COTA": "NAO_APLICA",
            })
    return pd.DataFrame(linhas)
