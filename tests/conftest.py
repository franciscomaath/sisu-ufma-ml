import pandas as pd
import pytest


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
