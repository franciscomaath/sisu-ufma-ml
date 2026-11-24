"""Configuração global: semente, caminhos e divisão temporal."""
from pathlib import Path

SEMENTE = 42

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_FIGURAS = RAIZ / "reports" / "figures"

CSV_BRUTO = DIR_RAW / "sisu_ufma_2017_2023.csv"
PARQUET_LIMPO = DIR_PROCESSED / "sisu_limpo.parquet"
HISTORICO_CORTES = DIR_MODELOS / "historico_cortes.parquet"
HISTORICO_CORTES_JOBLIB = DIR_MODELOS / "historico_cortes.joblib"

# Divisão temporal (nunca aleatória). 2017 só serve de histórico de cortes.
ANOS_HISTORICO = [2017]
ANOS_TREINO = [2018, 2019, 2020, 2021]
ANOS_VALIDACAO = [2022]
ANOS_TESTE = [2023]

ALVO = "APROVADO"

# Colunas que vazam o resultado: nunca podem ser atributos.
COLUNAS_VAZAMENTO = ["NOTA_CORTE", "CLASSIFICACAO", "MATRICULA"]

# Identificação da inscrição / auxiliares que não entram como atributos do modelo.
COLUNAS_NAO_ATRIBUTO = [ALVO, "ANO"] + COLUNAS_VAZAMENTO

# Chave do histórico de cortes (sem o ano). O ano anterior é resolvido na junção.
CHAVE_CURSO = ["CODIGO_CAMPUS", "NOME_CURSO_NORM", "GRAU", "TURNO"]
CHAVE_HISTORICO = CHAVE_CURSO + ["MODALIDADE", "EDICAO"]
