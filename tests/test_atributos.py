"""D2: lista explícita de atributos e seleção estrita."""
import pandas as pd
import pytest

from src import config
from src.features import colunas_atributos, selecionar_atributos

PROIBIDAS = [
    "NOTA_CORTE", "CLASSIFICACAO", "MATRICULA",   # vazamento
    "NOTA_SUFICIENTE", "APROVADO",                 # alvo atual e alvo antigo
    "ANO",                                         # o modelo precisa generalizar para anos novos
    "SEXO", "IDADE", "UF_CANDIDATO", "MUNICIPIO_CANDIDATO",   # demográficas (ética, D2)
    "MUNICIPIO_CAMPUS", "GRUPO_CONCORRENCIA",      # redundantes
]


def test_listas_da_decisao_d2():
    assert config.ATRIBUTOS_NUMERICOS == [
        "NOTA_L", "NOTA_CH", "NOTA_CN", "NOTA_M", "NOTA_R", "NOTA_CANDIDATO",
        "PERCENTUAL_BONUS", "CORTE_ANO_ANTERIOR", "DIF_CORTE_ANTERIOR", "TEM_NOTA_ZERO"]
    assert config.ATRIBUTOS_CATEGORICOS == [
        "CODIGO_CAMPUS", "NOME_CURSO_NORM", "GRAU", "TURNO", "MODALIDADE", "EDICAO", "OPCAO"]
    assert config.ATRIBUTOS == config.ATRIBUTOS_NUMERICOS + config.ATRIBUTOS_CATEGORICOS


def test_listas_sem_repeticao_nem_sobreposicao():
    assert len(config.ATRIBUTOS) == len(set(config.ATRIBUTOS))
    assert not set(config.ATRIBUTOS_NUMERICOS) & set(config.ATRIBUTOS_CATEGORICOS)


def _tabela(colunas):
    return pd.DataFrame({coluna: [1] for coluna in colunas})


@pytest.mark.parametrize("coluna", PROIBIDAS)
def test_coluna_proibida_fora_dos_atributos(coluna):
    assert coluna not in config.ATRIBUTOS
    assert coluna not in colunas_atributos(_tabela(config.ATRIBUTOS + [coluna]))


def test_colunas_atributos_devolve_copia():
    colunas_atributos(_tabela(config.ATRIBUTOS)).append("SEXO")
    assert "SEXO" not in config.ATRIBUTOS


def test_selecao_estrita_descarta_extras_e_mantem_ordem():
    df = _tabela(config.ATRIBUTOS + PROIBIDAS)
    df = df[df.columns[::-1]]  # embaralha a ordem
    out = selecionar_atributos(df)
    assert list(out.columns) == config.ATRIBUTOS


def test_selecao_com_atributo_ausente_da_erro_claro():
    df = _tabela(config.ATRIBUTOS).drop(columns=["OPCAO"])
    with pytest.raises(ValueError, match="OPCAO"):
        selecionar_atributos(df)
