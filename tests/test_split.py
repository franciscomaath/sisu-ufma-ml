from src import config
from src.limpeza import limpar
from src.split import dividir_temporal


def test_anos_disjuntos(df_bruto):
    tr, va, te = dividir_temporal(limpar(df_bruto))
    a, b, c = set(tr.ANO), set(va.ANO), set(te.ANO)
    assert not (a & b) and not (a & c) and not (b & c)


def test_anos_da_config():
    todos = config.ANOS_TREINO + config.ANOS_VALIDACAO + config.ANOS_TESTE
    assert len(todos) == len(set(todos))
    assert max(config.ANOS_TREINO) < min(config.ANOS_VALIDACAO) < min(config.ANOS_TESTE)
    assert not set(config.ANOS_HISTORICO) & set(todos)
