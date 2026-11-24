from src.features import adicionar_corte_anterior, construir_historico
from src.limpeza import limpar


def test_corte_anterior_vem_do_ano_anterior(df_bruto):
    df = limpar(df_bruto)
    out = adicionar_corte_anterior(df, construir_historico(df))
    esperado = {2017: None, 2018: 600.0, 2019: 610.0}  # NOTA_CORTE do ano N-1
    for ano, corte in esperado.items():
        vals = out.loc[out.ANO == ano, "CORTE_ANO_ANTERIOR"]
        if corte is None:
            assert vals.isna().all()
        else:
            assert (vals == corte).all()  # nunca o corte do próprio ano


def test_dif_corte_anterior(df_bruto):
    df = limpar(df_bruto)
    out = adicionar_corte_anterior(df, construir_historico(df))
    linha = out[(out.ANO == 2019) & (out.NOTA_CANDIDATO == 700.0)].iloc[0]
    assert linha.DIF_CORTE_ANTERIOR == 700.0 - 610.0


def test_sem_duplicar_linhas(df_bruto):
    df = limpar(df_bruto)
    assert len(adicionar_corte_anterior(df, construir_historico(df))) == len(df)
