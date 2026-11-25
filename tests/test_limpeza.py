import pandas as pd

from src import config
from src.limpeza import limpar, modalidade_canonica


def test_alvo_binario_e_idade_fora_da_faixa(df_bruto):
    out = limpar(df_bruto)
    assert set(out[config.ALVO].unique()) <= {0, 1}
    assert out["IDADE"].isna().sum() == 3  # nascido em 2068, uma linha por ano
    assert out["IDADE"].dropna().between(14, 80).all()


def test_nota_igual_ao_corte_e_suficiente_sem_aprovacao(df_bruto):
    i = df_bruto.index[1]
    df_bruto.loc[i, "NOTA_CANDIDATO"] = df_bruto.loc[i, "NOTA_CORTE"]
    out = limpar(df_bruto)
    assert out.loc[i, "NOTA_SUFICIENTE"] == 1
    assert out.loc[i, "APROVADO"] == 0
    assert out.loc[2, "NOTA_SUFICIENTE"] == 0


def test_corte_ausente_deixa_alvo_indeterminado(df_bruto):
    i = df_bruto.index[0]
    df_bruto.loc[i, "NOTA_CORTE"] = float("nan")
    log = {}
    out = limpar(df_bruto, log)
    assert pd.isna(out.loc[i, "NOTA_SUFICIENTE"])
    assert out.loc[i, "APROVADO"] == 1
    assert log["linhas_sem_alvo"] == 1


def test_nota_zero_marcada_e_linhas_mantidas(df_bruto):
    out = limpar(df_bruto)
    assert len(out) == len(df_bruto)
    assert out["TEM_NOTA_ZERO"].sum() == 3


def test_colunas_descartadas_e_idempotencia(df_bruto):
    out = limpar(df_bruto)
    for c in ["TP_COTA", "QT_VAGAS_CONCORRENCIA", "CLASSIFICACAO", "MATRICULA", "DT_NASCIMENTO"]:
        assert c not in out.columns
    assert "NOTA_CORTE" in out.columns  # só para o histórico
    assert limpar(df_bruto).equals(out)  # determinística


def test_modalidade_canonica():
    assert modalidade_canonica("Ampla concorrência") == "AC"
    t = "Candidatos autodeclarados pretos ou pardos, com renda familiar bruta per capita igual ou inferior a 1,5 salário mínimo"
    assert modalidade_canonica(t) == "PP_RENDA"
    assert modalidade_canonica("Candidatos que, independentemente da renda (art. 14)") == "INDEP"
