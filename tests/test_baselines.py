import pandas as pd

from src import config
from src.baselines import (
    ESCORE_SEM_HISTORICO, criar_regressao_logistica, escore_regra_corte_anterior, main,
    prever_regra_corte_anterior,
)
from src.features import adicionar_corte_anterior, construir_historico
from src.limpeza import limpar


def test_regra_corte_anterior():
    df = pd.DataFrame({
        "NOTA_CANDIDATO": [650, 600, 550, 650],
        "CORTE_ANO_ANTERIOR": pd.array([600, 600, 600, None], dtype="Float64"),
    })
    assert prever_regra_corte_anterior(df).tolist() == [1, 1, 0, 0]  # empate conta como suficiente


def test_regra_segue_o_alvo_no_empate():
    df = pd.DataFrame({"NOTA_CANDIDATO": [625.94, 625.93], "CORTE_ANO_ANTERIOR": [625.94, 625.94]})
    assert prever_regra_corte_anterior(df).tolist() == [1, 0]


def test_escore_da_regra_poe_sem_historico_no_fim():
    df = pd.DataFrame({"DIF_CORTE_ANTERIOR": pd.array([12.5, -700.0, None], dtype="Float64")})
    escore = escore_regra_corte_anterior(df)
    assert escore.tolist() == [12.5, -700.0, ESCORE_SEM_HISTORICO]
    assert escore.iloc[2] < escore.iloc[1]


def test_logistica_aceita_historico_ausente_e_categoria_nova(df_bruto):
    df = limpar(df_bruto)
    df = adicionar_corte_anterior(df, construir_historico(df))
    treino = df.loc[df["ANO"] == 2018, config.ATRIBUTOS].copy()
    validacao = df.loc[df["ANO"] == 2019, config.ATRIBUTOS].copy()
    historico = ["CORTE_ANO_ANTERIOR", "DIF_CORTE_ANTERIOR"]
    treino.loc[treino.index[0], historico] = float("nan")
    validacao.loc[validacao.index[0], historico] = float("nan")
    validacao.loc[validacao.index[0], "NOME_CURSO_NORM"] = "CURSO NOVO"

    modelo = criar_regressao_logistica()
    modelo.fit(treino, df.loc[treino.index, config.ALVO].astype("int8"))
    probabilidades = modelo.predict_proba(validacao)[:, 1]

    assert len(probabilidades) == len(validacao)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
    numericos = modelo["preprocessamento"].named_transformers_["numericos"]
    assert numericos["imputacao"].indicator_.features_.tolist() == [
        config.ATRIBUTOS_NUMERICOS.index(coluna) for coluna in historico
    ]


def test_comparacao_dos_baselines_no_terminal(df_bruto, tmp_path, monkeypatch, capsys):
    amostra = df_bruto[df_bruto["ANO"] == 2019]
    df = limpar(pd.concat([df_bruto, amostra.assign(ANO=2022), amostra.assign(ANO=2023)], ignore_index=True))
    df.loc[df.index[df["ANO"] == 2022][-1], config.ALVO] = pd.NA
    parquet = tmp_path / "sisu_limpo.parquet"
    df.to_parquet(parquet, index=False)
    monkeypatch.setattr(config, "PARQUET_LIMPO", parquet)

    main()

    saida = capsys.readouterr().out
    assert "Treino: 6 linhas usadas, 0 sem alvo." in saida
    assert "Validação: 2 linhas usadas, 1 sem alvo." in saida
    assert "Regra do corte anterior" in saida
    assert "Logística sem pesos" in saida
    assert "Logística com pesos balanceados" in saida
    assert "PR-AUC (AP)" in saida
    assert "2023" not in saida
