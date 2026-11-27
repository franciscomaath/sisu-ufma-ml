import numpy as np
import pandas as pd
import pytest

from src import config
from src import verificar_dados as vd
from src.features import construir_historico, montar_entrada_treino
from src.limpeza import limpar


def _notas(nota: float, bonus: float, original: float, corte: float) -> pd.DataFrame:
    return pd.DataFrame({
        "ANO": [2023], "NOTA_L": [nota], "NOTA_CH": [nota], "NOTA_CN": [nota], "NOTA_M": [nota],
        "NOTA_R": [nota], "PERCENTUAL_BONUS": [bonus],
        "NOTA_CANDIDATO": [original], "NOTA_CORTE": [corte],
        config.ALVO: pd.array([int(original >= corte)], dtype="Int8"),
    })


def test_comparar_formula_mostra_efeito_no_alvo():
    t = vd.comparar_formula(_notas(500.1, 5.0, 525.11, 525.11)).set_index("regra")
    exata = t.loc["Meio para cima, 2 casas (conta inteira)"]
    assert (exata["divergentes"], exata["alvo_1_para_0"]) == (0, 0)
    sem = t.loc["Sem arredondar"]
    assert (sem["divergentes"], sem["alvo_1_para_0"], sem["so_ponto_flutuante"]) == (1, 1, 0)


def test_comparar_formula_separa_ruido_de_ponto_flutuante():
    # A média dessas notas é 625,94, mas em ponto flutuante dá 625,9399999999999
    df = _notas(0.0, 0.0, 625.94, 625.94)
    df[["NOTA_L", "NOTA_CH", "NOTA_CN", "NOTA_M", "NOTA_R"]] = [[627.5, 615.2, 420.3, 646.7, 820.0]]
    t = vd.comparar_formula(df).set_index("regra")
    sem = t.loc["Sem arredondar"]
    assert sem["divergentes"] == 0
    assert (sem["alvo_1_para_0"], sem["so_ponto_flutuante"]) == (1, 1)
    assert t.loc["Meio para cima, 2 casas (conta inteira)", "alvo_1_para_0"] == 0


def test_equivalencia_da_nota_aponta_diferenca():
    certa = vd.equivalencia_da_nota(_notas(500.1, 5.0, 525.11, 525.11))
    assert certa.to_dict("records") == [{
        "ANO": 2023, "linhas": 1, "diferentes": 0, "empates_meio_centesimo": 1, "alvo_diferente": 0}]
    errada = vd.equivalencia_da_nota(_notas(500.1, 5.0, 525.10, 525.11))
    assert (errada["diferentes"].iloc[0], errada["alvo_diferente"].iloc[0]) == (1, 1)


def test_contagens_e_linhas_sem_alvo(df_bruto):
    df_bruto.loc[df_bruto.index[0], "NOTA_CORTE"] = np.nan  # linha de 2017, aprovada
    df = limpar(df_bruto)
    por_ano = vd.contagens_alvo(df)
    assert por_ano["inscricoes"].tolist() == [3, 3, 3]
    assert por_ano["sem_alvo"].tolist() == [1, 0, 0]
    assert por_ano["conjunto"].tolist() == ["Histórico", "Treino", "Treino"]
    conjuntos = vd.contagens_por_conjunto(por_ano)
    assert conjuntos["inscricoes"].tolist() == [3, 6]
    sem = vd.resumo_sem_alvo(df)
    assert sem.to_dict("records") == [
        {"ANO": 2017, "sem_alvo": 1, "sem_corte": 1, "sem_nota": 0, "aprovado_1": 1}]


def test_conflito_de_corte_e_detectado(df_bruto):
    assert vd.conflitos_de_corte(limpar(df_bruto))["Chaves com mais de um corte"] == 0
    df_bruto.loc[df_bruto.index[0], "NOTA_CORTE"] = 999.0
    df_bruto.loc[df_bruto.index[3], "NOTA_CORTE"] = np.nan  # outra chave: 2018
    resumo = vd.conflitos_de_corte(limpar(df_bruto))
    assert resumo["Chaves com mais de um corte"] == 1
    assert resumo["Chaves que misturam corte preenchido e ausente"] == 1


def test_cobertura_ignora_ano_de_historico(df_bruto):
    df = limpar(df_bruto)
    entradas = montar_entrada_treino(df, construir_historico(df))
    t = vd.cobertura_detalhada(df.assign(CORTE_ANO_ANTERIOR=entradas["CORTE_ANO_ANTERIOR"]))
    assert t["ANO"].tolist() == [2018, 2019]
    assert t["cobertura_%"].tolist() == [100.0, 100.0]


def test_linhas_diferentes_trata_ausentes_como_iguais():
    a = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": ["a", "b", "c"]}, index=[7, 8, 9])
    b = a.reset_index(drop=True)
    assert vd.linhas_diferentes(a, b) == 0
    assert vd.linhas_diferentes(a, b.assign(x=[1.0, 2.0, 4.0])) == 2
    with pytest.raises(ValueError, match="colunas ou tamanhos"):
        vd.linhas_diferentes(a, b[["y", "x"]])


def test_treino_x_inferencia_aponta_historico_diferente(df_bruto):
    df = limpar(df_bruto)
    historico = construir_historico(df)
    igual = vd.consistencia_treino_inferencia(df, historico, tamanho_amostra=5)
    assert igual["treino_x_inferencia"].tolist() == [0, 0, 0, 0]
    assert set(igual["tipos_e_indice_iguais"]) == {"sim"}

    alterado = historico.assign(CORTE_ANO_ANTERIOR=historico["CORTE_ANO_ANTERIOR"] + 1)
    diferente = vd.consistencia_treino_inferencia(df, alterado, tamanho_amostra=5)
    assert diferente["treino_x_inferencia"].iloc[:3].tolist() == [0, 3, 3]  # 2017 segue sem corte


def test_consulta_direta_e_uso_do_ano_anterior(df_bruto):
    df = limpar(df_bruto)
    direta = vd.corte_por_consulta_direta(df)
    assert direta[df["ANO"] == 2017].isna().all()
    assert direta[df["ANO"] == 2019].tolist() == [610.0] * 3
    t = vd.uso_do_ano_anterior(df).drop(columns=["ANO", "linhas"])
    assert (t == 0).all().all()


def test_sem_historico_separa_motivos(df_bruto):
    df_bruto.loc[df_bruto["ANO"] == 2018, "NOTA_CORTE"] = np.nan  # chave existe, sem valor
    t = vd.sem_historico(limpar(df_bruto)).set_index("ANO")
    assert t.loc[2017, ["sem_corte", "chave_inexistente_em_n_1"]].tolist() == [3, 3]
    assert t.loc[2019, ["sem_corte", "chave_sem_valor_em_n_1"]].tolist() == [3, 3]
    assert t.loc[2018, "sem_corte"] == 0
    assert t["corte_e_dif_incoerentes"].sum() == 0


def test_cenarios_sem_historico(df_bruto):
    df = limpar(df_bruto)
    t = vd.cenarios_sem_historico(df, construir_historico(df))
    assert t["com_corte"].tolist() == [3, 0, 0]


def test_colunas_proibidas_nao_chegam_as_entradas(df_bruto):
    df = limpar(df_bruto)
    t = vd.atributos_nas_entradas(df, montar_entrada_treino(df, construir_historico(df)))
    assert set(t["nas_entradas"]) == {"não"}
    assert t.set_index("coluna_proibida").loc["NOTA_CORTE", "no_parquet"] == "sim"
    with pytest.raises(ValueError):
        vd.atributos_nas_entradas(df, df)


def test_main_grava_relatorio_sem_alterar_historico(df_bruto, tmp_path, monkeypatch):
    df = limpar(df_bruto)
    parquet, historico = tmp_path / "limpo.parquet", tmp_path / "historico.parquet"
    df.to_parquet(parquet, index=False)
    construir_historico(df).to_parquet(historico, index=False)
    antes = historico.read_bytes()
    monkeypatch.setattr(config, "PARQUET_LIMPO", parquet)
    monkeypatch.setattr(config, "HISTORICO_CORTES", historico)
    monkeypatch.setattr(vd, "DIR_RELATORIOS", tmp_path / "reports")

    vd.main()

    relatorio = (tmp_path / "reports" / "verificacao_dados.md").read_text(encoding="utf-8")
    assert "Histórico versionado idêntico ao reconstruído: sim" in relatorio
    assert historico.read_bytes() == antes
