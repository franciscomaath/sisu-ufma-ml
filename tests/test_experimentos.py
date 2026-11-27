import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src import experimentos as ex
from src import modelos
from src import particoes as pt
from src.baselines import ESCORE_SEM_HISTORICO


def test_metricas_da_regra_separam_decisao_e_ranking():
    x = pd.DataFrame({"NOTA_CANDIDATO": [700.0, 600.0, 650.0, 500.0],
                      "CORTE_ANO_ANTERIOR": [650.0, 650.0, 650.0, np.nan],
                      "DIF_CORTE_ANTERIOR": [50.0, -50.0, 0.0, np.nan]})
    y = pd.Series([1, 0, 1, 1])
    r = ex.avaliar_regra(x, y)
    assert (r["precisao"], r["revocacao"]) == (1.0, pytest.approx(2 / 3))  # empate conta como 1
    assert r["sem_historico"] == 1
    assert r["roc_auc"] == pytest.approx(2 / 3)  # o sem histórico (positivo) fica no fim
    assert ESCORE_SEM_HISTORICO < -1000


def test_melhor_limiar_f1():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.9])
    resultado = ex.melhor_limiar_f1(y, p)
    assert resultado["limiar"] == 0.3 and resultado["f1"] == pytest.approx(6 / 7)


def test_metricas_probabilidade():
    m = ex.metricas_probabilidade([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0])
    assert m["brier"] == 0 and m["ap"] == 1 and m["taxa_observada"] == 0.5


def test_subamostra_estratificada_mantem_taxa_e_e_reprodutivel():
    x = pd.DataFrame({"a": np.arange(1000)})
    y = pd.Series([1] * 100 + [0] * 900)
    xs, ys = ex.subamostra_estratificada(x, y, 200)
    assert len(ys) == 200 and ys.sum() == 20
    assert xs.index.equals(ex.subamostra_estratificada(x, y, 200)[0].index)
    assert ex.subamostra_estratificada(x, y, 5000)[0] is x


def test_calibradores_nao_alteram_o_modelo(df_limpo_sintetico):
    from src.features import construir_historico, montar_entrada_treino
    df = df_limpo_sintetico.dropna(subset=[config.ALVO])
    x = montar_entrada_treino(df, construir_historico(df))
    y = df[config.ALVO].astype(int)
    treino, b = (df["ANO"] < 2021).to_numpy(), (df["ANO"] >= 2021).to_numpy()
    modelo, _, _ = ex.ajustar("logistica", {"C": 1.0, "peso": None}, x[treino], y[treino])
    antes = modelo.predict_proba(x[b])
    calibradores = ex.ajustar_calibradores(modelo, x[b], y[b])
    assert np.array_equal(modelo.predict_proba(x[b]), antes)
    for calibrado in calibradores.values():
        assert len(calibrado.calibrated_classifiers_) == 1


@pytest.fixture
def ambiente(df_limpo_sintetico, tmp_path, monkeypatch):
    """Redireciona todos os arquivos para tmp_path e reduz a grade."""
    parquet = tmp_path / "limpo.parquet"
    df_limpo_sintetico.to_parquet(parquet, index=False)
    historico = tmp_path / "historico.parquet"
    from src.features import construir_historico
    construir_historico(df_limpo_sintetico).to_parquet(historico, index=False)
    relatorios, modelos_dir = tmp_path / "reports", tmp_path / "models"
    modelos_dir.mkdir()
    monkeypatch.setattr(config, "PARQUET_LIMPO", parquet)
    monkeypatch.setattr(config, "HISTORICO_CORTES", historico)
    monkeypatch.setattr(pt, "DIR_EXPERIMENTOS", relatorios)
    monkeypatch.setattr(pt, "ARQUIVO_MAPA", relatorios / "mapa.csv")
    for nome, caminho in {
        "DIR_EXPERIMENTOS": relatorios, "ARQUIVO_MAPA": relatorios / "mapa.csv",
        "ARQUIVO_BUSCA": relatorios / "busca.csv", "ARQUIVO_ENSAIO": relatorios / "ensaio.json",
        "ARQUIVO_SELECAO": relatorios / "selecao.json", "MODELO_SELECIONADO": modelos_dir / "sel.joblib",
        "PIPELINE_FINAL": modelos_dir / "final.joblib", "METADADOS": modelos_dir / "meta.json",
        "FIGURA_CALIBRACAO": tmp_path / "fig.png",
    }.items():
        monkeypatch.setattr(ex, nome, caminho)
    monkeypatch.setattr(modelos, "GRADE", {
        "logistica": [{"C": 1.0, "peso": None}],
        "lightgbm": [{"num_leaves": 15, "min_child_samples": 20, "peso": None}],
        "mlp": [{"hidden_layer_sizes": (8,), "alpha": 1e-4, "peso": modelos.PESO_BALANCEADO}],
    })
    monkeypatch.setattr(modelos, "TAMANHO_ENSAIO_MLP", 500)
    monkeypatch.setattr(modelos, "CONFIG_ENSAIO_MLP", {"hidden_layer_sizes": (8,), "alpha": 1e-4, "peso": None})
    return tmp_path


def test_etapas_de_ponta_a_ponta(ambiente, df_limpo_sintetico):
    pt.main()
    ex.main(["--etapa", "tudo"])
    busca = pd.read_csv(ex.ARQUIVO_BUSCA)
    assert busca["familia"].tolist() == ["regra", "logistica", "lightgbm", "mlp"]
    meta = json.loads(ex.METADADOS.read_text(encoding="utf-8"))
    assert meta["metricas_teste"]["status"] == "pendente"
    assert meta["recarga_preserva_probabilidades"] is True
    assert meta["atributos"] == config.ATRIBUTOS
    tamanhos = meta["divisao_validacao"]["tamanhos"]
    rotuladas_2022 = df_limpo_sintetico.query("ANO == 2022")[config.ALVO].notna().sum()
    assert sum(tamanhos[p]["linhas"] for p in "ABC") == rotuladas_2022
    final = joblib.load(ex.PIPELINE_FINAL)
    assert hasattr(final, "predict_proba")
    assert ex.FIGURA_CALIBRACAO.is_file()


def test_divisao_diferente_do_mapa_e_erro(ambiente):
    pt.main()
    mapa = pd.read_csv(pt.ARQUIVO_MAPA)
    mapa["parte"] = mapa["parte"].map({"A": "B", "B": "C", "C": "A"})
    mapa.to_csv(pt.ARQUIVO_MAPA, index=False)
    with pytest.raises(RuntimeError, match="difere do mapa"):
        ex.carregar_dados()


def test_nao_le_o_teste(ambiente, monkeypatch):
    pt.main()
    lidos = []
    original = pd.read_parquet

    def espiao(*args, **kwargs):
        resultado = original(*args, **kwargs)
        lidos.append(set(resultado["ANO"]))
        return resultado

    monkeypatch.setattr(pd, "read_parquet", espiao)
    ex.carregar_dados()
    assert not set().union(*lidos) & set(config.ANOS_TESTE)


def test_pontos_de_calibracao_por_faixas_fixas():
    t = ex.pontos_de_calibracao([0, 1, 0, 1, 1], [0.05, 0.05, 0.55, 0.95, 1.0])
    assert t["inicio"].tolist() == [0.0, 0.5, 0.9]  # faixas vazias omitidas; 1,0 cai na última
    assert t["inscricoes"].tolist() == [2, 1, 2]
    assert t["fracao_observada"].tolist() == [0.5, 0.0, 1.0]
