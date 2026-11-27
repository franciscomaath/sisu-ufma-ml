import numpy as np
import pandas as pd
import pytest

from src import config
from src import modelos
from src.features import construir_historico, montar_entrada_treino


def test_grade_registrada_na_d5():
    assert [len(modelos.GRADE[f]) for f in ("logistica", "lightgbm", "mlp")] == [6, 8, 8]
    assert {p["C"] for p in modelos.GRADE["logistica"]} == {0.1, 1.0, 10.0}
    assert {p["peso"] for grade in modelos.GRADE.values() for p in grade} == {None, modelos.PESO_BALANCEADO}


def test_pesos_balanceados_iguais_ao_class_weight():
    y = pd.Series([0, 0, 0, 1])
    assert modelos.pesos_balanceados(y).tolist() == pytest.approx([4 / 6] * 3 + [2.0])
    assert modelos.razao_negativos_positivos(y) == 3.0
    with pytest.raises(ValueError):
        modelos.pesos_balanceados(pd.Series([0, 0]))


def test_parametros_chegam_ao_classificador():
    y = pd.Series([0] * 9 + [1])
    lgbm = modelos.criar_classificador("lightgbm", {"num_leaves": 15, "min_child_samples": 20,
                                                    "peso": modelos.PESO_BALANCEADO}, y)
    assert lgbm.scale_pos_weight == 9.0 and lgbm.random_state == config.SEMENTE
    log = modelos.criar_classificador("logistica", {"C": 10.0, "peso": None}, y)
    assert (log.C, log.class_weight, log.max_iter) == (10.0, None, 1000)
    with pytest.raises(ValueError):
        modelos.criar_classificador("svm", {}, y)
    with pytest.raises(ValueError):
        modelos.criar_classificador("logistica", {"C": 1.0, "peso": "outro"}, y)


def test_peso_da_mlp_vai_como_sample_weight():
    y = pd.Series([0, 0, 1])
    balanceada = {"hidden_layer_sizes": (4,), "alpha": 1e-4, "peso": modelos.PESO_BALANCEADO}
    assert set(modelos.argumentos_de_ajuste("mlp", balanceada, y)) == {"classificador__sample_weight"}
    assert modelos.argumentos_de_ajuste("mlp", {**balanceada, "peso": None}, y) == {}
    assert modelos.argumentos_de_ajuste("logistica", {"C": 1.0, "peso": modelos.PESO_BALANCEADO}, y) == {}


@pytest.mark.parametrize("familia", ["logistica", "lightgbm", "mlp"])
def test_cada_familia_ajusta_e_preve(familia, df_limpo_sintetico):
    df = df_limpo_sintetico.dropna(subset=[config.ALVO])
    x = montar_entrada_treino(df, construir_historico(df))
    y = df[config.ALVO].astype(int)
    parametros = modelos.GRADE[familia][-1]
    modelo = modelos.criar_modelo(familia, parametros, y)
    modelo.fit(x, y, **modelos.argumentos_de_ajuste(familia, parametros, y))
    p = modelo.predict_proba(x)[:, 1]
    assert p.shape == (len(x),) and np.all((p >= 0) & (p <= 1))
