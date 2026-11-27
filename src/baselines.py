"""Compara a regra do corte anterior e a regressão logística na validação."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.features import adicionar_corte_anterior, colunas_atributos, construir_historico
from src.split import dividir_temporal


def prever_regra_corte_anterior(df: pd.DataFrame) -> pd.Series:
    """Prevê 1 se a nota superar o corte anterior. Empates e ausências valem 0."""
    return df["NOTA_CANDIDATO"].gt(df["CORTE_ANO_ANTERIOR"]).fillna(False).astype("int8")


def criar_regressao_logistica(peso_classe: str | None = None) -> Pipeline:
    """Monta o pré-processamento e a regressão para ajuste conjunto no treino."""
    numericos = Pipeline([
        ("imputacao", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("escala", StandardScaler()),
    ])
    preprocessamento = ColumnTransformer([
        ("numericos", numericos, config.ATRIBUTOS_NUMERICOS),
        ("categoricos", OneHotEncoder(handle_unknown="ignore"), config.ATRIBUTOS_CATEGORICOS),
    ])
    return Pipeline([
        ("preprocessamento", preprocessamento),
        ("regressao", LogisticRegression(
            class_weight=peso_classe, solver="liblinear", random_state=config.SEMENTE,
        )),
    ])


def medir_baseline(alvo, escores, previsoes) -> dict[str, float]:
    """Calcula AP a partir dos escores e as demais métricas a partir das classes."""
    return {
        "PR-AUC (AP)": average_precision_score(alvo, escores),
        "Precisão": precision_score(alvo, previsoes, zero_division=0),
        "Revocação": recall_score(alvo, previsoes, zero_division=0),
        "F1": f1_score(alvo, previsoes, zero_division=0),
    }


def main() -> None:
    if not config.PARQUET_LIMPO.is_file():
        raise SystemExit(f"Parquet ausente em {config.PARQUET_LIMPO}. Execute python -m src.limpeza.")

    anos_usados = config.ANOS_HISTORICO + config.ANOS_TREINO + config.ANOS_VALIDACAO
    df = pd.read_parquet(config.PARQUET_LIMPO, filters=[("ANO", "in", anos_usados)])
    if config.ALVO not in df.columns:
        raise SystemExit(f"Coluna {config.ALVO} ausente. Regenere o parquet com python -m src.limpeza.")

    anos_historico = config.ANOS_HISTORICO + config.ANOS_TREINO
    historico = construir_historico(df[df["ANO"].isin(anos_historico)])
    treino, validacao, _ = dividir_temporal(df)
    rotuladas = []
    for nome, conjunto in [("Treino", treino), ("Validação", validacao)]:
        inscricoes = conjunto.dropna(subset=[config.ALVO])
        if inscricoes[config.ALVO].nunique() != 2:
            raise SystemExit(f"{nome} precisa conter exemplos das duas classes em {config.ALVO}.")
        print(f"{nome}: {len(inscricoes)} linhas usadas, {len(conjunto) - len(inscricoes)} sem alvo.")
        rotuladas.append(adicionar_corte_anterior(inscricoes, historico))

    treino, validacao = rotuladas
    x_treino = treino[colunas_atributos(treino)]
    x_validacao = validacao[colunas_atributos(validacao)]
    y_treino = treino[config.ALVO].astype("int8")
    y_validacao = validacao[config.ALVO].astype("int8")

    previsoes_regra = prever_regra_corte_anterior(x_validacao)
    resultados = {
        "Regra do corte anterior": medir_baseline(y_validacao, previsoes_regra, previsoes_regra),
    }
    for nome, peso in [("Logística sem pesos", None), ("Logística com pesos balanceados", "balanced")]:
        modelo = criar_regressao_logistica(peso)
        modelo.fit(x_treino, y_treino)
        probabilidades = modelo.predict_proba(x_validacao)[:, 1]
        resultados[nome] = medir_baseline(y_validacao, probabilidades, probabilidades >= 0.5)

    anos_validacao = ", ".join(map(str, config.ANOS_VALIDACAO))
    print(f"\nBaselines na validação ({anos_validacao})")
    print(pd.DataFrame.from_dict(resultados, orient="index").to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
