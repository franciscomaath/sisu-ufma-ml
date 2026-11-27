"""Experimentos da validação (D4, D5, D8): busca em A, calibração em B, escolhas em C.

Não lê nem avalia 2023. Uso:
    python -m src.experimentos --etapa ensaio      # custo da MLP
    python -m src.experimentos --etapa busca       # grade da D5 em A
    python -m src.experimentos --etapa calibracao  # B, C, limiar e modelo final
    python -m src.experimentos                     # as três, em ordem
"""
import argparse
import json
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone

import joblib
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from src import config
from src import modelos
from src.baselines import escore_regra_corte_anterior, prever_regra_corte_anterior
from src.features import COLUNAS_ENTRADA, construir_historico, montar_entrada_treino
from src.particoes import ARQUIVO_MAPA, DIR_EXPERIMENTOS, aplicar_mapa, particionar, rotuladas_validacao, sha256
from src.particoes import mapa_grupos, verificar_partes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANOS_LIDOS = config.ANOS_HISTORICO + config.ANOS_TREINO + config.ANOS_VALIDACAO
MODELO_SELECIONADO = config.DIR_MODELOS / "modelo_selecionado_A.joblib"
PIPELINE_FINAL = config.DIR_MODELOS / "pipeline_final.joblib"
METADADOS = config.DIR_MODELOS / "metadados.json"
ARQUIVO_BUSCA = DIR_EXPERIMENTOS / "busca_A.csv"
ARQUIVO_ENSAIO = DIR_EXPERIMENTOS / "ensaio_mlp.json"
ARQUIVO_SELECAO = DIR_EXPERIMENTOS / "selecao_A.json"
FIGURA_CALIBRACAO = config.DIR_FIGURAS / "calibracao_C.png"
CORES = {"Sem recalibração": "#257de0", "Sigmoid": "#f47230", "Isotônica": "#00b894"}
MARCADORES = {"Sem recalibração": "o", "Sigmoid": "s", "Isotônica": "^"}


# ----------------------------------------------------------------------------- dados

def carregar_dados() -> dict:
    """Entradas de treino (2018–2021) e de A, B e C (2022), com o histórico de 2017–2021."""
    df = pd.read_parquet(config.PARQUET_LIMPO, filters=[("ANO", "in", ANOS_LIDOS)])
    if set(df["ANO"]) & set(config.ANOS_TESTE):
        raise RuntimeError("O teste não pode ser lido nesta etapa.")
    historico = construir_historico(df[df["ANO"].isin(config.ANOS_HISTORICO + config.ANOS_TREINO)])
    treino = df[df["ANO"].isin(config.ANOS_TREINO)].dropna(subset=[config.ALVO])
    validacao = rotuladas_validacao(df)

    mapa = pd.read_csv(ARQUIVO_MAPA)
    parte = aplicar_mapa(validacao, mapa)
    refeito = mapa_grupos(validacao, particionar(validacao))
    if not refeito.equals(mapa[refeito.columns].astype(refeito.dtypes.to_dict())):
        raise RuntimeError("A divisão refeita difere do mapa salvo em reports/experimentos.")
    verificar_partes(validacao, parte)

    dados = {"treino": (montar_entrada_treino(treino, historico), treino[config.ALVO].astype("int8"))}
    x_validacao = montar_entrada_treino(validacao, historico)
    for nome in "ABC":
        linhas = parte.index[parte == nome]
        dados[nome] = (x_validacao.loc[linhas], validacao.loc[linhas, config.ALVO].astype("int8"))
    return dados


def subamostra_estratificada(x: pd.DataFrame, y: pd.Series, tamanho: int):
    """Subamostra do treino estratificada pelo alvo, semente 42 (D5)."""
    if tamanho >= len(y):
        return x, y
    divisor = StratifiedShuffleSplit(n_splits=1, train_size=tamanho, random_state=config.SEMENTE)
    posicoes, _ = next(divisor.split(x, y))
    posicoes = np.sort(posicoes)
    return x.iloc[posicoes], y.iloc[posicoes]


# ----------------------------------------------------------------------------- métricas

def metricas(y, escore, previsao=None, limiar: float = 0.5) -> dict:
    """AP e ROC-AUC do escore; precisão, revocação e F1 da previsão (ou de escore >= limiar)."""
    y = np.asarray(y).astype(int)
    escore = np.asarray(escore, dtype=float)
    previsao = (escore >= limiar).astype(int) if previsao is None else np.asarray(previsao).astype(int)
    return {
        "ap": average_precision_score(y, escore),
        "roc_auc": roc_auc_score(y, escore),
        "precisao": precision_score(y, previsao, zero_division=0),
        "revocacao": recall_score(y, previsao, zero_division=0),
        "f1": f1_score(y, previsao, zero_division=0),
    }


def metricas_probabilidade(y, p) -> dict:
    """Métricas de uma probabilidade, inclusive as de calibração."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    return {
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "ap": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "media_prevista": float(p.mean()),
        "taxa_observada": float(y.mean()),
    }


def avaliar_regra(x: pd.DataFrame, y: pd.Series) -> dict:
    """Regra histórica: decisão binária (>=) e ranking pelo escore contínuo (D5)."""
    previsao = prever_regra_corte_anterior(x)
    resultado = metricas(y, escore_regra_corte_anterior(x), previsao)
    resultado["ap_da_saida_binaria"] = average_precision_score(y, previsao)
    resultado["sem_historico"] = int(x["CORTE_ANO_ANTERIOR"].isna().sum())
    return resultado


def melhor_limiar_f1(y, p) -> dict:
    """Limiar que maximiza o F1 (em empate, o menor limiar)."""
    precisao, revocacao, limiares = precision_recall_curve(np.asarray(y).astype(int), p)
    precisao, revocacao = precisao[:-1], revocacao[:-1]
    soma = precisao + revocacao
    f1 = np.divide(2 * precisao * revocacao, soma, out=np.zeros_like(soma), where=soma > 0)
    i = int(np.argmax(f1))
    return {"limiar": float(limiares[i]), "f1": float(f1[i]),
            "precisao": float(precisao[i]), "revocacao": float(revocacao[i])}


# ----------------------------------------------------------------------------- ajuste

def ajustar(familia: str, parametros: dict, x: pd.DataFrame, y: pd.Series):
    """Ajusta o Pipeline completo. Retorna (modelo, segundos, avisos)."""
    modelo = modelos.criar_modelo(familia, parametros, y)
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        inicio = time.perf_counter()
        modelo.fit(x, y, **modelos.argumentos_de_ajuste(familia, parametros, y))
        segundos = time.perf_counter() - inicio
    avisos = sorted({f"{aviso.category.__name__}: {str(aviso.message)[:120]}" for aviso in capturados})
    return modelo, segundos, avisos


def detalhes_do_ajuste(modelo) -> str:
    classificador = modelo.named_steps["classificador"]
    if hasattr(classificador, "n_iter_"):
        iteracoes = np.ravel(classificador.n_iter_)
        return f"iterações={int(iteracoes.max())}"
    return ""


# ----------------------------------------------------------------------------- etapas

def etapa_ensaio(dados: dict) -> dict:
    """Ensaio de custo da MLP e decisão sobre a subamostra (regra fixada na D5)."""
    x, y = dados["treino"]
    x_ensaio, y_ensaio = subamostra_estratificada(x, y, modelos.TAMANHO_ENSAIO_MLP)
    _, segundos, avisos = ajustar("mlp", modelos.CONFIG_ENSAIO_MLP, x_ensaio, y_ensaio)
    configuracoes = len(modelos.GRADE["mlp"])
    projecao = segundos * len(y) / len(y_ensaio) * configuracoes / 60
    usar_subamostra = projecao > modelos.LIMITE_MINUTOS_MLP
    resultado = {
        "configuracao": modelos.descrever(modelos.CONFIG_ENSAIO_MLP),
        "linhas_ensaio": len(y_ensaio), "positivos_ensaio": int(y_ensaio.sum()),
        "segundos_ensaio": round(segundos, 1), "linhas_treino": len(y),
        "configuracoes_na_grade": configuracoes,
        "minutos_projetados_grade_completa": round(projecao, 1),
        "limite_minutos": modelos.LIMITE_MINUTOS_MLP,
        "usar_subamostra": bool(usar_subamostra),
        "linhas_por_configuracao": modelos.TAMANHO_SUBAMOSTRA_MLP if usar_subamostra else len(y),
        "avisos": avisos,
    }
    DIR_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
    ARQUIVO_ENSAIO.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return resultado


def etapa_busca(dados: dict) -> pd.DataFrame:
    """Grade da D5 nas mesmas linhas de A. Salva a tabela e o melhor modelo aprendido."""
    if not ARQUIVO_ENSAIO.is_file():
        raise SystemExit("Execute antes a etapa de ensaio.")
    ensaio = json.loads(ARQUIVO_ENSAIO.read_text(encoding="utf-8"))
    x_treino, y_treino = dados["treino"]
    x_a, y_a = dados["A"]
    linhas = []

    def registrar(linha: dict) -> None:
        linhas.append(linha)
        pd.DataFrame(linhas).to_csv(ARQUIVO_BUSCA, index=False, lineterminator="\n")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in linha.items()}, flush=True)

    regra = avaliar_regra(x_a, y_a)
    registrar({"familia": "regra", "configuracao": "NOTA_CANDIDATO >= CORTE_ANO_ANTERIOR",
               "linhas_treino": 0, **regra, "brier": np.nan, "segundos_ajuste": 0.0,
               "segundos_predicao": 0.0, "detalhes": "escore: DIF_CORTE_ANTERIOR", "avisos": ""})

    melhor = {"ap": -1.0, "roc_auc": -1.0}
    x_mlp, y_mlp = subamostra_estratificada(x_treino, y_treino, ensaio["linhas_por_configuracao"])
    for familia, grade in modelos.GRADE.items():
        x, y = (x_mlp, y_mlp) if familia == "mlp" else (x_treino, y_treino)
        for parametros in grade:
            modelo, segundos, avisos = ajustar(familia, parametros, x, y)
            inicio = time.perf_counter()
            p = modelo.predict_proba(x_a)[:, 1]
            predicao = time.perf_counter() - inicio
            resultado = metricas(y_a, p)
            registrar({"familia": familia, "configuracao": modelos.descrever(parametros),
                       "linhas_treino": len(y), **resultado, "ap_da_saida_binaria": np.nan,
                       "sem_historico": int(x_a["CORTE_ANO_ANTERIOR"].isna().sum()),
                       "brier": brier_score_loss(y_a, p), "segundos_ajuste": round(segundos, 1),
                       "segundos_predicao": round(predicao, 2), "detalhes": detalhes_do_ajuste(modelo),
                       "avisos": " | ".join(avisos)})
            chave = (round(resultado["ap"], 4), resultado["roc_auc"])
            if chave > (round(melhor["ap"], 4), melhor["roc_auc"]):
                melhor = {**resultado, "familia": familia, "parametros": parametros}
                joblib.dump(modelo, MODELO_SELECIONADO)

    selecao = {"familia": melhor["familia"], "parametros": melhor["parametros"],
               "configuracao": modelos.descrever(melhor["parametros"]),
               "ap_A": melhor["ap"], "roc_auc_A": melhor["roc_auc"], "ap_regra_A": regra["ap"],
               "criterio": "maior AP em A; empate na 4ª casa decidido pela ROC-AUC"}
    ARQUIVO_SELECAO.write_text(json.dumps(selecao, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(selecao, ensure_ascii=False, indent=2, default=str))
    return pd.DataFrame(linhas)


def ajustar_calibradores(modelo, x_b: pd.DataFrame, y_b: pd.Series) -> dict:
    """Sigmoid e isotônica ajustadas só em B, com o modelo congelado (D8)."""
    return {nome: CalibratedClassifierCV(FrozenEstimator(modelo), method=metodo).fit(x_b, y_b)
            for nome, metodo in [("Sigmoid", "sigmoid"), ("Isotônica", "isotonic")]}


def pontos_de_calibracao(y, p, n_faixas: int = 10) -> pd.DataFrame:
    """Faixas de largura fixa em [0, 1]: inscrições, média prevista e fração observada (faixas vazias omitidas)."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    faixa = np.minimum((p * n_faixas).astype(int), n_faixas - 1)
    t = pd.DataFrame({"faixa": faixa, "p": p, "y": y}).groupby("faixa").agg(
        inscricoes=("y", "size"), media_prevista=("p", "mean"), fracao_observada=("y", "mean"))
    t.insert(0, "inicio", t.index / n_faixas)
    return t.reset_index(drop=True)


def figura_calibracao(y_c, probabilidades: dict, caminho) -> pd.DataFrame:
    """Curva de calibração em C, com 10 faixas de largura 0,1. Retorna os pontos e as contagens."""
    pontos = []
    fig, eixo = plt.subplots(figsize=(6.4, 5.2))
    for nome, p in probabilidades.items():
        t = pontos_de_calibracao(y_c, p)
        brier = brier_score_loss(np.asarray(y_c).astype(int), p)
        eixo.plot(t["media_prevista"], t["fracao_observada"], marker=MARCADORES[nome], markersize=8,
                  linewidth=2, color=CORES[nome], label=f"{nome} (Brier {brier:.4f})".replace(".", ","))
        pontos.append(t.assign(opcao=nome))
    eixo.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#666666", label="Calibração perfeita")
    eixo.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidade média prevista na faixa (faixas de 0,1)",
             ylabel="Fração observada de nota suficiente",
             title="Curva de calibração na parte C da validação (2022)")
    eixo.grid(color="#e6e6e6", linewidth=0.8)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    eixo.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=300)
    plt.close(fig)
    return pd.concat(pontos, ignore_index=True)[["opcao", "inicio", "inscricoes", "media_prevista", "fracao_observada"]]


def caminho_relativo(caminho) -> str:
    """Caminho relativo à raiz do projeto, com '/'; fora dela, só o nome do arquivo."""
    try:
        return caminho.relative_to(config.RAIZ).as_posix()
    except ValueError:
        return caminho.name


def versao_do_codigo() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=config.RAIZ).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "alteracoes_nao_commitadas": bool(git("status", "--porcelain", "--", "src"))}


def etapa_calibracao(dados: dict) -> dict:
    """Congela o modelo de A, calibra em B, escolhe calibração e limiar em C e salva o objeto final."""
    selecao = json.loads(ARQUIVO_SELECAO.read_text(encoding="utf-8"))
    modelo = joblib.load(MODELO_SELECIONADO)
    x_a, y_a = dados["A"]
    ap_recarregado = average_precision_score(y_a, modelo.predict_proba(x_a)[:, 1])
    if not np.isclose(ap_recarregado, selecao["ap_A"], rtol=0, atol=1e-12):
        raise RuntimeError("O modelo salvo não reproduz a AP registrada em A.")

    x_b, y_b = dados["B"]
    x_c, y_c = dados["C"]
    opcoes = {"Sem recalibração": modelo, **ajustar_calibradores(modelo, x_b, y_b)}
    probabilidades = {nome: objeto.predict_proba(x_c)[:, 1] for nome, objeto in opcoes.items()}
    comparacao = pd.DataFrame([{"opcao": nome, **metricas_probabilidade(y_c, p)}
                               for nome, p in probabilidades.items()])
    escolhida = comparacao.loc[comparacao["brier"].idxmin(), "opcao"]
    pontos = figura_calibracao(y_c, probabilidades, FIGURA_CALIBRACAO)
    limiar = melhor_limiar_f1(y_c, probabilidades[escolhida])
    previsao = (probabilidades[escolhida] >= limiar["limiar"]).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_c, previsao, labels=[0, 1]).ravel()
    limiar.update({"opcao": escolhida, "vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)})

    final = opcoes[escolhida]
    joblib.dump(final, PIPELINE_FINAL)
    recarregado = joblib.load(PIPELINE_FINAL)
    identicas = bool(np.array_equal(recarregado.predict_proba(x_c), final.predict_proba(x_c)))
    if not identicas:
        raise RuntimeError("Salvar e recarregar alterou as probabilidades.")

    comparacao.to_csv(DIR_EXPERIMENTOS / "calibracao_C.csv", index=False, lineterminator="\n")
    pontos.to_csv(DIR_EXPERIMENTOS / "curva_calibracao_C.csv", index=False, lineterminator="\n")
    pd.DataFrame([limiar]).to_csv(DIR_EXPERIMENTOS / "limiar_C.csv", index=False, lineterminator="\n")

    historico_app = pd.read_parquet(config.HISTORICO_CORTES)
    tamanhos = {nome: {"linhas": len(y), "positivos": int(y.sum())} for nome, (_, y) in dados.items()}
    metadados = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "codigo": versao_do_codigo(),
        "versoes": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "lightgbm": lightgbm.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                    "joblib": joblib.__version__},
        "semente": config.SEMENTE,
        "arquivo_modelo": PIPELINE_FINAL.name,
        "uso": "predict_proba(montar_entrada_inferencia(entrada, historico, ano_alvo))[:, 1]",
        "atributos": config.ATRIBUTOS,
        "atributos_numericos": config.ATRIBUTOS_NUMERICOS,
        "atributos_categoricos": config.ATRIBUTOS_CATEGORICOS,
        "colunas_entrada": COLUNAS_ENTRADA,
        "modelo": {"familia": selecao["familia"], "parametros": selecao["parametros"],
                   "anos_treino": config.ANOS_TREINO,
                   "historico_de_cortes_no_treino": config.ANOS_HISTORICO + config.ANOS_TREINO},
        "calibracao": {"opcao": escolhida, "ajustada_em": "B (partição 3 de 2022)",
                       "escolhida_em": "C (partição 4 de 2022), pelo menor Brier"},
        "limiar": limiar["limiar"],
        "limiar_criterio": "maior F1 em C",
        "divisao_validacao": {"arquivo": caminho_relativo(ARQUIVO_MAPA),
                              "sha256": sha256(ARQUIVO_MAPA), "tamanhos": tamanhos},
        "metricas_validacao": {
            "A_selecao": {"ap": selecao["ap_A"], "roc_auc": selecao["roc_auc_A"]},
            "C_escolha": comparacao.set_index("opcao").loc[escolhida].to_dict() | {"f1_no_limiar": limiar["f1"]},
            "aviso": "A e C foram usadas para escolher; não são estimativas de desempenho final.",
        },
        "metricas_teste": {"status": "pendente", "ano": config.ANOS_TESTE},
        "historico_aplicacao": {"arquivo": "historico_cortes.parquet",
                                "ultimo_ano": int(historico_app["ANO"].max()),
                                "ano_alvo_previsto": int(historico_app["ANO"].max()) + 1},
        "recarga_preserva_probabilidades": identicas,
    }
    METADADOS.write_text(json.dumps(metadados, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(comparacao.to_string(index=False))
    print(json.dumps(limiar, ensure_ascii=False, indent=2))
    return metadados


def main(argumentos=None) -> None:
    analisador = argparse.ArgumentParser()
    analisador.add_argument("--etapa", choices=["ensaio", "busca", "calibracao", "tudo"], default="tudo")
    etapa = analisador.parse_args(argumentos).etapa
    dados = carregar_dados()
    for nome, (x, y) in dados.items():
        print(f"{nome}: {len(y)} linhas, {int(y.sum())} positivos", flush=True)
    if etapa in ("ensaio", "tudo"):
        etapa_ensaio(dados)
    if etapa in ("busca", "tudo"):
        etapa_busca(dados)
    if etapa in ("calibracao", "tudo"):
        etapa_calibracao(dados)


if __name__ == "__main__":
    main()
