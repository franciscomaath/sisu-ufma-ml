import pandas as pd
import pytest

from src import config
from src import particoes as pt


def _validacao(df):
    return pt.rotuladas_validacao(df)


def test_so_rotuladas_de_2022(df_limpo_sintetico):
    v = _validacao(df_limpo_sintetico)
    assert set(v["ANO"]) == set(config.ANOS_VALIDACAO)
    assert v[config.ALVO].notna().all()


def test_particoes_respeitam_os_grupos_e_sao_reprodutiveis(df_limpo_sintetico):
    v = _validacao(df_limpo_sintetico)
    particao = pt.particionar(v)
    assert set(particao) == {0, 1, 2, 3, 4}
    por_grupo = pd.DataFrame({"g": pt.chave_grupo(v), "p": particao}).groupby("g")["p"].nunique()
    assert (por_grupo == 1).all()
    assert particao.equals(pt.particionar(v))  # semente fixa


def test_mapa_salvo_reconstroi_as_partes(df_limpo_sintetico, tmp_path):
    v = _validacao(df_limpo_sintetico)
    mapa = pt.mapa_grupos(v, pt.particionar(v))
    assert set(mapa["parte"]) == {"A", "B", "C"}
    assert mapa["linhas"].sum() == len(v)
    mapa.to_csv(tmp_path / "mapa.csv", index=False)
    parte = pt.aplicar_mapa(v, pd.read_csv(tmp_path / "mapa.csv"))
    esperado = pt.particionar(v).map(pt.PARTES)
    assert parte.tolist() == esperado.tolist()
    resumo = pt.resumo_partes(v, parte)
    assert resumo["linhas"].sum() == len(v)
    assert (resumo["positivos"] + resumo["negativos"] == resumo["linhas"]).all()


def test_grupo_fora_do_mapa_e_erro(df_limpo_sintetico):
    v = _validacao(df_limpo_sintetico)
    mapa = pt.mapa_grupos(v, pt.particionar(v)).iloc[1:]
    with pytest.raises(ValueError, match="ausentes do mapa"):
        pt.aplicar_mapa(v, mapa)


def test_verificacao_detecta_sobreposicao_e_classe_ausente(df_limpo_sintetico):
    v = _validacao(df_limpo_sintetico)
    parte = pt.particionar(v).map(pt.PARTES)
    assert set(pt.verificar_partes(v, parte).values()) == {0}
    dividido = parte.copy()
    grupo = pt.chave_grupo(v)
    linhas = grupo.index[grupo == grupo.iloc[0]]
    dividido.loc[linhas[0]] = "B" if dividido.loc[linhas[1]] != "B" else "C"
    with pytest.raises(ValueError, match="grupos em mais de uma parte"):
        pt.verificar_partes(v, dividido)
    so_negativos = v.assign(**{config.ALVO: 0})
    with pytest.raises(ValueError, match="duas classes"):
        pt.verificar_partes(so_negativos, parte)


def test_particionar_recusa_linha_sem_alvo(df_limpo_sintetico):
    v = df_limpo_sintetico[df_limpo_sintetico["ANO"] == 2017]
    with pytest.raises(ValueError, match="rotuladas"):
        pt.particionar(v)
