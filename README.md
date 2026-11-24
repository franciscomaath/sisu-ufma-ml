# sisu-ufma-ml

Previsão da **probabilidade de aprovação no SISU da UFMA** a partir das notas do ENEM
(classificação binária, alvo `APROVADO`). Trabalho final de Inteligência Artificial (UFMA, 2026.2).

## Problema
Cada linha do conjunto é uma inscrição (candidato × curso). O objetivo é estimar a chance de aprovação
usando apenas informação disponível **antes** do resultado. Atributos como `NOTA_CORTE`, `CLASSIFICACAO`
e `MATRICULA` são vazamento e não entram no modelo. O atributo-chave é o **corte do ano anterior**.
A validação é temporal (treino 2018–2021, validação 2022, teste 2023), nunca aleatória.

## Fonte dos dados
Microdados abertos do SISU/MEC, recorte UFMA 2017–2023 (2 edições/ano):
Portal de Dados Abertos do MEC — https://dadosabertos.mec.gov.br/sisu
Licença: **a confirmar**.

## Instalação
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dados
Coloque o CSV em `data/raw/sisu_ufma_2017_2023.csv` (não versionado, ~343 MB).

```bash
python -m src.limpeza    # gera data/processed/sisu_limpo.parquet
python -m src.features   # gera models/historico_cortes.parquet|joblib e mostra a cobertura
pytest -q                # testes
```

## Estrutura
`data/` dados (não versionados) · `notebooks/` EDA · `src/` limpeza, atributos, divisão ·
`app/` Streamlit · `models/` artefatos · `reports/figures/` figuras do artigo · `tests/`.

## Resultados
_A preencher._

## Vídeo
_A preencher._
