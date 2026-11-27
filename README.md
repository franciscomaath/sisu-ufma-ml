# sisu-ufma-ml

Estimativa da **chance de um candidato ter nota suficiente para ingressar em um curso da UFMA pelo
SISU**, a partir das notas do ENEM. Trabalho final da disciplina de Inteligência Artificial
(UFMA, 2026.2, Profa. Dra. Alana Oliveira).

## O problema

O candidato ao SISU escolhe o curso **antes** de saber a nota de corte daquele ano. Este projeto
aprende, com os resultados de anos anteriores, como o corte de cada curso e modalidade se comporta,
e devolve uma **probabilidade** de a nota do candidato ser suficiente.

- **Tarefa:** classificação binária.
- **Unidade de previsão:** uma inscrição (candidato × curso × modalidade).
- **Alvo:** `NOTA_SUFICIENTE = NOTA_CANDIDATO >= NOTA_CORTE` (ver `docs/DECISOES.md`, D1).
- **Atributo-chave:** o **corte do ano anterior** do mesmo curso, modalidade e edição, que já é
  público no momento da inscrição.
- **Validação temporal:** treino 2018–2021, validação 2022, teste 2023. Nunca aleatória.
- **Sem vazamento:** `NOTA_CORTE`, `CLASSIFICACAO` e `MATRICULA` só existem depois do resultado e
  não entram no modelo.

## Dados

- **Fonte:** microdados abertos do SISU, Ministério da Educação, Portal de Dados Abertos do MEC
  (https://dadosabertos.mec.gov.br/sisu). Licença: **a confirmar**.
- **Recorte:** UFMA, 2017–2023, duas edições por ano. 1.080.713 inscrições, 31 colunas, 132 cursos, 11 campi.
- **Privacidade:** o conjunto não tem nome, CPF ou outro identificador pessoal.
- **Download do CSV consolidado** (~343 MB, não versionado):
  https://drive.google.com/file/d/1BCRwZ5n2NQ6Ej1_FD0seHs4HKo0Apbpj/view
  Salve como `data/raw/sisu_ufma_2017_2023.csv`.
- **Como o CSV foi consolidado a partir dos arquivos do MEC:** _a documentar._

## Instalação

Requer **Python 3.11 ou mais recente** (testado com 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como reproduzir

```bash
# 1. Colocar o CSV em data/raw/sisu_ufma_2017_2023.csv
python -m src.limpeza    # gera data/processed/sisu_limpo.parquet
python -m src.features   # gera models/historico_cortes.parquet e mostra a cobertura por ano
python -m src.baselines  # compara a regra e as regressões na validação de 2022
python -m src.verificar_dados  # confere a preparação; grava em reports/conciliacao/
pytest -q                # testes (os que usam o parquet real são pulados se ele não existir)
```

## Estado do projeto

| Etapa | Status |
|---|---|
| Limpeza, histórico de cortes, divisão temporal e testes | ✅ Pronta (`reports/relatorio_etapa1.md`) |
| Alvo `NOTA_SUFICIENTE` (D1) | ✅ Implementado |
| Lista explícita de atributos (D2) | ✅ Implementada |
| Nota do candidato (D12) e montagem compartilhada da entrada (D13) | ✅ Verificadas nos dados de 2017–2023 (`reports/relatorio_conciliacao.md`) |
| Regra do corte anterior e regressão logística | ✅ Comparadas na validação de 2022 |
| LightGBM, MLP, calibração e avaliação final | ⏳ A fazer |
| Aplicação Streamlit | ⏳ A fazer |
| Artigo SBC e vídeo | ⏳ A fazer |

## Estrutura

```
data/raw/            CSV bruto (não versionado)
data/processed/      parquet limpo (não versionado)
docs/                enunciado e decisões do projeto
notebooks/           exploração e figuras
src/                 limpeza, atributos, divisão, treino
app/                 aplicação Streamlit
models/              histórico de cortes e modelo salvo
reports/             relatórios, tabelas de resultados e figuras do artigo
tests/               testes automatizados
```

## Documentação

- [`docs/ENUNCIADO.md`](docs/ENUNCIADO.md): o que o trabalho exige e como é avaliado.
- [`docs/DECISOES.md`](docs/DECISOES.md): decisões de modelagem e seus motivos.
- [`reports/relatorio_etapa1.md`](reports/relatorio_etapa1.md): limpeza e investigação dos dados.
- [`CLAUDE.md`](CLAUDE.md): regras do projeto para quem usa o Claude Code.

## Resultados

Na validação de 2022, a regressão logística sem pesos teve a maior PR-AUC (0,6991). A versão com pesos balanceados teve o maior F1 (0,6016) com limiar de 0,5. [Relatório dos baselines](reports/baselines_2022.md)

## Aplicação

_A preencher: como executar e link público._

## Vídeo

_A preencher._

## Limitações

Esta ferramenta estima uma probabilidade com base em anos anteriores. **Não garante aprovação**:
o corte de cada ano depende da concorrência, das vagas e das regras daquela edição.

## Uso de IA generativa

Ferramentas de IA generativa (Claude) foram usadas como apoio no planejamento, na escrita de código
e na revisão. A declaração completa está no artigo.
