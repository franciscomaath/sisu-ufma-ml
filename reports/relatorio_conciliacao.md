# Relatório da conciliação e da verificação da preparação

Data: 27/11/2026. Branch `integracao/preparacao-dados-reais`, a partir de `179593a`
(`origin/main`), publicada para revisão por pull request. Sem merge.

**Resumo.** O commit `68a3e06` foi conciliado com o código atual, a D12 passou a ter o
arredondamento que reproduz o CSV, e as entradas de treino e inferência coincidem nos dados reais.
As entradas dos baselines publicados não mudaram. Nenhum modelo foi treinado nos dados reais e
nenhuma métrica de desempenho de 2023 foi calculada.

Este relatório sucede `reports/relatorio_integracao.md`, que descreve o estado anterior (sem o
patch) e permanece como registro histórico.

## 1. Patch

`preparacao-68a3e06.patch`, SHA-256
`406b7d5b74e40dcc84bc8d88388173d8347e68c70bcb28710b934ddb1fd1d703`: confere com o esperado.
Feito sobre `9eec742`. Foi aplicado por partes; o arquivo não foi versionado.

| Parte do patch | O que foi feito |
|---|---|
| Listas da D2 em `config.py` | Nada a aplicar: já idênticas em `179593a` |
| `colunas_atributos()` sem argumento | Não adotada. Mantida `colunas_atributos(df)` |
| `selecionar_atributos(df)` | Adotada sobre `colunas_atributos(df)`; o erro é `ValueError` (no patch, `KeyError`) |
| Validação do histórico | Adotada |
| `calcular_nota_candidato` | Substituída pela regra exata da D12. A do patch não arredondava e fazia a média de 4 provas se uma faltasse |
| `diferenca_nota_recalculada` | Adotada |
| `montar_entrada_treino` e `montar_entrada_inferencia` | Adotadas; `TEM_NOTA_ZERO` passou a usar `limpeza.marcar_nota_zero`, a mesma da limpeza |
| `tests/test_montagem.py` e acréscimos em `tests/test_features.py` | Aplicados sem alteração |
| `tests/test_atributos.py` | Adaptado a `colunas_atributos(df)` e a `ValueError` |
| `tests/test_vazamento.py` | Não aplicado: mantido o de `179593a`, mais completo |
| Regra do corte anterior com `>=` (D5) | Não adotada: contraria o código e os resultados publicados. Registrada como P3 |
| Textos do README e do `DECISOES.md` | Reescritos conforme o estado verificado |
| `reports/relatorio_preparacao_sem_csv.md` | Incluído sem alteração, como histórico. Descreve a versão antiga (por exemplo, `colunas_atributos()` sem argumento e 46 testes) |

## 2. D12: a regra da nota

Em `features.calcular_nota_candidato`, única função usada no treino e na inferência:

1. cada prova vira um inteiro em décimos;
2. somam-se as cinco;
3. `produto = soma × (100 + PERCENTUAL_BONUS)`;
4. um só arredondamento, no fim: `centésimos = (produto + 25) // 50` (meio centésimo para cima);
5. `nota = centésimos / 100`.

Prova ausente, não finita ou com 2 casas, e bônus fracionado, geram `ValueError`.

A regra foi **reproduzida nos dados históricos de 2017–2023**. Não é apresentada como regra oficial
atual do SISU: nenhuma fonte oficial foi consultada. O alvo continua calculado com a nota do CSV.

## 3. Resultados nos dados reais

Saída de `python -m src.verificar_dados`: `reports/conciliacao/verificacao_dados.md` e CSVs.

| Verificação | Resultado |
|---|---|
| Nota recalculada x original, igualdade exata | 0 diferenças em 1.080.713 linhas, com 319 empates (todos em 2023) |
| Alvo com a nota recalculada | Nenhuma linha mudaria |
| Treino x inferência, 17 colunas, valores, tipos e índice, por ano | 0 diferenças em 1.080.713 linhas |
| Idem, uma inscrição por vez (amostra, semente 42) | 0 diferenças em 1.000 |
| Entradas montadas x colunas do parquet (15 atributos) | 0 diferenças |
| Só o histórico de N−1 | 0 diferenças |
| Cortes de todos os outros anos adulterados | 0 diferenças |
| `NOTA_CORTE` do próprio ano adulterada na entrada | 0 diferenças |
| Corte N−1 por uma segunda implementação | 0 diferenças |
| Sem o histórico de N−1 | Nenhuma linha recebe corte |
| Colunas excluídas pela D2 nas entradas | Nenhuma das 12 conferidas |
| Histórico versionado x reconstruído | Idêntico (8.343 linhas, nenhum conflito) |

A inferência recebe só as colunas que a aplicação teria (`COLUNAS_ENTRADA`) e usa o histórico
versionado.

### Sem histórico

| Ano | Linhas sem corte | Chave inexistente em N−1 | Chave sem valor em N−1 |
|---|---|---|---|
| 2017 | 252.687 | 252.687 | 0 |
| 2018 | 9.566 | 9.566 | 0 |
| 2019 | 50.920 | 50.907 | 13 |
| 2020 | 39.156 | 39.142 | 14 |
| 2021 | 6.582 | 6.575 | 7 |
| 2022 | 1.772 | 1.742 | 30 |
| 2023 | 7.580 | 7.475 | 105 |

Nessas linhas, corte e diferença ficam ausentes juntos e os outros 15 atributos ficam preenchidos.
Um campus inexistente ou um ano-alvo sem N−1 não geram erro: o corte fica ausente. Com as
inscrições de 2023 e ano-alvo 2024, 91.178 recebem o corte de 2023 e 78 não.

### Alvo (usado para atualizar D6–D8)

| Conjunto | Anos | Com alvo | Positivos | Taxa |
|---|---|---|---|---|
| Histórico | 2017 | 252.606 | 6.923 | 2,74% |
| Treino | 2018–2021 | 644.845 | 29.034 | 4,50% |
| Validação | 2022 | 91.750 | 7.154 | 7,80% |
| Teste | 2023 | 91.178 | 7.076 | 7,76% |

As contagens e a cobertura do corte anterior são iguais às da primeira verificação.

## 4. Efeito nos baselines publicados

`src/baselines.py` **mudou**: as entradas passaram a ser montadas por `montar_entrada_treino`, o que
inclui recalcular a nota. A regra, o Pipeline, o modelo e as métricas não mudaram.

**As entradas são idênticas.** O `main()` real de `179593a` e o atual foram executados com o parquet
real e interrompidos antes do ajuste. Valores, tipos, colunas e ordem das linhas coincidem em
`x_treino` (644.845 × 17), `y_treino`, `x_validacao` (91.750 × 17) e `y_validacao`. A tabela
`verificacao_entradas_baselines.csv` repete a comparação dentro do projeto.

Os baselines **não foram executados de novo**. Os números abaixo são os publicados e continuam
vinculados à configuração em que foram obtidos:

- commit `179593a`; treino 2018–2021, validação 2022, histórico de cortes de 2017–2021;
- 17 atributos da D2; mediana com indicador, padronização e one-hot; `LogisticRegression` com
  `liblinear`, semente 42, limiar 0,5, com e sem `class_weight="balanced"`;
- regra: `NOTA_CANDIDATO > CORTE_ANO_ANTERIOR`.

| Abordagem (validação 2022) | PR-AUC (AP) | Precisão | Revocação | F1 |
|---|---|---|---|---|
| Regra do corte anterior | 0,2987 | 0,8890 | 0,2722 | 0,4167 |
| Regressão logística sem pesos | 0,6991 | 0,9271 | 0,2523 | 0,3967 |
| Regressão logística com pesos balanceados | 0,6836 | 0,6661 | 0,5485 | 0,6016 |

`reports/baselines_2022.md` e `.csv` não foram alterados.

## 5. Arquivos alterados

| Arquivo | Alteração desde `179593a` |
|---|---|
| `.gitignore` | Ignora só `/sisu_ufma_2017_2023.csv` |
| `src/features.py` | Validação do histórico, seleção de atributos, nota (D12), montagem (D13) |
| `src/limpeza.py` | `marcar_nota_zero` |
| `src/baselines.py` | Usa `montar_entrada_treino` |
| `src/verificar_dados.py` | Verificações |
| `tests/test_atributos.py`, `test_montagem.py`, `test_nota.py`, `test_dados_reais.py`, `test_verificar_dados.py` | Novos |
| `tests/test_features.py` | Testes do histórico |
| `docs/DECISOES.md` | D1 (referência obsoleta), D2, D5 (remissão), D6, D7, D8, D12, D13 e propostas |
| `README.md` | Comandos e estado |
| `reports/` | Relatórios e tabelas; nenhum anterior a esta branch foi modificado |

Commits: `c946c22` (`.gitignore`), `d567e0e` (primeira verificação), `69efac5` (conciliação e D12),
`a473b95` (verificação conciliada) e o commit deste relatório.

## 6. Comandos e testes

Python 3.13.17, `.venv` com `requirements.txt`.

```bash
python -m src.limpeza
python -m src.features         # histórico regravado sem diferença no git
python -m src.verificar_dados
python -m pytest -q -rs
```

| Versão | Sem parquet | Com parquet real |
|---|---|---|
| `179593a` | 16 passaram, 1 pulado | 17 passaram |
| `d567e0e` | 25 passaram, 1 pulado | – |
| `69efac5` | 80 passaram, 1 pulado | – |
| `a473b95` (código final) | 84 passaram, 8 pulados | 92 passaram |

A coluna "sem parquet" foi obtida em cópias limpas de cada commit.

## 7. Pendências

**Decisões da equipe**

1. **Protocolo de calibração (D8).** Continua pendente; decidir antes de treinar o modelo final.
2. **P3:** regra com `>` ou `>=`.
3. **P4:** redação da D9 sobre pessoas repetidas entre anos.
4. **Alcance temporal.** O histórico termina em 2023: a montagem só encontra corte anterior para
   ano-alvo até 2024. Decidir se a aplicação será apresentada como retrospectiva ou se os dados
   serão atualizados.
5. **Fonte da regra da nota:** confirmar em edital ou norma antes de citá-la como regra do SISU.

**Ainda não existe**

6. Modelo final, `pipeline_final.joblib`, `metadados.json` e aplicação; o tratamento dos ausentes
   pelo Pipeline final não foi verificado.
7. Na aplicação: validar a faixa das notas e do bônus e oferecer as opções de curso e modalidade a
   partir do histórico, porque um texto diferente resulta em corte ausente sem erro.

**Documentação**

8. A D5 cita "~560 mil linhas de treino"; o treino tem 644.845 linhas com alvo.
9. O relatório do `68a3e06` aponta diferença entre "132 cursos e 11 campi" (README) e as contagens
   do histórico; não foi conferida.
10. Licença dos dados e consolidação do CSV continuam abertas.
