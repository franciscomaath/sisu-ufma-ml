# Relatório complementar — Preparação sem o CSV (26/11/2026)

Complementa `reports/relatorio_etapa1.md`, que permanece como registro histórico (alvo antigo
`APROVADO`, 11 testes). **Nenhum modelo foi treinado e nenhuma métrica de 2023 foi calculada.**
Nenhum dado bruto ou processado foi versionado.

- Base: `main` = `origin/main`, commit `9eec7425a95c6d0d5df23c70732e1ecf9d136aef`.
- Branch de trabalho: `claude/cool-hypatia-1xejpr`.
- Ambiente: Python 3.13.18, venv criado fora do repositório com `pip install -r requirements.txt`.

## 1. Legenda

- **Sintético:** implementado e testado só com dados sintéticos (e, onde indicado, com o
  `models/historico_cortes.parquet` versionado).
- **Real:** verificado nos dados reais. **Nada desta etapa está nessa categoria**, porque o CSV bruto
  e o parquet limpo não existem no ambiente.

## 2. O que foi feito

| # | Item | Estado |
|---|---|---|
| 1 | D2: `ATRIBUTOS_NUMERICOS`, `ATRIBUTOS_CATEGORICOS`, `ATRIBUTOS` em `config.py`; `colunas_atributos()` devolve a lista explícita; `selecionar_atributos(df)` seleção estrita | Sintético |
| 2 | D12/D13: `calcular_nota_candidato`, `diferenca_nota_recalculada`, `montar_entrada_treino`, `montar_entrada_inferencia` | Sintético (a consulta ao histórico também foi testada com o parquet versionado) |
| 3 | `construir_historico()` dá `ValueError` com mais de um corte não nulo diferente por chave e ano | Sintético |
| 4 | Baseline histórico descrito com `>=` em D5 (não implementado, não treinado) | Documentação |
| 5 | README e `docs/DECISOES.md` atualizados com o estado real | Documentação |

## 3. Contrato da montagem da entrada (`src/features.py`)

```python
montar_entrada_treino(df, historico) -> DataFrame
montar_entrada_inferencia(df, historico, ano_alvo: int) -> DataFrame
```

- **Entrada `df`:** `CODIGO_CAMPUS`, `NOME_CURSO_NORM`, `GRAU`, `TURNO`, `MODALIDADE`, `EDICAO`, `OPCAO`,
  `PERCENTUAL_BONUS`, `NOTA_L`, `NOTA_CH`, `NOTA_CN`, `NOTA_M`, `NOTA_R`; no treino, também `ANO`.
  Falta de coluna → `ValueError`. Outras colunas são ignoradas e descartadas.
- **`historico`:** tabela de `construir_historico` / `models/historico_cortes.parquet`
  (chave + `ANO` + `CORTE_ANO_ANTERIOR`, que ali é o corte do próprio `ANO` da linha).
- **Ano:** a linha do ano N usa o corte do ano N−1, mesma oferta, modalidade e edição. No treino N vem
  de `ANO`; na inferência N é `ano_alvo`, explícito. Cortes de N ou posteriores nunca são usados.
- **Saída:** exatamente `config.ATRIBUTOS`, na ordem da lista, com o índice de `df`. A entrada não é modificada.
- **Cálculos:** `NOTA_CANDIDATO` = média das 5 notas × (1 + `PERCENTUAL_BONUS`/100), sempre recalculada
  (inclusive no treino); `TEM_NOTA_ZERO` = alguma das 5 notas igual a 0; `CORTE_ANO_ANTERIOR` e
  `DIF_CORTE_ANTERIOR` via `adicionar_corte_anterior` (reaproveitada). Sem correspondência → NaN (D3).
- **Fora do escopo da função:** validação de faixa das notas, tratamento de categoria desconhecida,
  imputação e codificação (ficam para a aplicação e o `Pipeline`).

## 4. Comandos e resultados

```bash
pip install -r requirements.txt   # no venv, fora do repositório
python -m pytest -q -rs -p no:cacheprovider
```

```
..............................................s                          [100%]
SKIPPED [1] tests/test_vazamento.py:17: parquet limpo não gerado
46 passed, 1 skipped in 0.48s
```

| Arquivo | Testes |
|---|---|
| `tests/test_atributos.py` (novo) | 17 |
| `tests/test_montagem.py` (novo) | 12 |
| `tests/test_features.py` | 8 (5 novos, do histórico) |
| `tests/test_limpeza.py` | 6 |
| `tests/test_split.py` | 2 |
| `tests/test_vazamento.py` | 2 (1 pulado) |

- **Teste pulado:** `test_atributos_no_parquet_real`, que exige `data/processed/sisu_limpo.parquet`.
- Antes: 12 passaram e 1 foi pulado. Depois: 46 passaram e 1 foi pulado, mesmo teste pulado.
- `test_inferencia_com_historico_real_versionado` **não** foi pulado: usa `models/historico_cortes.parquet`.
  Ele só confirma a consulta N−1 numa chave real do histórico; não é verificação de dados do CSV.

## 5. Pendências que exigem o CSV

1. Confirmar a fórmula de `NOTA_CANDIDATO` (D12) com `diferenca_nota_recalculada` e decidir uma tolerância;
   se houver divergência, rever D12 antes de treinar, pois o alvo usa `NOTA_CANDIDATO` do CSV.
2. Reproduzir a limpeza e as contagens do alvo atual por ano e conjunto (50.187 positivos, 334 sem alvo
   etc. são números documentados em D1, não reproduzidos).
3. Rodar `construir_historico` no parquet limpo e confirmar que não há conflito de cortes. O relatório da
   Etapa 1 afirma 0 chaves ambíguas com `.max()`, mas isso agora é checado explicitamente.
4. Recalcular cobertura de `CORTE_ANO_ANTERIOR` com `montar_entrada_treino`; os 73%–98% são da Etapa 1.
5. Executar `test_atributos_no_parquet_real` sem ser pulado.
6. Exploração e figuras; reconciliar "132 cursos e 11 campi" (README) com 112 combinações e 12 códigos de campus
   do histórico; licença e consolidação do CSV.

## 6. Decisões metodológicas ainda abertas (não alteradas por mim)

- **Protocolo de calibração e limiar (D8):** hoje tudo é ajustado e comparado em 2022. A comparação de
  calibradores nas mesmas observações do ajuste favorece resultados otimistas. Precisa de protocolo
  explícito (guia, §9.4) antes do treino.
- **Escore do baseline:** a regra `>=` devolve só 0/1; qual escore usar para ROC-AUC e PR-AUC?
- **`NOTA_CANDIDATO` recalculada também no treino:** adotei um único caminho (D13). Se a fórmula não
  bater com o CSV, a alternativa é usar a nota do dado no treino, o que reabre o risco de divergência.
- **Frase de D4/D9:** a divisão por ano não demonstra que uma pessoa não aparece em anos diferentes
  (não há ID). O texto continua com a formulação antiga; não mexi por estar fora do escopo.
- **Cenário da aplicação:** com histórico até 2023, `ano_alvo=2024` é uma simulação retrospectiva, não
  previsão validada para 2026 ou 2027.
- **Cursos sem edição na outra metade, `BONUS_MA`, 2023 Medicina regional (D3/D10):** seguem como NaN.
