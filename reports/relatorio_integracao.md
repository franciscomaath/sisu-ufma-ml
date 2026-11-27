# Relatório de integração e verificação com os dados reais

Data: 27/11/2026. Branch `integracao/preparacao-dados-reais`, criada a partir de `origin/main`
(`179593aaf69e6348623f9e8ac6204514b9faeebb`).

## 1. Situação

- **A conciliação com o commit `68a3e06` não foi feita: o patch não estava disponível.**
- A limpeza foi reproduzida com o CSV real e as verificações pedidas foram executadas, exceto a
  consistência com uma montagem de inferência, que não existe nesta base (seção 5).
- Nenhum modelo foi treinado nos dados reais e nenhuma métrica de desempenho de 2023 foi calculada.
- Nada foi commitado, enviado ou mesclado. A `main` continua em `179593a`.

## 2. Patch e conciliação

Procurei o patch no repositório e em pastas locais da máquina de trabalho. O commit `68a3e06` não
existia no repositório local e o remoto só tinha a referência `main`.

| Item | Situação |
|---|---|
| Preservar D2 e baselines | Sem diferença em relação a `origin/main` |
| Montagem de entrada (D13) | Pendente: depende do patch |
| Cálculo compartilhado da nota (D12) | Pendente: depende do patch |
| Validação do histórico | Pendente: depende do patch (hoje não há conflito nos dados) |
| Testes do patch | Pendente: depende do patch |
| Assinatura de `colunas_atributos()` | Pendente: em `9eec742` e em `179593a` ela é `colunas_atributos(df)`; a versão diferente só está no patch |

Não reescrevi essas contribuições por conta própria, para não criar uma terceira versão concorrente.
Para gerar o patch onde o commit existe:

```bash
git format-patch -1 68a3e06 --stdout > 68a3e06.patch
```

## 3. Alterações nesta branch

| Arquivo | Alteração |
|---|---|
| `src/verificar_dados.py` | Novo: verificações somente leitura |
| `tests/test_verificar_dados.py` | Novo: 9 testes com dados sintéticos |
| `reports/verificacao_dados.md` e `reports/verificacao_*.csv` | Novos: saída do script |
| `docs/DECISOES.md` | Seção "Propostas em aberto" (P1 e P2); nenhuma decisão mudou |
| `README.md` | Uma linha com o comando de verificação |
| `reports/relatorio_integracao.md` | Este relatório |

## 4. Ambiente e comandos

- Windows 11. Não havia Python instalado: instalei o 3.13.17 com
  `winget install --id Python.Python.3.13 --exact --source winget --scope user`.
- Ambiente em `.venv`, com `pip install -r requirements.txt`; `pip check` sem conflitos.
  O `requirements.txt` fixa só as dependências diretas.
- CSV: 359.673.102 bytes, SHA-256
  `AD073F0C345A0333A32609266B2C716DF5C0CB85DF3A3A0A274F7734664283B9`.
  `data/raw/sisu_ufma_2017_2023.csv` é um *hard link* para o arquivo da raiz do projeto.

```bash
python -m src.limpeza          # parquet limpo
python -m src.features         # histórico e cobertura
python -m src.verificar_dados  # tabelas de verificação
python -m pytest -q -rs        # testes
```

`python -m src.baselines` não foi executado, porque treina modelos.

| Momento | Resultado dos testes |
|---|---|
| Base `179593a`, sem parquet | 16 passaram, 1 pulado (parquet não gerado) |
| Testes da base, com parquet real | 17 passaram |
| Base e testes novos | 26 passaram |

`src.features` regravou `models/historico_cortes.parquet` sem diferença no git e gerou o
`.joblib`, que é ignorado.

## 5. Resultados

### Limpeza

1.080.713 linhas; 334 sem alvo; 3.067 idades fora de [14, 80]; 7 linhas com nota zero.

### Alvo `NOTA_SUFICIENTE`

| Ano | Conjunto | Inscrições | Com alvo | Sem alvo | Positivos | Taxa |
|---|---|---|---|---|---|---|
| 2017 | Histórico | 252.687 | 252.606 | 81 | 6.923 | 2,74% |
| 2018 | Treino | 186.574 | 186.564 | 10 | 7.012 | 3,76% |
| 2019 | Treino | 189.741 | 189.735 | 6 | 7.496 | 3,95% |
| 2020 | Treino | 156.340 | 156.325 | 15 | 7.325 | 4,69% |
| 2021 | Treino | 112.269 | 112.221 | 48 | 7.201 | 6,42% |
| 2022 | Validação | 91.846 | 91.750 | 96 | 7.154 | 7,80% |
| 2023 | Teste | 91.256 | 91.178 | 78 | 7.076 | 7,76% |

| Conjunto | Inscrições | Com alvo | Sem alvo | Positivos | Taxa |
|---|---|---|---|---|---|
| Histórico | 252.687 | 252.606 | 81 | 6.923 | 2,74% |
| Treino | 644.924 | 644.845 | 79 | 29.034 | 4,50% |
| Validação | 91.846 | 91.750 | 96 | 7.154 | 7,80% |
| Teste | 91.256 | 91.178 | 78 | 7.076 | 7,76% |

As 334 linhas sem alvo são todas por falta de `NOTA_CORTE`; 104 delas têm `APROVADO = 1`
(16 em 2021, 51 em 2022, 37 em 2023).

### Fórmula da nota: há divergência

| Regra | Linhas divergentes | Maior diferença | Alvo mudaria de 1 para 0 |
|---|---|---|---|
| Sem arredondar (texto atual da D12) | 113.458 | 0,005 | 1.387 (1.228 só por ponto flutuante) |
| 2 casas, meio para cima, conta inteira | 0 | 0 | 0 |
| `round(2)` do pandas | 128 | 0,01 | 0 |
| Truncar em 2 casas | 56.967 | 0,01 | 159 |

- "Divergente" é diferença acima de 1e-9. Em nenhuma regra o alvo mudaria de 0 para 1.
- Há 319 empates exatos no meio do centésimo, todos em 2023; o `round(2)` erra 128 deles.
- Sem arredondar, o alvo mudaria em todos os anos (154 a 241 linhas por ano), inclusive nos sem
  bônus, porque a média em ponto flutuante pode ficar logo abaixo do corte.
- O alvo e os cálculos adotados **não foram modificados**. Proposta em `docs/DECISOES.md`, P1.

### Cortes por chave e ano

8.343 chaves, nenhuma com mais de um corte e nenhuma misturando corte preenchido e ausente.
Em 132 chaves (334 linhas) não há corte; elas estão no histórico versionado com valor ausente.
`construir_historico` usa `max()`, portanto um conflito futuro passaria sem aviso.

O histórico reconstruído é idêntico ao versionado.

### Cobertura do corte anterior

| Ano | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|
| Linhas com corte | 177.008 | 138.821 | 117.184 | 105.687 | 90.074 | 83.676 |
| Cobertura | 94,9% | 73,2% | 75,0% | 94,1% | 98,1% | 91,7% |
| Edição 1 | 96,7% | 92,4% | 63,4% | 93,1% | 96,7% | 90,7% |
| Edição 2 | 91,9% | 52,2% | 96,0% | 95,8% | 99,8% | 93,1% |

Os totais coincidem com o relatório da etapa 1 e com a D3.

### Montagem de treino e inferência

**Não existe função de montagem de inferência em `179593a`**, então a consistência entre treino e
aplicação não pôde ser verificada. O que foi testado, com `adicionar_corte_anterior`:

| Comparação com a montagem em lote | Linhas | Diferentes | Acima de 1e-9 |
|---|---|---|---|
| Histórico versionado em vez do reconstruído | 1.080.713 | 0 | 0 |
| Histórico só com anos anteriores | 1.080.713 | 0 | 0 |
| Uma inscrição por vez (amostra, semente 42) | 1.000 | 0 | 0 |
| Nota recalculada com a regra de 2 casas | 1.080.713 | 0 | 0 |
| Nota recalculada sem arredondar | 1.080.713 | 291.305 | 47.666 |

Falta testar o que a aplicação terá de derivar da entrada: nota a partir das 5 provas,
`TEM_NOTA_ZERO`, nome de curso normalizado, modalidade, tipos das colunas e previsão para um ano
sem corte próprio.

## 6. Resultados publicados, preservados

Os baselines de Kadu em `reports/baselines_2022.md` e `.csv` não foram alterados nem recalculados.

- **Versão:** commit `179593a` ("feat: implementa D2 e avalia os baselines em 2022").
- **Dados:** treino 2018–2021 (644.845 linhas com alvo), validação 2022 (91.750), histórico de
  cortes de 2017–2021.
- **Configuração:** 17 atributos da D2; imputação pela mediana com indicador, padronização e
  one-hot; `LogisticRegression` com `liblinear`, semente 42, limiar 0,5, com e sem
  `class_weight="balanced"`.

| Abordagem (validação 2022) | PR-AUC (AP) | Precisão | Revocação | F1 |
|---|---|---|---|---|
| Regra do corte anterior | 0,2987 | 0,8890 | 0,2722 | 0,4167 |
| Regressão logística sem pesos | 0,6991 | 0,9271 | 0,2523 | 0,3967 |
| Regressão logística com pesos balanceados | 0,6836 | 0,6661 | 0,5485 | 0,6016 |

As contagens de linhas do relatório dos baselines (644.845 e 91.750, com 79 e 96 excluídas) foram
reproduzidas. As métricas acima são as publicadas; não foram conferidas por nova execução.

## 7. Pendências

**Para continuar a integração**

1. Fornecer o patch do `68a3e06`.
2. Depois dele: montagem de entrada, cálculo da nota, validação do histórico, testes,
   `colunas_atributos()` e nova execução da verificação.

**Decisões da equipe**

3. P1: incluir o arredondamento na D12.
4. P2: atualizar as taxas de D6, D7 e D8, que descrevem `APROVADO`.

**Cuidados**

5. O CSV na raiz do projeto não está no `.gitignore`: não usar `git add .`. Pode-se removê-lo da
   raiz (o *hard link* em `data/raw/` permanece) ou ignorá-lo.
6. `reports/relatorio_etapa1.md` é anterior à D1: ainda cita `APROVADO` como alvo e 11 testes.
7. Seguem abertas a licença dos dados e a documentação da consolidação do CSV.
