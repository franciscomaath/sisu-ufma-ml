# Decisões do projeto

Registro das decisões de modelagem do `sisu-ufma-ml`. Cada decisão diz **o quê**, **por quê** e
**o que precisa ser implementado**. O relatório da Etapa 1 (`reports/relatorio_etapa1.md`) listava
estas questões como pendentes (D1 a D8); este documento as resolve.

As decisões estão marcadas como **Adotadas**. Se alguém da equipe discordar de alguma, altere este
arquivo em um commit próprio antes de mudar o código, para o registro continuar coerente.

---

## Visão geral

| # | Tema | Decisão | Status |
|---|---|---|---|
| D1 | Alvo | `NOTA_SUFICIENTE = NOTA_CANDIDATO >= NOTA_CORTE` | Implementada |
| D2 | Atributos | Lista explícita (allowlist) em `config.py` | Implementada e verificada nos dados |
| D3 | Corte anterior ausente | NaN + indicador de ausência no Pipeline | Adotada |
| D4 | Divisão | Temporal: treino 2018–2021, validação 2022, teste 2023 | Implementada |
| D5 | Modelos | Regra + regressão logística (baselines), LightGBM, MLP | Baselines avaliados em 2022 |
| D6 | Desbalanceamento | Peso de classe, comparado na validação | Comparado na regressão logística |
| D7 | Métricas | Seleção por PR-AUC; Brier para calibração | Adotada |
| D8 | Calibração e limiar | Ajustados em 2022 | Adotada; protocolo de calibração pendente |
| D9 | Candidato repetido | Limitação declarada no artigo | Adotada |
| D10 | Casos sem histórico | Aceitar a perda e declarar | Adotada |
| D11 | Histórico de cortes | Parquet versionado no git | Implementada |
| D12 | Nota do candidato | Calculada das 5 notas; 2 casas, meio centésimo para cima | Implementada; reproduz o CSV de 2017–2023 |
| D13 | Contrato Modelo ↔ Aplicação | Ver seção própria | Montagem implementada e verificada; modelo e aplicação não existem |

---

## D1. Alvo: `NOTA_SUFICIENTE` em vez de `APROVADO`

**Decisão.** O alvo do modelo passa a ser:

```python
NOTA_SUFICIENTE = (NOTA_CANDIDATO >= NOTA_CORTE)
```

**Por quê.** No SISU, quem é aprovado na 1ª opção não é chamado na 2ª. Por isso `APROVADO`
mistura duas situações diferentes: "a nota não bastava" e "foi aprovado em outra opção".
Os dados confirmam:

Entre as inscrições com nota e corte disponíveis:

| | Não aprovado | Aprovado |
|---|---|---|
| Nota abaixo do corte | 1.030.192 | 0 |
| Nota igual ou acima do corte | 10.355 | 39.832 |

Das 10.355 inscrições com nota suficiente e `APROVADO = N`, **10.341 são da 2ª opção**.
Com `APROVADO` como alvo, o modelo aprenderia que a 2ª opção "atrapalha", quando a nota bastava.

`NOTA_SUFICIENTE` responde exatamente à pergunta do usuário da aplicação:
*"com a minha nota, eu passaria neste curso, nesta modalidade?"*

**Cuidados.**
- `NOTA_CORTE` do próprio ano continua **proibida como atributo**. Ela só serve para construir o alvo
  e o histórico de cortes.
- As 334 inscrições sem corte permanecem no parquet com alvo ausente (`Int8` anulável).
  Dessas, 104 têm `APROVADO = 1`, o que explica a diferença entre 39.832 e os 39.936
  aprovados do relatório da Etapa 1. As linhas sem alvo ficam fora do ajuste e da avaliação supervisionada.
- Manter `APROVADO` no parquet, apenas para uma tabela de comparação no artigo
  (desempenho do modelo final medido contra o alvo antigo).

**Implementação.** `config.ALVO` aponta para `NOTA_SUFICIENTE`, que é criada em `src/limpeza.py`.
`APROVADO` fica fora de `config.ATRIBUTOS` (D2).

---

## D2. Atributos: lista explícita

**Decisão.** Os atributos do modelo são definidos por uma **lista explícita** em `src/config.py`.

**Atributos permitidos:**

```python
ATRIBUTOS_NUMERICOS = [
    "NOTA_L", "NOTA_CH", "NOTA_CN", "NOTA_M", "NOTA_R",
    "NOTA_CANDIDATO", "PERCENTUAL_BONUS",
    "CORTE_ANO_ANTERIOR", "DIF_CORTE_ANTERIOR",
    "TEM_NOTA_ZERO",
]
ATRIBUTOS_CATEGORICOS = [
    "CODIGO_CAMPUS", "NOME_CURSO_NORM", "GRAU", "TURNO",
    "MODALIDADE", "EDICAO", "OPCAO",
]
ATRIBUTOS = ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS
```

**Excluídos e motivo:**

| Coluna | Motivo |
|---|---|
| `NOTA_CORTE`, `CLASSIFICACAO`, `MATRICULA` | Vazamento: só existem depois do resultado |
| `APROVADO` | Alvo antigo (D1) |
| `ANO` | O modelo precisa generalizar para anos que não viu |
| `SEXO`, `IDADE` | Ética: o corte depende só da nota e da modalidade. Ver variante abaixo |
| `UF_CANDIDATO`, `MUNICIPIO_CANDIDATO` | Ética e cardinalidade (3.994 municípios). O efeito "estudou no MA" já está no bônus |
| `MUNICIPIO_CAMPUS` | Redundante com `CODIGO_CAMPUS` |
| `GRUPO_CONCORRENCIA` | Redundante com `MODALIDADE` |

**Variante para a seção de ética.** Treinar o melhor modelo também **com** `SEXO` e `IDADE`, só
para comparação, e reportar a diferença. Se melhorar a métrica, isso indica que o modelo estaria
usando um atalho demográfico, o que é assunto para a discussão de vieses.

**Implementação.** `features.colunas_atributos(df)` verifica a presença das colunas obrigatórias e
retorna os atributos na ordem de `config.ATRIBUTOS`; `features.selecionar_atributos(df)` devolve só
essas colunas. A montagem da entrada (D13) termina nessa seleção. Os testes conferem a lista e a
exclusão de colunas indevidas, e nas entradas montadas dos dados de 2017–2023 não há nenhuma
coluna excluída.

---

## D3. Corte do ano anterior ausente

**Contexto.** A cobertura de `CORTE_ANO_ANTERIOR` é de 73% a 98% conforme o ano. As faltas se
concentram em 2019–2020: o grupo `BONUS_MA` surgiu em 2019 (sem histórico) e nem todos os cursos
oferecem vagas nas duas edições.

| Ano | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|
| Cobertura | 94,9% | 73,2% | 75,0% | 94,1% | 98,1% | 91,7% |

**Decisão.** Manter NaN e deixar o Pipeline imputar **com indicador de ausência**
(`SimpleImputer(add_indicator=True)`), para o modelo distinguir "sem histórico" de "corte baixo".

**Experimento opcional (validação).** Fallback: quando não houver corte da mesma edição no ano
anterior, usar o corte do mesmo curso e modalidade na **outra edição** do ano anterior. Adotar só
se melhorar a PR-AUC em 2022, e reportar no artigo de qualquer forma.

---

## D4. Divisão temporal

| Conjunto | Anos | Uso |
|---|---|---|
| Histórico | 2017 | Só alimenta o corte do ano anterior |
| Treino | 2018–2021 | Ajuste do Pipeline e dos modelos |
| Validação | 2022 | Hiperparâmetros, calibração, limiar, escolha do modelo |
| Teste | 2023 | **Avaliado uma única vez**, no final |

Nunca usar divisão aleatória: o mesmo candidato aparece em duas linhas (opções 1 e 2) e o objetivo
é prever anos futuros. Implementado em `src/split.py`.

---

## D5. Modelos

O enunciado exige um baseline e **pelo menos dois modelos de famílias diferentes**.
Random Forest e LightGBM são da **mesma família** (ensemble de árvores), por isso não basta usar os dois.

| Papel | Modelo | Família |
|---|---|---|
| Baseline 1 | Regra: suficiente se `NOTA_CANDIDATO > CORTE_ANO_ANTERIOR` (sem corte anterior: prever 0) | Regra |
| Baseline 2 | Regressão logística | Linear |
| Modelo 1 | LightGBM | Ensemble de árvores |
| Modelo 2 | Rede neural MLP (`MLPClassifier` com early stopping) | Rede neural |

SVM não é viável com ~560 mil linhas de treino. Se a MLP for lenta, pode treinar em subamostra
estratificada do treino, com semente fixa e documentada.

A regra do Baseline 1 é o adversário principal: o modelo de ML precisa superá-la. Se não superar,
isso também é um resultado e deve ser discutido com honestidade no artigo.

A regra usa `>` e o alvo usa `>=` (D1). A diferença está registrada na proposta P3.

---

## D6. Desbalanceamento

A classe positiva é rara. Nas linhas com alvo `NOTA_SUFICIENTE`:

| Ano | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|
| Com alvo | 252.606 | 186.564 | 189.735 | 156.325 | 112.221 | 91.750 | 91.178 |
| Positivos | 6.923 | 7.012 | 7.496 | 7.325 | 7.201 | 7.154 | 7.076 |
| Taxa | 2,74% | 3,76% | 3,95% | 4,69% | 6,42% | 7,80% | 7,76% |

No conjunto de treino (2018–2021) são 29.034 positivos em 644.845 linhas (4,50%).

Comparar, na validação, o modelo com e sem peso de classe (`class_weight="balanced"` /
`scale_pos_weight`). Não usar SMOTE nem reamostragem fora do Pipeline.

---

## D7. Métricas

- **Seleção do modelo:** PR-AUC na validação (mais informativa que ROC-AUC com classe rara).
- **Reportar no teste, para todos os modelos:** F1, precisão, revocação, ROC-AUC, PR-AUC, Brier e
  matriz de confusão. Destacar a melhor de cada coluna na tabela.
- **Não usar acurácia como métrica principal:** prever "não" para todos já acertaria 95,50% das
  linhas com alvo no treino e 92,20% na validação.

---

## D8. Calibração e limiar

**Contexto.** A taxa de positivos sobe ao longo dos anos (3,76% em 2018, 7,80% em 2022, 7,76% em
2023; D6) porque o número de positivos varia pouco (de 7.012 a 7.496 por ano em 2018–2023) e as
inscrições caem. Com 4,50% de positivos no treino, espera-se que o modelo **subestime** as
probabilidades em 2022–2023 (ainda a conferir na validação).

**Decisão.**
- Recalibrar o modelo escolhido **em 2022**, cuja taxa é parecida com a de 2023. Comparar Platt
  (sigmoid) e isotônica pelo Brier score.
- Escolher o limiar de decisão em 2022 (maximizando F1) e congelá-lo antes do teste.
- Mostrar a curva de calibração antes e depois no artigo.

A aplicação mostra a **probabilidade calibrada**, por isso a calibração é parte central do trabalho,
e não um detalhe.

**Pendente: protocolo de calibração.** O texto acima, junto com a D4, usa 2022 para escolher
hiperparâmetros e modelo, ajustar o calibrador, comparar os calibradores pelo Brier e escolher o
limiar. Comparar nas mesmas observações usadas no ajuste favorece resultados otimistas, sobretudo
para a isotônica. A equipe ainda precisa decidir o protocolo, antes de treinar o modelo final.

---

## D9. Candidato repetido

O mesmo candidato pode aparecer nas opções 1 e 2 e o dado não tem identificador pessoal. A divisão
por ano evita que a mesma pessoa esteja no treino e no teste, mas as linhas não são independentes
dentro de um ano. Declarar como limitação. `OPCAO` entra como atributo.

---

## D10. Casos sem histórico

Aceitar a perda de corte anterior e declarar como limitação:
- `AC_MEDICINA_REGIONAL` (2023, 5.105 linhas, só no teste);
- campi que mudaram de código (Balsas em 2023) e Turismo, que mudou de campus em 2020.

Manter `CODIGO_CAMPUS` na chave: trocar por município uniria campi diferentes de São Luís e de Imperatriz.

---

## D11. Histórico de cortes

`models/historico_cortes.parquet` (52 KB) fica **versionado**, para a aplicação funcionar logo
após clonar o repositório. O `.joblib` continua ignorado.

---

## D12. Nota do candidato

**Decisão.** Média simples das 5 notas, com o bônus, em 2 casas decimais, com o meio centésimo
para cima:

```
NOTA_CANDIDATO = arredondar2( média das 5 notas × (1 + PERCENTUAL_BONUS / 100) )
```

**Ordem das operações**, em aritmética inteira, para não depender de ponto flutuante:

1. cada prova vira um inteiro em décimos (650,3 → 6503);
2. somam-se as cinco;
3. `produto = soma × (100 + PERCENTUAL_BONUS)`;
4. um único arredondamento, no fim, depois do bônus: `centésimos = (produto + 25) // 50`;
5. `NOTA_CANDIDATO = centésimos / 100`.

As provas devem ter no máximo 1 casa decimal, o bônus deve ser inteiro e nada pode faltar; caso
contrário a função dá `ValueError`. A faixa das notas e os bônus permitidos são validados pela
aplicação.

**Implementação.** `features.calcular_nota_candidato(df)` é a única função da nota, usada no treino
e na inferência por meio da montagem da D13. A aplicação pede as 5 notas e o bônus; não há pesos
por curso.

**Verificado** (`python -m src.verificar_dados` e `tests/test_dados_reais.py`): nas 1.080.713
inscrições de 2017–2023, a nota recalculada é exatamente igual à do CSV, inclusive nos 319 empates
de meio centésimo, todos em 2023. Alternativas rejeitadas:

| Regra | Linhas divergentes | Linhas em que o alvo mudaria |
|---|---|---|
| Sem arredondar | 113.458 | 1.387 (1.228 só por ponto flutuante) |
| Meio para cima, conta inteira (adotada) | 0 | 0 |
| `round(2)` do pandas | 128 | 0 |
| Truncar em 2 casas | 56.967 | 159 |

**Alcance.** A regra foi obtida por reprodução dos dados históricos. **Não é apresentada como regra
oficial atual do SISU:** nenhuma fonte oficial foi consultada. Para citá-la como regra do SISU, ou
aplicá-la a edições futuras com essa garantia, é preciso confirmar no edital ou em norma.

**O alvo não muda:** `NOTA_SUFICIENTE` continua calculado com a `NOTA_CANDIDATO` do CSV (D1).

---

## D13. Contrato entre Modelo e Aplicação

Para as duas frentes trabalharem em paralelo, o Modelo entrega e a Aplicação consome:

| Artefato | Conteúdo |
|---|---|
| `models/pipeline_final.joblib` | Objeto único: pré-processamento + modelo + calibrador. A aplicação chama só `predict_proba` |
| `models/historico_cortes.parquet` | Cortes por curso, modalidade, edição e ano |
| `models/metadados.json` | Lista de atributos, limiar, métricas de teste, versões das bibliotecas, data, último ano do histórico |

**Montagem da entrada.** `NOTA_CANDIDATO`, `TEM_NOTA_ZERO`, `CORTE_ANO_ANTERIOR` e
`DIF_CORTE_ANTERIOR` são calculados por **um único caminho de código** em `src/features.py`, usado no
treino e na aplicação. Assim a aplicação **não reimplementa** o pré-processamento, como exige o
enunciado.

| Função | Uso | Corte consultado |
|---|---|---|
| `montar_entrada_treino(df, historico)` | dados históricos; exige a coluna `ANO` | `ANO − 1` de cada linha |
| `montar_entrada_inferencia(df, historico, ano_alvo)` | aplicação | `ano_alvo − 1` |

- **Entrada** (`features.COLUNAS_ENTRADA`): `CODIGO_CAMPUS`, `NOME_CURSO_NORM`, `GRAU`, `TURNO`,
  `MODALIDADE`, `EDICAO`, `OPCAO`, `PERCENTUAL_BONUS` e as 5 provas. Falta de coluna é erro.
- **Saída:** exatamente `config.ATRIBUTOS`, na ordem da lista, com o índice da entrada. As outras
  colunas são descartadas.
- **Sem histórico:** `CORTE_ANO_ANTERIOR` e `DIF_CORTE_ANTERIOR` ficam NaN (D3).
- **Histórico:** `construir_historico` dá `ValueError` se uma chave tiver mais de um corte no mesmo
  ano.
- **Fica com a aplicação:** validar a faixa das notas e do bônus e oferecer campus, curso, grau,
  turno e modalidade **a partir do histórico**. Um texto escrito de outra forma (por exemplo, o nome
  do curso sem normalização) não gera erro: só resulta em corte ausente.

**Verificado nos dados de 2017–2023** (`reports/conciliacao/verificacao_dados.md`):

- entradas de treino e de inferência iguais nas 1.080.713 linhas, ano a ano, e em 1.000 inscrições
  montadas uma por vez;
- só o corte de N−1 altera as entradas do ano N, também conferido por uma segunda implementação;
- nenhuma coluna excluída pela D2 nas entradas;
- sem histórico, corte e diferença ficam ausentes e os outros 15 atributos ficam preenchidos.

**Ainda não existe, portanto não foi verificado:** `pipeline_final.joblib`, `metadados.json`, a
aplicação e o tratamento dos ausentes pelo Pipeline final.

**Previsão para um ano novo.** A aplicação usa o corte do **ano mais recente** do histórico (2023),
chamando a montagem com `ano_alvo` igual a esse ano mais 1 (2024). Com `ano_alvo` maior, não há
corte de N−1 e os dois atributos ficam ausentes. O desempenho para edições posteriores a 2023 não
foi validado, e a aplicação precisa informar isso.

**Enquanto o modelo final não existir,** a aplicação pode usar um `pipeline_final.joblib` provisório
(por exemplo, uma regressão logística simples) com as mesmas colunas de entrada.

---

## Propostas em aberto (aguardam a equipe)

Nada nesta seção foi adotado: o código, o alvo e as decisões acima continuam como estão.
P1 (arredondamento da nota) e P2 (taxas do alvo atual) foram aplicadas em D12 e em D6–D8 em
27/11/2026. O protocolo de calibração também está em aberto (D8).

### P3. D5: a regra usa `>`, o alvo usa `>=`

**Situação.** A regra publicada prevê 1 quando `NOTA_CANDIDATO > CORTE_ANO_ANTERIOR`
(`src/baselines.py`, `reports/baselines_2022.md`). O commit `68a3e06` propunha `>=`, como no alvo.
A proposta não foi adotada, para o texto não se separar do código e dos resultados publicados.
Entre as linhas com alvo e corte anterior, a nota é igual a esse corte em 4 (2018), 7 (2019),
3 (2020), 9 (2021) e 3 (2022) linhas.

**Se a equipe adotar `>=`:** mudar D5 e o código e executar os baselines de novo, registrando o
resultado ao lado do publicado, sem substituí-lo.

### P4. D9: a divisão por ano não garante pessoas diferentes

**Situação.** A D9 afirma que a divisão por ano evita que a mesma pessoa esteja no treino e no
teste. Os dados não têm identificador, e uma pessoa pode se inscrever em anos diferentes.

**Proposta.** Dizer que a divisão separa os processos seletivos no tempo, mas não garante que as
pessoas sejam distintas, e declarar isso como limitação.

---

## Pendências fora da modelagem

- **Licença dos dados:** confirmar no Portal de Dados Abertos do MEC e atualizar o README.
- **Consolidação do CSV:** documentar como os arquivos do MEC foram baixados e unidos no recorte da
  UFMA 2017–2023 (o artigo precisa descrever a origem dos dados).
