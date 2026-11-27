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
| D4 | Divisão | Temporal: treino 2018–2021, validação 2022 (A/B/C por grupo), teste 2023 | Implementada |
| D5 | Modelos | Regra (`>=`) + regressão logística, LightGBM, MLP; busca registrada | Protocolo registrado |
| D6 | Desbalanceamento | Peso de classe, comparado na validação | Comparado na regressão logística |
| D7 | Métricas | Seleção por PR-AUC; Brier para calibração | Adotada |
| D8 | Calibração e limiar | Calibrador ajustado em B; escolha e limiar em C | Protocolo registrado |
| D9 | Candidato repetido | Limitação declarada no artigo, inclusive entre anos | Adotada |
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
| Validação | 2022 | Dividida em A, B e C (abaixo) |
| Teste | 2023 | **Avaliado uma única vez**, no final |

Nunca usar divisão aleatória **entre anos**: o mesmo candidato aparece em duas linhas (opções 1 e 2)
e o objetivo é prever anos futuros. Implementado em `src/split.py`.

**Divisão interna de 2022** (registrada em 27/11/2026, antes dos experimentos). As inscrições
rotuladas de 2022 são distribuídas em cinco partições com
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, estratificando por
`NOTA_SUFICIENTE` e agrupando por `config.CHAVE_HISTORICO` (campus, curso, grau, turno, modalidade e
edição). Todas as inscrições de uma mesma oferta e modalidade ficam, portanto, na mesma parte.

| Parte | Partições | Uso |
|---|---|---|
| A | 0, 1 e 2 | Seleção de modelos e hiperparâmetros |
| B | 3 | Ajuste dos calibradores |
| C | 4 | Comparação da calibração e escolha do limiar |

- As cinco partições só servem para construir A, B e C. **Não são cinco rodadas de validação
  cruzada.** As proporções (cerca de 60%, 20% e 20%) são aproximadas, porque os grupos têm tamanhos
  diferentes.
- O mapeamento de grupos para partições fica salvo em `reports/experimentos/particoes_2022.csv`.
- Para cada parte se registram grupos, linhas, positivos e negativos. Confere-se que não há grupo
  nem linha em mais de uma parte e que as três têm as duas classes.
- C serve para **escolher** a calibração e o limiar. Não é teste final: o teste continua sendo 2023.
- Implementado em `src/particoes.py`.

---

## D5. Modelos

O enunciado exige um baseline e **pelo menos dois modelos de famílias diferentes**.
Random Forest e LightGBM são da **mesma família** (ensemble de árvores), por isso não basta usar os dois.

| Papel | Modelo | Família |
|---|---|---|
| Baseline 1 | Regra: suficiente se `NOTA_CANDIDATO >= CORTE_ANO_ANTERIOR` (sem corte anterior: prever 0) | Regra |
| Baseline 2 | Regressão logística | Linear |
| Modelo 1 | LightGBM | Ensemble de árvores |
| Modelo 2 | Rede neural MLP (`MLPClassifier` com early stopping) | Rede neural |

SVM não é viável com as 644.845 linhas rotuladas do treino. Se a MLP for lenta, pode treinar em
subamostra estratificada do treino, com semente fixa e documentada (regra abaixo).

A regra do Baseline 1 é o adversário principal: o modelo de ML precisa superá-la. Se não superar,
isso também é um resultado e deve ser discutido com honestidade no artigo.

**Regra com `>=` (adotada em 27/11/2026, resolve a P3).** A regra passa a usar `>=`, como o alvo
(D1). Os resultados publicados em `reports/baselines_2022.md` foram obtidos com `>` no commit
`179593a`, sobre toda a validação de 2022, e ficam como registro histórico: não são comparáveis aos
números em A.

### Protocolo de comparação (registrado antes das execuções)

**Dados.** Todos os modelos são ajustados nas inscrições rotuladas de 2018–2021 e avaliados nas
**mesmas** linhas de A (D4). As entradas vêm de `montar_entrada_treino`, com o histórico de cortes
de 2017–2021. O pré-processamento aprendido (imputação pela mediana com indicador, padronização e
one-hot, como nos baselines) faz parte de cada `Pipeline` e é ajustado só nas linhas de treino
daquele modelo.

**Métricas em A.** Seleção por *average precision* (AP, D7). Também se registram ROC-AUC e Brier
(sem recalibração), e precisão, revocação e F1 com limiar 0,5, só como informação.

**Regra histórica.**
- Métricas de **decisão**: precisão, revocação e F1 da saída 0/1.
- Métricas de **ranking** (AP e ROC-AUC): escore contínuo `DIF_CORTE_ANTERIOR`. Sem corte anterior,
  o escore é −10.000, abaixo de qualquer diferença possível (as notas ficam entre 0 e cerca de
  1.050). Esses casos ficam empatados no fim da ordenação.
- A AP da saída 0/1 também é registrada, para comparação com o relatório publicado.

**Busca de hiperparâmetros** (grade completa, semente 42 em tudo):

| Modelo | Fixos | Grade | Configurações |
|---|---|---|---|
| Regressão logística | `solver="liblinear"`, `max_iter=1000` | `C` ∈ {0,1; 1; 10} × `class_weight` ∈ {nenhum, `"balanced"`} | 6 |
| LightGBM | `n_estimators=400`, `learning_rate=0,05` | `num_leaves` ∈ {15, 63} × `min_child_samples` ∈ {20, 200} × `scale_pos_weight` ∈ {1, negativos/positivos do treino} | 8 |
| MLP | `adam`, `learning_rate_init=0,001`, `batch_size=512`, `max_iter=50`, `early_stopping=True`, `validation_fraction=0,1`, `n_iter_no_change=5` | `hidden_layer_sizes` ∈ {(64,), (128, 64)} × `alpha` ∈ {0,0001; 0,001} × peso de classe ∈ {nenhum, balanceado} | 8 |

Na MLP, o peso balanceado é passado como `sample_weight` (n / (2 · contagem da classe)), porque o
`MLPClassifier` não tem `class_weight`. O early stopping separa 10% do treino da própria MLP; essa
parte não vem de 2022.

**Ensaio de custo da MLP.** Antes da busca, ajusta-se a maior configuração ((128, 64), sem peso) numa
subamostra estratificada de 50.000 linhas do treino e mede-se o tempo. A projeção para o treino
completo é esse tempo × 644.845 / 50.000 por configuração. Se a projeção das 8 configurações passar
de 40 minutos, toda a busca da MLP usa uma subamostra estratificada de **200.000** linhas do treino
(`StratifiedShuffleSplit`, semente 42), a mesma para as 8 configurações. O pré-processamento da MLP é
ajustado nessa subamostra.

**Seleção.** Escolhe-se a configuração aprendida (logística, LightGBM ou MLP) com maior AP em A. Em
empate até a quarta casa, vence a maior ROC-AUC. A regra não é candidata: é a referência.

Também se registram o tempo de ajuste, o tempo de predição em A e os avisos de convergência de cada
configuração. Todos os resultados são reportados, inclusive os inferiores à regra.

Implementado em `src/modelos.py` e `src/experimentos.py`.

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

**Decisão** (protocolo adotado em 27/11/2026, antes dos experimentos; usa as partes A, B e C de
2022 da D4).

1. O modelo escolhido em A fica **congelado**: classificador e pré-processamento não são reajustados.
2. Sigmoid (Platt) e isotônica são ajustadas **só em B**, com
   `CalibratedClassifierCV(FrozenEstimator(modelo), method=...)`. Com o estimador congelado, o
   sklearn ajusta um único calibrador sobre todo o B, na escala de `decision_function` (conferido
   antes dos experimentos).
3. Em C se comparam três opções: sem recalibração, sigmoid e isotônica. Vence o **menor Brier em C**,
   acompanhado da curva de calibração das três.
4. O limiar é o que **maximiza o F1 em C** para a opção escolhida.
5. C foi usada para escolher, não para estimar o desempenho final. Depois dessas escolhas, nada é
   reajustado sem revisar este protocolo. A avaliação em 2023 fica pendente.

A aplicação mostra a **probabilidade calibrada**, por isso a calibração é parte central do trabalho,
e não um detalhe.

Implementado em `src/experimentos.py`.

---

## D9. Candidato repetido

O mesmo candidato pode aparecer nas opções 1 e 2, e o dado não tem identificador pessoal. A divisão
por ano separa os processos seletivos no tempo, mas **não comprova que as pessoas sejam diferentes**
entre treino, validação e teste: quem se inscreveu em mais de um ano aparece em mais de um conjunto.
Dentro de um ano, as linhas também não são independentes; na divisão interna de 2022 (D4), as duas
opções de uma pessoa podem cair em partes diferentes. Declarar as duas coisas como limitação.
`OPCAO` entra como atributo. (Corrigido em 27/11/2026; resolve a P4.)

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

## Propostas resolvidas

- **P1** (arredondamento da nota) e **P2** (taxas do alvo atual): aplicadas em D12 e D6–D8.
- **P3** (regra com `>=`): adotada na D5 em 27/11/2026. Na validação de 2022 a nota é igual ao corte
  anterior em 3 linhas rotuladas. A regra com `>=` é avaliada em A, junto com os modelos; os
  baselines sobre toda a validação de 2022 não foram executados de novo, para não usar B e C fora do
  protocolo da D8.
- **P4** (D9 e pessoas repetidas entre anos): texto corrigido na D9.
- **Protocolo de calibração:** definido na D8.

---

## Pendências fora da modelagem

- **Licença dos dados:** confirmar no Portal de Dados Abertos do MEC e atualizar o README.
- **Consolidação do CSV:** documentar como os arquivos do MEC foram baixados e unidos no recorte da
  UFMA 2017–2023 (o artigo precisa descrever a origem dos dados).
