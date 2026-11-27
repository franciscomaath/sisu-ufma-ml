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
| D2 | Atributos | Lista explícita (allowlist) em `config.py` | Implementada |
| D3 | Corte anterior ausente | NaN + indicador de ausência no Pipeline | Adotada |
| D4 | Divisão | Temporal: treino 2018–2021, validação 2022, teste 2023 | Implementada |
| D5 | Modelos | Regra + regressão logística (baselines), LightGBM, MLP | Baselines avaliados em 2022 |
| D6 | Desbalanceamento | Peso de classe, comparado na validação | Comparado na regressão logística |
| D7 | Métricas | Seleção por PR-AUC; Brier para calibração | Adotada |
| D8 | Calibração e limiar | Ajustados em 2022 | Adotada |
| D9 | Candidato repetido | Limitação declarada no artigo | Adotada |
| D10 | Casos sem histórico | Aceitar a perda e declarar | Adotada |
| D11 | Histórico de cortes | Parquet versionado no git | Implementada |
| D12 | Nota do candidato | A aplicação calcula a partir das 5 notas | Adotada |
| D13 | Contrato Modelo ↔ Aplicação | Ver seção própria | Adotada |

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
`APROVADO` está em `COLUNAS_NAO_ATRIBUTO`.

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

**Implementação.** `colunas_atributos()` verifica a presença das colunas obrigatórias e retorna os atributos na ordem de `config.ATRIBUTOS`. Os testes conferem a lista e a exclusão de colunas indevidas.

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

---

## D6. Desbalanceamento

A classe positiva vai de 3% a 6% conforme o ano. Comparar, na validação, o modelo com e sem peso
de classe (`class_weight="balanced"` / `scale_pos_weight`). Não usar SMOTE nem reamostragem fora
do Pipeline.

---

## D7. Métricas

- **Seleção do modelo:** PR-AUC na validação (mais informativa que ROC-AUC com classe rara).
- **Reportar no teste, para todos os modelos:** F1, precisão, revocação, ROC-AUC, PR-AUC, Brier e
  matriz de confusão. Destacar a melhor de cada coluna na tabela.
- **Não usar acurácia como métrica principal:** prever "não" para todos já dá ~94%.

---

## D8. Calibração e limiar

**Contexto.** A taxa de positivos sobe ao longo dos anos (3,1% em 2018, 6,2% em 2022, 6,1% em 2023)
porque o número de aprovados é quase fixo (~5,7 mil/ano) e as inscrições caem. Treinando em
2018–2021, o modelo tende a **subestimar** as probabilidades em 2022–2023.

**Decisão.**
- Recalibrar o modelo escolhido **em 2022**, cuja taxa é parecida com a de 2023. Comparar Platt
  (sigmoid) e isotônica pelo Brier score.
- Escolher o limiar de decisão em 2022 (maximizando F1) e congelá-lo antes do teste.
- Mostrar a curva de calibração antes e depois no artigo.

A aplicação mostra a **probabilidade calibrada**, por isso a calibração é parte central do trabalho,
e não um detalhe.

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

Verificado em 100% das linhas:

```
NOTA_CANDIDATO = média simples das 5 notas × (1 + PERCENTUAL_BONUS / 100)
```

Não há pesos por curso. A aplicação pede as 5 notas e se o candidato tem bônus, e calcula
`NOTA_CANDIDATO` sozinha. Adicionar um teste que confirme a fórmula no parquet real
(pulado se o parquet não existir).

---

## D13. Contrato entre Modelo e Aplicação

Para as duas frentes trabalharem em paralelo, o Modelo entrega e a Aplicação consome:

| Artefato | Conteúdo |
|---|---|
| `models/pipeline_final.joblib` | Objeto único: pré-processamento + modelo + calibrador. A aplicação chama só `predict_proba` |
| `models/historico_cortes.parquet` | Cortes por curso, modalidade, edição e ano |
| `models/metadados.json` | Lista de atributos, limiar, métricas de teste, versões das bibliotecas, data, último ano do histórico |

**Montagem da entrada.** O cálculo de `NOTA_CANDIDATO`, `CORTE_ANO_ANTERIOR` e `DIF_CORTE_ANTERIOR`
fica em **uma única função** em `src/features.py` (por exemplo, `montar_entrada(...)`), usada tanto
no treino quanto na aplicação. Assim a aplicação **não reimplementa** o pré-processamento, como exige
o enunciado.

**Previsão para um ano novo.** A aplicação usa o corte do **ano mais recente** do histórico (2023)
como "corte do ano anterior" para a próxima edição.

**Enquanto o modelo final não existir,** a aplicação pode usar um `pipeline_final.joblib` provisório
(por exemplo, uma regressão logística simples) com as mesmas colunas de entrada.

---

## Pendências fora da modelagem

- **Licença dos dados:** confirmar no Portal de Dados Abertos do MEC e atualizar o README.
- **Consolidação do CSV:** documentar como os arquivos do MEC foram baixados e unidos no recorte da
  UFMA 2017–2023 (o artigo precisa descrever a origem dos dados).
