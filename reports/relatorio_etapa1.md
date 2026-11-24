# Relatório da Etapa 1 — Preparação dos dados (sisu-ufma-ml)

Trabalho final de Inteligência Artificial (UFMA, 2026.2). Este relatório cobre as Tarefas 1 a 5:
repositório, limpeza, atributo "corte do ano anterior", divisão temporal e testes mínimos.
**Nenhum modelo foi treinado.**

---

## 1. Contexto do projeto

**Objetivo.** Prever a probabilidade de aprovação no SISU da UFMA a partir das notas do ENEM.
É uma classificação binária; o alvo é `APROVADO`.

**Entregas do trabalho.** (a) modelo de ML com protocolo rigoroso e sem vazamento de dados;
(b) aplicação Streamlit que carrega o pipeline salvo; (c) artigo no formato SBC.

**Dados.** Microdados abertos do SISU/MEC, recorte UFMA 2017–2023, 2 edições por ano.
São 1.080.713 inscrições (candidato × curso) e 31 colunas. Fonte: https://dadosabertos.mec.gov.br/sisu
(licença **a confirmar**).

**Por que o cuidado com vazamento.** Com a `NOTA_CORTE` do próprio ano, a regra "nota ≥ corte"
quase determina o alvo (0 aprovados abaixo do corte). `NOTA_CORTE`, `CLASSIFICACAO` e `MATRICULA`
só existem depois do resultado e não podem ser atributos. O que é legítimo é o **corte do ano
anterior**, que já era público antes da inscrição.

**Por que divisão temporal.** O mesmo candidato aparece em duas linhas (opções 1 e 2) e o objetivo é
prever anos futuros. Uma divisão aleatória misturaria passado e futuro e inflaria a métrica.

| Conjunto | Anos |
|---|---|
| Histórico (só alimenta o corte anterior) | 2017 |
| Treino | 2018–2021 |
| Validação | 2022 |
| Teste | 2023 |

**Regra de pré-processamento.** Tudo o que aprende com os dados (escala, encoders, imputação)
fica no `Pipeline` do sklearn e será ajustado só no treino, na próxima etapa. Esta etapa tem apenas
limpeza determinística.

---

## 2. O que foi feito

### 2.1 Repositório (Tarefa 1)
- Pasta e repositório renomeados para `sisu-ufma-ml`; branch `main`.
- Estrutura criada conforme o enunciado, com `.gitkeep` nas pastas vazias.
- `requirements.txt` com as versões exatas instaladas num `.venv` (pandas 3.1.1, pyarrow 26.0.1,
  numpy 2.5.5, scikit-learn 1.9.1, lightgbm 4.7.0, shap 0.54.1, matplotlib 3.11.2, seaborn 0.13.2,
  streamlit 1.67.0, joblib 1.6.0, pytest 9.2.0, jupyter 1.1.1).
- `.gitignore`: Python, Jupyter, `.venv/`, `data/raw/`, `data/processed/`, `models/*.joblib`, `*.crdownload`.
- `README.md` com problema, fonte, instalação, onde colocar o CSV e seções vazias de Resultados e Vídeo.
- `src/config.py`: semente 42, caminhos, anos de cada divisão, lista de colunas de vazamento,
  chave do histórico de cortes.
- O CSV (359 MB, 1.080.714 linhas com o cabeçalho) está em `data/raw/` e **não** é versionado.
- Um único commit (`ad5395d`), sem linha de co-autoria, conforme o CLAUDE.md pessoal.

> Observação: o caminho do enunciado (`C:\Users\...`) não existe nesta máquina. O CSV foi baixado
> para dentro da pasta do projeto e movido para `data/raw/`.

### 2.2 Limpeza (Tarefa 2) — `src/limpeza.py`
`limpar(df)` é determinística e testável, e aceita um dicionário opcional de log com as contagens.
Executar com `python -m src.limpeza` gera `data/processed/sisu_limpo.parquet`.

| # | Decisão | Linhas afetadas | Justificativa |
|---|---|---|---|
| 1 | `APROVADO` S/N → 0/1 | 39.936 positivos (3,7%) | Alvo binário. |
| 2 | `IDADE = ANO − DT_NASCIMENTO`; fora de [14, 80] vira NaN | 3.067 viram NaN | `DT_NASCIMENTO` é só o ano e tem valores impossíveis (até 2068). A imputação fica para o Pipeline. |
| 3 | Notas zero: **marcar** com `TEM_NOTA_ZERO`, manter a linha | 7 linhas (CN e M); 0 aprovadas | São raríssimas. Remover não muda nada e a marca é mais transparente. |
| 4 | Criar `NOME_CURSO_NORM` (maiúsculas, sem acento/espaços repetidos) | todas | Há nomes com espaço sobrando entre anos (ex.: "CIÊNCIAS BIOLÓGICAS "). |
| 5 | Criar `MODALIDADE` canônica a partir do texto da cota | todas | Os códigos de cota mudaram entre anos (ver 2.3). |
| 6 | Descartar `TP_COTA` | 89% nulos | Quase vazia. |
| 7 | Descartar `QT_VAGAS_CONCORRENCIA` | 79% nulos | Quase vazia. |
| 8 | Descartar `CLASSIFICACAO`, `MATRICULA` | todas | Vazamento. |
| 9 | Descartar `NOME_CAMPUS` | todas | Redundante com `CODIGO_CAMPUS` e `MUNICIPIO_CAMPUS`. |
| 10 | Descartar `DT_NASCIMENTO` | todas | Substituída por `IDADE`. |
| 11 | Descartar `MOD_CONCORRENCIA`, `TIPO_MOD_CONCORRENCIA`, `SUBGRUPO_COTA` | todas | Instáveis entre anos; cobertas por `MODALIDADE`. |
| 12 | Descartar `CODIGO_CURSO`, `NOME_CURSO` | todas | Código instável; o curso é identificado por campus + nome + grau + turno. |
| 13 | `NOTA_CORTE` fica no parquet só para o histórico | – | Fora da lista de atributos (`colunas_atributos`). |

Duplicatas exatas: nenhuma. Nulos remanescentes nas colunas mantidas: nenhum além de `IDADE`.

### 2.3 Investigação da estabilidade da chave (Tarefa 3)
O enunciado pedia para investigar antes de definir a chave. Três achados:

1. **`CODIGO_CURSO` não é confiável.** Em 2018/edição 1 o código usa outra numeração (códigos
   pequenos como 1, 3, 7, 9...), que agrupa vários campi e turnos. Em 2018 há 123 códigos contra 83
   nos anos vizinhos, e só 75 coincidem com 2017/2019. Um mesmo código também cobre mais de um turno
   em outros anos, e o turno de um curso muda entre anos.
2. **`SUBGRUPO_COTA` mudou de significado em 2021.** O mesmo texto "candidatos com renda ≤ 1,5 SM..."
   é `SOCIAL` até 2021 e `D` de 2021 em diante, enquanto `D` em 2017 significava deficiência.
3. **Grupo + subgrupo não identifica o corte.** Com essa chave, 3.088 de 4.744 combinações
   (ano, edição, curso, grupo, subgrupo) tinham mais de um valor de corte, porque o texto da cota
   distingue modalidades diferentes dentro do mesmo subgrupo. Usar `CODIGO_CURSO` + modalidade
   canônica ainda deixava 484 chaves ambíguas.

**Chave adotada:** `CODIGO_CAMPUS + NOME_CURSO_NORM + GRAU + TURNO + MODALIDADE + EDICAO`.
`MODALIDADE` é derivada do texto da cota, combinando etnia (`PP` pretos/pardos, `I` indígenas),
deficiência (`D`) e renda (`RENDA` ≤ 1,5 SM ou `INDEP`), além de `AC` (ampla concorrência),
`BONUS_MA` (bônus Maranhão, 2019–2021) e `AC_MEDICINA_REGIONAL` (2023). Com essa chave há
**0 chaves com mais de um corte** (8.343 chaves).

### 2.4 Atributo "corte do ano anterior" (Tarefa 3) — `src/features.py`
- `construir_historico(df)`: um corte por (curso, modalidade, edição, ano).
- `adicionar_corte_anterior(df, historico)`: para uma inscrição do ano N junta o corte do ano N−1
  (mesma edição), criando `CORTE_ANO_ANTERIOR` e `DIF_CORTE_ANTERIOR = NOTA_CANDIDATO − CORTE_ANO_ANTERIOR`.
  Sem correspondência, o valor fica NaN (a imputação é do Pipeline). A junção usa `validate="m:1"`.
- A tabela de histórico foi salva em `models/historico_cortes.parquet` (52 KB, versionado) e
  `models/historico_cortes.joblib` (ignorado), para a aplicação calcular o atributo a partir do curso escolhido.
- 2017 não tem ano anterior e serve só de histórico.

**Cobertura por ano** (% de linhas com corte anterior encontrado):

| Ano | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|
| Cobertura | – | 94,9% | 73,2% | 75,0% | 94,1% | 98,1% | 91,7% |

Linhas com corte anterior: 2018: 177.008 · 2019: 138.821 · 2020: 117.184 · 2021: 105.687 ·
2022: 90.074 · 2023: 83.676. Treino, validação e teste têm, portanto, cobertura de 73% a 98%.

### 2.5 Divisão temporal (Tarefa 4) — `src/split.py`
`dividir_temporal(df)` devolve (treino 2018–2021, validação 2022, teste 2023) usando só os anos de
`config`. Não há aleatoriedade; 2017 fica de fora.

### 2.6 Testes (Tarefa 5) — `tests/`
11 testes, todos passando (`pytest -q`), sobre dados sintéticos (rodam sem o CSV):
- nenhuma coluna de vazamento (`NOTA_CORTE`, `CLASSIFICACAO`, `MATRICULA`) nem o alvo entre os atributos;
- o corte anterior de uma linha do ano N vem sempre do ano N−1 (e é NaN no primeiro ano);
- `DIF_CORTE_ANTERIOR` correto e a junção não duplica linhas;
- treino, validação e teste não compartilham anos, e a ordem temporal é respeitada;
- limpeza: alvo 0/1, idade fora da faixa vira NaN, notas zero marcadas, colunas descartadas, idempotência
  e a função `modalidade_canonica`;
- um teste extra confere o parquet real, e é pulado se ele não existir.

---

## 3. Distribuição do alvo por ano (após a limpeza)

| Ano | Inscrições | Aprovados | % aprovados |
|---|---|---|---|
| 2017 | 252.687 | 5.654 | 2,2% |
| 2018 | 186.574 | 5.756 | 3,1% |
| 2019 | 189.741 | 5.812 | 3,1% |
| 2020 | 156.340 | 5.748 | 3,7% |
| 2021 | 112.269 | 5.696 | 5,1% |
| 2022 | 91.846 | 5.658 | 6,2% |
| 2023 | 91.256 | 5.612 | 6,1% |
| **Total** | **1.080.713** | **39.936** | **3,7%** |

O número de aprovados é praticamente constante (cerca de 5,7 mil por ano, limitado pelas vagas),
enquanto as inscrições caem a cada ano. Por isso a taxa de aprovação sobe de 2,2% para 6,1%.

---

## 4. Decisões pendentes para a equipe

### D1. Cobertura baixa do corte anterior em 2019 e 2020 (treino)
**Contexto.** Em 2019 surgiu o grupo `BONUS_MA` (bônus para quem estudou no Maranhão), que dividiu a
ampla concorrência em dois grupos. Ele não existia em 2018, então as 38.612 linhas de `BONUS_MA`
de 2019 têm 0% de cobertura. Há ainda um efeito de edição: em 2019/2 só 52% das linhas têm corte
anterior, e em 2020/1, 63%, porque nem todos os cursos/turnos oferecem as mesmas vagas nas duas edições.
Nas modalidades de cota a cobertura de 2019 e 2020 é alta (cerca de 90% e 96%).

**Impacto.** O treino fica com cerca de 25% de `CORTE_ANO_ANTERIOR` ausente em 2019–2020, concentrado
em ampla concorrência. O modelo pode aprender "corte ausente" como sinal de grupo e não de dificuldade.

**Opções.**
- (a) Manter NaN e usar um indicador de ausência no Pipeline (mais simples e honesto).
- (b) Para `BONUS_MA`, usar o corte de `AC` do ano anterior como substituto (semanticamente diferente,
  pois o corte do bônus inclui o acréscimo na nota).
- (c) Usar o corte do mesmo curso na outra edição do ano anterior quando a edição própria não existir.
- (d) Treinar só em 2021 em diante, onde a cobertura passa de 94% (perde dados).

**Recomendação.** (a) como base e avaliar (b)/(c) como experimento na validação. Precisa de decisão
antes de montar o Pipeline.

### D2. Mudança de taxa de aprovação entre treino e teste
**Contexto.** A taxa vai de 3,1% (2018) a 6,2% (2022–2023). Treinando em 2018–2021 (média de 3,6%),
o modelo tende a subestimar probabilidades em 2022–2023.

**Impacto.** A calibração e a métrica de limiar podem ficar ruins mesmo com bom ranking (AUC).

**Opções.** Reportar métricas independentes de limiar (AUC-PR, AUC-ROC) e a calibração; recalibrar
na validação (2022); ponderar anos recentes. Precisa de decisão sobre quais métricas são as principais
no artigo e se haverá recalibração.

### D3. Candidato repetido e ausência de identificador
**Contexto.** O mesmo candidato pode aparecer nas opções 1 e 2, e o dado não tem ID nem CPF. Não é
possível agrupar por pessoa.

**Impacto.** Dentro de um mesmo ano o vazamento entre linhas do mesmo candidato é limitado, porque a
divisão é por ano. Mas a independência das linhas não vale, e os intervalos de confiança ficam otimistas.

**Opção.** Documentar como limitação no artigo; se quiserem, tratar `OPCAO` como atributo.

### D4. Alvo mistura "ser aprovado" e "não ser chamado"
**Contexto.** `APROVADO = S` indica aprovação no resultado do SISU. Pela descrição do dado, não sabemos se distingue quem foi aprovado em uma
opção e desistiu, nem se um candidato aprovado na opção 1 deixa de ser convocado na opção 2.

**Impacto.** A interpretação do resultado ("probabilidade de aprovação") é por inscrição, não por pessoa.

**Decisão.** Confirmar que a unidade de previsão no app é "inscrição em um curso/modalidade".

### D5. Atributos de modalidade em 2023 (Lei de Cotas e Medicina regional)
**Contexto.** Em 2023 a Lei de Cotas mudou e a UFMA criou vagas de ampla concorrência regionais de
Medicina (`AC_MEDICINA_REGIONAL`, 5.105 linhas), sem histórico anterior. Cinco textos diferentes
foram agrupados nesta mesma modalidade.

**Impacto.** São linhas só do teste, sem corte anterior. A cobertura de 2023 (91,7%) já considera isso.

**Opção.** Manter assim (e declarar a limitação) ou tratar como `AC` de Medicina para o fallback.

### D6. Campi com código novo
**Contexto.** Há campi que mudaram de código (Balsas: 1062821 até 2022, 1138998 em 2023) e o curso
de Turismo trocou de campus em 2020 (1006314 → 1107952, "Complexo Santa Amélia"). Esses casos perdem
o corte anterior por usar `CODIGO_CAMPUS` na chave.

**Impacto.** Pequeno (poucas linhas), mas é uma causa da cobertura abaixo de 100%.

**Opção.** Usar `MUNICIPIO_CAMPUS` em vez do código do campus na chave, o que recupera Balsas, mas
une campi diferentes do mesmo município (São Luís tem dois, Imperatriz tem dois). Recomendo manter o código
do campus e aceitar a perda.

### D7. Licença dos dados
Continua **a confirmar** no README. Precisa ser verificada no portal de dados abertos do MEC antes de
publicar o artigo ou o repositório.

### D8. Versionamento do histórico de cortes
`models/historico_cortes.parquet` (52 KB) está no git para a aplicação funcionar após clonar. O
`.joblib` é ignorado. Confirmar se o grupo prefere versionar só o parquet (como está) ou gerar tudo
pelo script.

---

## 5. Como reproduzir

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# colocar o CSV em data/raw/sisu_ufma_2017_2023.csv
python -m src.limpeza     # data/processed/sisu_limpo.parquet
python -m src.features    # models/historico_cortes.* e cobertura por ano
pytest -q                 # 11 testes
```

## 6. Próximos passos sugeridos
1. Resolver D1 e D2 (afetam o Pipeline e as métricas).
2. Montar o `Pipeline` do sklearn (imputação, escala, codificação) ajustado só no treino.
3. Treinar baselines (regressão logística e LightGBM), com validação em 2022 e teste único em 2023.
4. Calibração, SHAP e figuras para o artigo.
5. Aplicação Streamlit carregando o pipeline e `historico_cortes`.
