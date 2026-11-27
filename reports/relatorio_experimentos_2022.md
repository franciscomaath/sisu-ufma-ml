# Relatório dos experimentos na validação de 2022

Data: 27/11/2026. Branch `experimentos/validacao-2022`, a partir de `2c72266`.

**Resumo.** O protocolo foi registrado no `docs/DECISOES.md` (commit `90c79f9`) antes das
execuções. Na parte A de 2022, a **regressão logística com `C=10` e sem peso de classe** teve a
maior *average precision* (0,7158), à frente da MLP (0,6521), do LightGBM (0,5977) e da regra do
corte anterior (0,5171). A calibração **isotônica**, ajustada em B, teve o menor Brier em C, e o
limiar de maior F1 em C é **0,3086**.

**Nenhuma métrica de 2023 foi calculada.** Os scripts não leem o teste.

As partes A e C serviram para **escolher**; seus números não são estimativas do desempenho final.

## 1. Divisão obtida

`python -m src.particoes`: inscrições rotuladas de 2022, `StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=42)`, estratificado pelo alvo e agrupado por `config.CHAVE_HISTORICO`.
Foram 91.750 linhas em 1.023 grupos.

| Partição | Parte | Grupos | Linhas | Positivos | Negativos |
|---|---|---|---|---|---|
| 0 | A | 196 | 18.307 | 1.432 | 16.875 |
| 1 | A | 206 | 18.369 | 1.429 | 16.940 |
| 2 | A | 206 | 18.380 | 1.434 | 16.946 |
| 3 | B | 204 | 18.346 | 1.430 | 16.916 |
| 4 | C | 211 | 18.348 | 1.429 | 16.919 |

| Parte | Uso | Grupos | Linhas | Positivos | Negativos | Taxa | Fração das linhas |
|---|---|---|---|---|---|---|---|
| A | Seleção | 608 | 55.056 | 4.295 | 50.761 | 7,80% | 60,0% |
| B | Ajuste do calibrador | 204 | 18.346 | 1.430 | 16.916 | 7,79% | 20,0% |
| C | Escolha da calibração e do limiar | 211 | 18.348 | 1.429 | 16.919 | 7,79% | 20,0% |

- **Conferências** (`verificar_partes`): nenhum grupo em mais de uma parte, nenhuma linha sem parte
  e as duas classes presentes em A, B e C.
- **Mapa:** `reports/experimentos/particoes_2022.csv`, SHA-256
  `c465dd2c2add6644f9f2103642f2c59c02feb6160c2d764d2e04393bb83cc96c`. Os experimentos refazem a
  divisão e param se ela diferir do mapa salvo.
- **Limitação:** as duas opções de uma mesma pessoa podem cair em partes diferentes (D9).

## 2. Exploração (só o treino, 2018–2021)

`python -m src.exploracao` gera três figuras em `reports/figures/` e tabelas
`reports/experimentos/exploracao_*.csv`.

| Figura | Conteúdo |
|---|---|
| `alvo_treino_por_ano.png` | Inscrições com alvo e taxa de positivos por ano |
| `ausencia_corte_anterior_treino.png` | % sem corte anterior por ano e grupo de modalidade |
| `taxa_por_diferenca_corte_treino.png` | Taxa de nota suficiente por faixa de nota − corte anterior |

**Achados**

- **Alvo:** a taxa sobe de 3,76% (2018) para 6,42% (2021), enquanto as inscrições com alvo caem de
  186.564 para 112.221.
- **Ausência do corte anterior:** 106.211 linhas de treino, concentradas no bônus Maranhão: 100% na
  edição 2 de 2019 (o grupo não existia em 2018) e 100% na edição 1 de 2020. Nos outros grupos, de
  0% a 9,3% por ano e edição. Nessas linhas a taxa de positivos é 5,17%.
- **Diferença até o corte anterior:** abaixo de −50 pontos a taxa fica abaixo de 1,3%. Ela passa de
  17,04% (−20 a −10) para 61,69% (0 a 10) e se estabiliza entre 89% e 98% acima de +20. Ficar acima
  do corte do ano anterior não garante a vaga.

**Extremos**

| Atributo | Mínimo | Máximo | Observação |
|---|---|---|---|
| `NOTA_CN`, `NOTA_M` | 0 | 831,0 e 989,0 | 7 linhas com nota zero (`TEM_NOTA_ZERO`) |
| `NOTA_R` | 40 | 1.000 | – |
| `NOTA_CANDIDATO` | 217,54 | 1.021,42 | Passa de 1.000 por causa do bônus |
| `DIF_CORTE_ANTERIOR` | −570,75 | 202,61 | O escore −10.000 dos casos sem histórico fica abaixo de todos |

**Limitações da exploração:** usa só o treino, e as faixas das pontas agregam os extremos; a faixa
inicial (−150 ou menos) tem 219.070 linhas.

## 3. Comparação em A

**Configuração.** A grade e os fixos estão na D5. Todos os modelos foram ajustados nas 644.845
linhas rotuladas de 2018–2021 e avaliados nas mesmas 55.056 linhas de A. O pré-processamento faz
parte de cada `Pipeline`, ajustado no treino do próprio modelo, e a semente é 42.

**Ensaio de custo da MLP** (`reports/experimentos/ensaio_mlp.json`). A configuração (128, 64) em
50.000 linhas levou 5,0 s. A projeção para as 8 configurações no treino completo foi de 8,6 minutos,
abaixo do limite de 40. Pela regra registrada, **a MLP usou o treino completo**, sem subamostra.

### Regra histórica (`NOTA_CANDIDATO >= CORTE_ANO_ANTERIOR`)

Em A, 1.252 linhas não têm corte anterior e recebem 0 na decisão e −10.000 no escore.

| Tipo | Métrica | Valor |
|---|---|---|
| Ranking, escore `DIF_CORTE_ANTERIOR` | AP | 0,5171 |
| Ranking, escore `DIF_CORTE_ANTERIOR` | ROC-AUC | 0,8444 |
| Decisão 0/1 | Precisão | 0,8857 |
| Decisão 0/1 | Revocação | 0,2796 |
| Decisão 0/1 | F1 | 0,4251 |
| Decisão 0/1 | AP da saída binária | 0,3039 |

### Modelos aprendidos

A tabela tem as 22 configurações, ordenadas como foram executadas. Precisão, revocação e F1 usam o
limiar 0,5 e são só informativos.

| Modelo | Configuração | AP | ROC-AUC | Brier | Precisão | Revocação | F1 | Ajuste (s) |
|---|---|---|---|---|---|---|---|---|
| Logística | C=0,1, sem peso | 0,6777 | 0,9509 | 0,0517 | 0,9340 | 0,2536 | 0,3988 | 7,6 |
| Logística | C=0,1, balanceado | 0,6793 | 0,9540 | 0,0456 | 0,6635 | 0,5485 | 0,6006 | 7,6 |
| Logística | C=1, sem peso | 0,7106 | 0,9594 | 0,0512 | 0,9324 | 0,2570 | 0,4030 | 8,9 |
| Logística | C=1, balanceado | 0,6953 | 0,9585 | 0,0439 | 0,6799 | 0,5569 | 0,6123 | 10,4 |
| **Logística** | **C=10, sem peso** | **0,7158** | **0,9605** | 0,0511 | 0,9338 | 0,2594 | 0,4060 | 8,9 |
| Logística | C=10, balanceado | 0,6974 | 0,9591 | 0,0437 | 0,6804 | 0,5597 | 0,6142 | 11,5 |
| LightGBM | 15 folhas, mín. 20, sem peso | 0,5963 | 0,9198 | 0,0522 | 0,8486 | 0,3106 | 0,4547 | 5,3 |
| LightGBM | 15 folhas, mín. 20, balanceado | 0,5669 | 0,8944 | 0,0577 | 0,5776 | 0,4705 | 0,5186 | 6,3 |
| LightGBM | 15 folhas, mín. 200, sem peso | 0,5977 | 0,9159 | 0,0521 | 0,8406 | 0,3094 | 0,4523 | 5,1 |
| LightGBM | 15 folhas, mín. 200, balanceado | 0,5655 | 0,8939 | 0,0578 | 0,5796 | 0,4747 | 0,5220 | 5,3 |
| LightGBM | 63 folhas, mín. 20, sem peso | 0,5641 | 0,9015 | 0,0528 | 0,8132 | 0,3295 | 0,4689 | 6,3 |
| LightGBM | 63 folhas, mín. 20, balanceado | 0,5575 | 0,8926 | 0,0581 | 0,5995 | 0,4510 | 0,5147 | 6,5 |
| LightGBM | 63 folhas, mín. 200, sem peso | 0,5600 | 0,8977 | 0,0529 | 0,8121 | 0,3290 | 0,4683 | 6,7 |
| LightGBM | 63 folhas, mín. 200, balanceado | 0,5475 | 0,8745 | 0,0584 | 0,5977 | 0,4515 | 0,5144 | 6,8 |
| MLP | (64,), α=0,0001, sem peso | 0,6079 | 0,9222 | 0,0517 | 0,8379 | 0,3285 | 0,4720 | 41,3 |
| MLP | (64,), α=0,0001, balanceado | 0,6521 | 0,9363 | 0,0491 | 0,6852 | 0,4983 | 0,5770 | 22,1 |
| MLP | (64,), α=0,001, sem peso | 0,5959 | 0,9191 | 0,0528 | 0,8128 | 0,3204 | 0,4596 | 50,2 |
| MLP | (64,), α=0,001, balanceado | 0,6481 | 0,9350 | 0,0488 | 0,6931 | 0,4901 | 0,5742 | 25,3 |
| MLP | (128, 64), α=0,0001, sem peso | 0,5552 | 0,9054 | 0,0553 | 0,7865 | 0,3087 | 0,4434 | 114,5 |
| MLP | (128, 64), α=0,0001, balanceado | 0,6075 | 0,9232 | 0,0562 | 0,6299 | 0,4810 | 0,5455 | 81,8 |
| MLP | (128, 64), α=0,001, sem peso | 0,5654 | 0,9100 | 0,0545 | 0,7656 | 0,3278 | 0,4591 | 99,0 |
| MLP | (128, 64), α=0,001, balanceado | 0,6259 | 0,9250 | 0,0539 | 0,6350 | 0,5050 | 0,5626 | 57,9 |

**Leitura**

- **Melhor por família (AP):** logística 0,7158, MLP 0,6521, LightGBM 0,5977. Todas as
  configurações aprendidas superaram a regra em AP e ROC-AUC.
- **Peso de classe:** piorou a AP da logística e do LightGBM e melhorou a da MLP. Em todas as
  famílias, aumentou a revocação com limiar 0,5.
- **LightGBM:** abaixo da logística em todas as configurações; as de 63 folhas foram piores que as de
  15. A hipótese de dificuldade de generalizar para um ano com outra taxa de positivos **não foi
  testada**.
- **Convergência:** nenhum aviso em nenhuma configuração. A MLP parou por early stopping entre 16 e 40
  épocas, antes do máximo de 50; a logística convergiu em 11 a 13 iterações.
- **Tempo:** 595 s de ajuste no total (logística 55 s, LightGBM 48 s, MLP 492 s); predição em A de
  0,07 a 0,38 s por configuração.

**Seleção** (`reports/experimentos/selecao_A.json`): regressão logística, `C=10`, sem peso de classe,
pelo critério registrado (maior AP em A). Não houve empate na quarta casa.

**Limitação:** `C=10` está na borda da grade. A grade não foi ampliada depois de ver os resultados,
para não ajustar o protocolo aos dados de A.

## 4. Calibração e limiar

O modelo escolhido foi congelado. Os dois calibradores foram ajustados só em B, com
`CalibratedClassifierCV(FrozenEstimator(...))`; com o estimador congelado, o sklearn ajusta um
único calibrador sobre todo o B. A comparação foi feita em C.

| Opção | Brier | Log loss | AP | ROC-AUC | Média prevista | Taxa observada |
|---|---|---|---|---|---|---|
| Sem recalibração | 0,0525 | 0,2644 | 0,7115 | 0,9646 | 2,55% | 7,79% |
| Sigmoid | 0,0385 | **0,1225** | **0,7115** | **0,9646** | 6,92% | 7,79% |
| **Isotônica** | **0,0382** | 0,1349 | 0,6909 | 0,9639 | 6,93% | 7,79% |

- **Viés sem recalibração:** a probabilidade média prevista (2,55%) fica muito abaixo da taxa
  observada (7,79%). Esse viés é maior do que a diferença entre as taxas do treino (4,50%) e de
  2022 explicaria. As duas calibrações o corrigem em boa parte.
- **Escolha:** pela regra da D8 (menor Brier em C), venceu a **isotônica**, por 0,00022. Ela tem log
  loss e AP piores que a sigmoid, porque cria empates nas probabilidades. A escolha foi mantida, como
  manda o protocolo; mudá-la exige revisar a D8.
- **Curva de calibração:** `reports/figures/calibracao_C.png`, com 10 faixas de largura 0,1. As
  contagens por faixa estão em `curva_calibracao_C.csv`; a primeira faixa concentra cerca de 82% a
  96% das linhas, conforme a opção.

**Limiar** (maior F1 em C, com a isotônica; `limiar_C.csv`)

| Limiar | F1 | Precisão | Revocação | VP | FP | FN | VN |
|---|---|---|---|---|---|---|---|
| 0,3086 | 0,6156 | 0,6678 | 0,5710 | 816 | 406 | 613 | 16.513 |

C foi usada para essas escolhas e não é teste. Nada foi reajustado depois delas.

## 5. Entrega

- **`models/pipeline_final.joblib`** (10.774 bytes, não versionado): o objeto `CalibratedClassifierCV`
  com a isotônica e o Pipeline congelado. A aplicação chama `predict_proba` sobre
  `montar_entrada_inferencia(...)`.
- **Recarga:** salvar e recarregar preserva as probabilidades em C, com igualdade exata
  (`np.array_equal`).
- **`models/metadados.json`:** atributos, colunas de entrada, modelo, calibração, limiar, divisão
  (com SHA-256), métricas de A e C com o aviso de que serviram para escolher, versões, commit e
  `metricas_teste.status = "pendente"`.
- **`models/modelo_selecionado_A.joblib`:** intermediário, não versionado.

**Versões do código.** A busca rodou no commit `1e0135f`. A calibração foi repetida no `1f58bda`,
que só mudou a figura de calibração, e produziu tabelas idênticas às da primeira execução (conferido
com `cmp`). Os metadados apontam para `1f58bda`.

## 6. Comandos, tempos e testes

```bash
python -m src.particoes
python -m src.exploracao
python -m src.experimentos --etapa ensaio       # 5,0 s de ajuste
python -m src.experimentos --etapa busca        # 595 s de ajuste
python -m src.experimentos --etapa calibracao   # cerca de 21 s no total
python -m pytest -q
```

Testes: **120 passaram** com o parquet real (eram 92 na `main`). Os novos testes usam dados
sintéticos, e um deles executa as três etapas de ponta a ponta numa grade reduzida.

## 7. Resultados anteriores preservados

`reports/baselines_2022.md` e `.csv` (commit `179593a`) usaram a regra com `>` e toda a validação de
2022, e continuam como registro histórico. Não são comparáveis aos números em A. Os baselines sobre
toda a validação não foram executados de novo, para não usar B e C fora do protocolo.

## 8. Limitações e pendências

1. **Avaliação em 2023 pendente:** deve ser feita uma única vez, com o objeto salvo e o limiar
   0,3086, sem nenhum ajuste.
2. **Margem da calibração:** a escolha da isotônica dependeu de 0,00022 de Brier. A equipe pode
   querer discutir isso, mas qualquer mudança exige revisar a D8.
3. **Grade:** `C=10` na borda da grade da logística.
4. **Pessoas repetidas:** as duas opções de uma pessoa podem estar em partes diferentes de 2022, e a
   mesma pessoa pode aparecer em anos diferentes (D9).
5. **Interpretação** (enunciado, item 7): importância de atributos ou SHAP e análise de erros ainda
   não foram feitas.
6. **Variante com `SEXO` e `IDADE`** (D2) e o **fallback de corte da outra edição** (D3, opcional)
   não foram executados.
7. **Aplicação:** usará o histórico até 2023, ano-alvo 2024 (D13). O desempenho para edições
   futuras não foi validado.
8. **Pipeline final fora do git:** o `pipeline_final.joblib` tem só 11 KB. Hoje é ignorado pelo
   `.gitignore`; a equipe pode decidir versioná-lo para a aplicação funcionar logo após o clone.
