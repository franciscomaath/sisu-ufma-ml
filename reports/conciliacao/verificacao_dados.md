# Verificação da preparação dos dados

Arquivo gerado por `python -m src.verificar_dados`. Não editar à mão.
Não treina modelos nem calcula métricas de desempenho.
Nas tabelas de comparação, os números contam linhas diferentes (0 = iguais).

Linhas no parquet limpo: 1080713.
Histórico versionado idêntico ao reconstruído: sim (8343 linhas).
Entradas montadas: 17 colunas, iguais à lista da D2 e na mesma ordem.

## Alvo por ano

| ANO | conjunto | inscricoes | rotuladas | sem_alvo | suficientes | taxa_% |
|---|---|---|---|---|---|---|
| 2017 | Histórico | 252687 | 252606 | 81 | 6923 | 2.74 |
| 2018 | Treino | 186574 | 186564 | 10 | 7012 | 3.76 |
| 2019 | Treino | 189741 | 189735 | 6 | 7496 | 3.95 |
| 2020 | Treino | 156340 | 156325 | 15 | 7325 | 4.69 |
| 2021 | Treino | 112269 | 112221 | 48 | 7201 | 6.42 |
| 2022 | Validação | 91846 | 91750 | 96 | 7154 | 7.8 |
| 2023 | Teste | 91256 | 91178 | 78 | 7076 | 7.76 |

## Alvo por conjunto

| conjunto | inscricoes | rotuladas | sem_alvo | suficientes | taxa_% |
|---|---|---|---|---|---|
| Histórico | 252687 | 252606 | 81 | 6923 | 2.74 |
| Treino | 644924 | 644845 | 79 | 29034 | 4.5 |
| Validação | 91846 | 91750 | 96 | 7154 | 7.8 |
| Teste | 91256 | 91178 | 78 | 7076 | 7.76 |

## Linhas sem alvo

| ANO | sem_alvo | sem_corte | sem_nota | aprovado_1 |
|---|---|---|---|---|
| 2017 | 81 | 81 | 0 | 0 |
| 2018 | 10 | 10 | 0 | 0 |
| 2019 | 6 | 6 | 0 | 0 |
| 2020 | 15 | 15 | 0 | 0 |
| 2021 | 48 | 48 | 0 | 16 |
| 2022 | 96 | 96 | 0 | 51 |
| 2023 | 78 | 78 | 0 | 37 |

## Fórmula da nota: regras comparadas

| regra | linhas | divergentes | maior_diferenca | alvo_1_para_0 | so_ponto_flutuante | alvo_0_para_1 |
|---|---|---|---|---|---|---|
| Sem arredondar | 1080713 | 113458 | 0.005000 | 1387 | 1228 | 0 |
| Meio para cima, 2 casas (conta inteira) | 1080713 | 0 | 0.000000 | 0 | 0 | 0 |
| round(2) do pandas | 1080713 | 128 | 0.010000 | 0 | 0 | 0 |
| Truncar em 2 casas | 1080713 | 56967 | 0.010000 | 159 | 0 | 0 |

## Nota da função compartilhada x original

| ANO | linhas | diferentes | empates_meio_centesimo | alvo_diferente |
|---|---|---|---|---|
| 2017 | 252687 | 0 | 0 | 0 |
| 2018 | 186574 | 0 | 0 | 0 |
| 2019 | 189741 | 0 | 0 | 0 |
| 2020 | 156340 | 0 | 0 | 0 |
| 2021 | 112269 | 0 | 0 | 0 |
| 2022 | 91846 | 0 | 0 | 0 |
| 2023 | 91256 | 0 | 319 | 0 |

## Cortes por chave

| verificacao | quantidade |
|---|---|
| Linhas com algum campo da chave ausente | 0 |
| Chaves (curso, modalidade, edição, ano) | 8343 |
| Chaves com mais de um corte | 0 |
| Chaves que misturam corte preenchido e ausente | 0 |
| Chaves sem nenhum corte | 132 |
| Linhas nas chaves sem nenhum corte | 334 |

## Cobertura do corte anterior

| ANO | conjunto | linhas | com_corte | cobertura_% | cobertura_ed1_% | cobertura_ed2_% | rotuladas_com_corte |
|---|---|---|---|---|---|---|---|
| 2018 | Treino | 186574 | 177008 | 94.9 | 96.7 | 91.9 | 176999 |
| 2019 | Treino | 189741 | 138821 | 73.2 | 92.4 | 52.2 | 138815 |
| 2020 | Treino | 156340 | 117184 | 75 | 63.4 | 96 | 117173 |
| 2021 | Treino | 112269 | 105687 | 94.1 | 93.1 | 95.8 | 105647 |
| 2022 | Validação | 91846 | 90074 | 98.1 | 96.7 | 99.8 | 90001 |
| 2023 | Teste | 91256 | 83676 | 91.7 | 90.7 | 93.1 | 83655 |

## Entradas de treino x inferência

| cenario | linhas | treino_x_inferencia | tipos_e_indice_iguais | treino_x_parquet |
|---|---|---|---|---|
| 2017 | 252687 | 0 | sim | 0 |
| 2018 | 186574 | 0 | sim | 0 |
| 2019 | 189741 | 0 | sim | 0 |
| 2020 | 156340 | 0 | sim | 0 |
| 2021 | 112269 | 0 | sim | 0 |
| 2022 | 91846 | 0 | sim | 0 |
| 2023 | 91256 | 0 | sim | 0 |
| Amostra, uma inscrição por vez | 1000 | 0 | sim | - |

## Uso exclusivo do corte de N−1

| ANO | linhas | so_historico_n_1 | outros_anos_adulterados | corte_do_proprio_ano_adulterado | consulta_direta | com_corte_sem_historico_n_1 |
|---|---|---|---|---|---|---|
| 2017 | 252687 | 0 | 0 | 0 | 0 | 0 |
| 2018 | 186574 | 0 | 0 | 0 | 0 | 0 |
| 2019 | 189741 | 0 | 0 | 0 | 0 | 0 |
| 2020 | 156340 | 0 | 0 | 0 | 0 | 0 |
| 2021 | 112269 | 0 | 0 | 0 | 0 | 0 |
| 2022 | 91846 | 0 | 0 | 0 | 0 | 0 |
| 2023 | 91256 | 0 | 0 | 0 | 0 | 0 |

## Colunas proibidas

| coluna_proibida | no_parquet | nas_entradas |
|---|---|---|
| NOTA_CORTE | sim | não |
| CLASSIFICACAO | não | não |
| MATRICULA | não | não |
| NOTA_SUFICIENTE | sim | não |
| APROVADO | sim | não |
| ANO | sim | não |
| SEXO | sim | não |
| IDADE | sim | não |
| UF_CANDIDATO | sim | não |
| MUNICIPIO_CANDIDATO | sim | não |
| MUNICIPIO_CAMPUS | sim | não |
| GRUPO_CONCORRENCIA | sim | não |

## Linhas sem corte anterior

| ANO | linhas | sem_corte | chave_inexistente_em_n_1 | chave_sem_valor_em_n_1 | corte_e_dif_incoerentes | outros_atributos_ausentes |
|---|---|---|---|---|---|---|
| 2017 | 252687 | 252687 | 252687 | 0 | 0 | 0 |
| 2018 | 186574 | 9566 | 9566 | 0 | 0 | 0 |
| 2019 | 189741 | 50920 | 50907 | 13 | 0 | 0 |
| 2020 | 156340 | 39156 | 39142 | 14 | 0 | 0 |
| 2021 | 112269 | 6582 | 6575 | 7 | 0 | 0 |
| 2022 | 91846 | 1772 | 1742 | 30 | 0 | 0 |
| 2023 | 91256 | 7580 | 7475 | 105 | 0 | 0 |

## Cenários de inferência

| cenario | linhas | com_corte | sem_corte |
|---|---|---|---|
| Ano-alvo 2024 (usa o corte de 2023) | 91256 | 91178 | 78 |
| Ano-alvo 2030, sem histórico de N−1 | 91256 | 0 | 91256 |
| Campus que não existe no histórico | 91256 | 0 | 91256 |

## Entradas dos baselines publicados x montagem compartilhada

| conjunto | linhas | linhas_diferentes | tipos_iguais |
|---|---|---|---|
| Treino | 644845 | 0 | sim |
| Validação | 91750 | 0 | sim |

## Empates entre nota e corte anterior (treino e validação)

| ANO | com_corte | nota_igual_ao_corte_anterior |
|---|---|---|
| 2018 | 176999 | 4 |
| 2019 | 138815 | 7 |
| 2020 | 117173 | 3 |
| 2021 | 105647 | 9 |
| 2022 | 90001 | 3 |
