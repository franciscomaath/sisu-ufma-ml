# Verificação da preparação dos dados

Arquivo gerado por `python -m src.verificar_dados`. Não editar à mão.
Não treina modelos nem calcula métricas de desempenho.

Linhas no parquet limpo: 1080713.
Histórico versionado idêntico ao reconstruído: sim (8343 linhas).

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

## Fórmula da nota: efeito por ano

| ANO | divergentes_sem_arredondar | alvo_mudaria_sem_arredondar | empates_meio_centesimo | divergentes_round2 |
|---|---|---|---|---|
| 2017 | 0 | 190 | 0 | 0 |
| 2018 | 0 | 186 | 0 | 0 |
| 2019 | 30983 | 222 | 0 | 0 |
| 2020 | 44053 | 241 | 0 | 0 |
| 2021 | 34078 | 235 | 0 | 0 |
| 2022 | 0 | 154 | 0 | 0 |
| 2023 | 4344 | 159 | 319 | 128 |

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

## Montagem em lote x cenários de inferência

| verificacao | linhas_comparadas | linhas_diferentes | acima_de_1e-9 |
|---|---|---|---|
| Histórico versionado x reconstruído (montagem em lote) | 1080713 | 0 | 0 |
| Histórico completo x só anos anteriores | 1080713 | 0 | 0 |
| Lote x uma inscrição por vez (amostra) | 1000 | 0 | 0 |
| Nota original x recalculada: meio para cima, 2 casas (conta inteira) | 1080713 | 0 | 0 |
| Nota original x recalculada: sem arredondar | 1080713 | 291305 | 47666 |
