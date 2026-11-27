# Comparação dos baselines na validação de 2022

## Dados e métodos

Usamos 644.845 inscrições rotuladas de 2018–2021 no treino e 91.750 inscrições rotuladas de 2022 na validação. Foram excluídas 79 linhas de treino e 96 de validação por falta do alvo `NOTA_SUFICIENTE`. Os três baselines foram avaliados nas mesmas inscrições de 2022. O conjunto de 2023 está reservado para a avaliação final.

- **Regra do corte anterior.** Prevê nota suficiente quando a nota do candidato supera o corte do ano anterior para a mesma oferta, modalidade e edição. Empates e ausência de histórico recebem a classe 0.
- **Regressão logística sem pesos.** Usa os 17 atributos da D2 e dá o mesmo peso a cada inscrição no treino.
- **Regressão logística com pesos balanceados.** Usa os mesmos atributos e dá mais peso à classe minoritária durante o treino.

O pré-processamento da regressão foi ajustado com os dados de treino. Ele preenche valores numéricos ausentes com a mediana, sinaliza as ausências, padroniza as variáveis numéricas e codifica as categóricas com one-hot encoding.

## Resultados

| Abordagem | PR-AUC (AP) | Precisão | Revocação | F1 |
|---|---:|---:|---:|---:|
| Regra do corte anterior | 0,2987 | 0,8890 | 0,2722 | 0,4167 |
| Regressão logística sem pesos | **0,6991** | **0,9271** | 0,2523 | 0,3967 |
| Regressão logística com pesos balanceados | 0,6836 | 0,6661 | **0,5485** | **0,6016** |

A PR-AUC foi calculada como *average precision* (AP), usando as duas classes previstas pela regra e as probabilidades previstas pelas regressões. Precisão, revocação e F1 foram calculados com as classes previstas, aplicando limiar de 0,5 nas regressões.

A regressão sem pesos liderou em PR-AUC. Com o limiar de 0,5, a regressão balanceada identificou 54,85% dos casos de nota suficiente, ante 25,23% da regressão sem pesos. A precisão caiu de 92,71% para 66,61% e o F1 subiu de 0,3967 para 0,6016.

## Reprodução

Com o ambiente do projeto ativo e `data/processed/sisu_limpo.parquet` gerado, execute `python -m src.baselines`. O [CSV dos resultados](baselines_2022.csv) contém os valores exibidos pelo comando. Na verificação da implementação, `python -m pytest -q` passou em 17 testes automatizados.
