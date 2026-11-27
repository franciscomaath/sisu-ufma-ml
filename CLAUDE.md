# CLAUDE.md: sisu-ufma-ml

Trabalho final de Inteligência Artificial (UFMA, 2026.2), feito em equipe de 4 pessoas.
Prevemos a **chance de um candidato ter nota suficiente no SISU da UFMA** para um curso e uma
modalidade, a partir das notas do ENEM. Entrega única em **03/12/2026**.

## Leia antes de qualquer tarefa

1. `docs/ENUNCIADO.md`: o que o trabalho exige e como é avaliado.
2. `docs/DECISOES.md`: decisões de modelagem já tomadas. **Não contrarie uma decisão sem avisar.**
   Se algo parecer errado, aponte o problema e proponha a mudança no próprio `DECISOES.md`.
3. `reports/relatorio_etapa1.md`: como os dados foram limpos e por quê.
4. `src/config.py`: caminhos, semente, anos de cada divisão e colunas.

## Comandos

```bash
python -m src.limpeza   # CSV bruto -> data/processed/sisu_limpo.parquet
python -m src.features  # histórico de cortes e cobertura por ano
python -m src.particoes      # divisão de 2022 em A/B/C
python -m src.exploracao     # figuras da análise exploratória (só treino)
python -m src.experimentos   # busca em A, calibração em B/C e modelo salvo (não usa 2023)
pytest -q               # todos os testes devem passar antes de cada commit
```

O CSV bruto (~343 MB) não está no git. Veja no README onde baixar.

## Regras inegociáveis

Estas regras protegem a nota. Violar qualquer uma delas é um erro grave, não um detalhe.

1. **Sem vazamento de dados.** `NOTA_CORTE`, `CLASSIFICACAO` e `MATRICULA` nunca entram como
   atributos. `NOTA_CORTE` só serve para construir o alvo e o histórico de cortes.
2. **Atributos só da lista explícita** `config.ATRIBUTOS` (decisão D2). Não use "todas as colunas
   menos o vazamento": isso deixa passar `SEXO`, `IDADE` e municípios.
3. **Divisão temporal entre anos, nunca aleatória.** Treino 2018–2021, validação 2022, teste 2023.
   Nada de `train_test_split` sobre o conjunto inteiro. Dentro de 2022, e só ali, as inscrições
   rotuladas são divididas por grupo (`config.CHAVE_HISTORICO`) em A (partições 0–2: seleção de
   modelo), B (partição 3: ajuste do calibrador) e C (partição 4: escolha da calibração e do
   limiar), com `src/particoes.py` (D4 e D8). Não misture os papéis das partes.
4. **Todo pré-processamento que aprende dos dados fica dentro de um `Pipeline` do sklearn**,
   ajustado só no treino. Escala, encoders e imputação nunca são ajustados em validação ou teste.
5. **O teste (2023) é usado uma única vez**, depois que modelo, hiperparâmetros, calibração e
   limiar foram escolhidos na validação. Não ajuste nada olhando o teste. C também não é teste: ela
   foi usada para escolher a calibração e o limiar.
6. **Semente 42** (`config.SEMENTE`) em tudo que for aleatório.
7. **A aplicação não reimplementa o pré-processamento.** Ela carrega `models/pipeline_final.joblib`
   e usa a mesma função de montagem de entrada de `src/features.py` (decisão D13).
8. **Nunca invente resultados, números ou referências.** Toda métrica citada sai de um script que
   rodou; toda referência do artigo precisa existir. O enunciado anula o componente se isso acontecer.

## Convenções

- Código, comentários, nomes de variáveis, mensagens de commit e textos da aplicação em **português**.
- Código reutilizável em `src/`; notebooks só para exploração e figuras. O treino precisa rodar por script.
- Figuras do artigo em `reports/figures/`, PNG 300 dpi, títulos e eixos em português.
- Tabelas de resultados em `reports/` (CSV e Markdown).
- Toda função nova em `src/` ganha pelo menos um teste em `tests/`, com dados sintéticos
  (os testes não podem depender do CSV real; testes com o parquet real devem ser pulados se ele não existir).
- Não versionar dados (`data/raw/`, `data/processed/`) nem `*.joblib` grandes.
- Uma mudança de decisão de modelagem atualiza `docs/DECISOES.md` no mesmo commit.

## Contrato entre Modelo e Aplicação

| Arquivo | Quem gera | Conteúdo |
|---|---|---|
| `models/pipeline_final.joblib` | Modelo | Pré-processamento + modelo + calibrador; a aplicação chama só `predict_proba` |
| `models/historico_cortes.parquet` | Etapa 1 | Cortes por curso, modalidade, edição e ano |
| `models/metadados.json` | Modelo | Atributos, limiar, métricas de teste, versões, data |

Detalhes em `docs/DECISOES.md`, D13.

## Ao terminar uma tarefa

Entregue um relatório curto: o que foi feito, os números obtidos (copiados da saída dos scripts),
as decisões tomadas e o que exige decisão da equipe.
