# Enunciado: Trabalho Final de Aprendizado de Máquina Aplicado

Resumo do enunciado da disciplina **DEIN0118 – Inteligência Artificial (CP), 2026.2**,
Profa. Dra. Alana Oliveira. Em caso de dúvida, vale o PDF original da professora.

**Entrega única: 03/12/2026.** Atraso perde 10% da nota por dia, até 3 dias.
**Valor:** 10 pontos da nota final.

---

## O que é o trabalho

Treinar um modelo de aprendizado de máquina para um problema real, colocá-lo em uma aplicação
que um usuário final consiga usar e relatar o desenvolvimento em um artigo no formato SBC.

Ao final, a equipe deve saber:
1. Formular um problema como tarefa de ML e justificar a formulação.
2. Preparar dados, treinar e comparar modelos com protocolo de avaliação adequado.
3. Interpretar resultados e limitações, incluindo vieses e riscos éticos.
4. Implantar o modelo em uma aplicação utilizável.
5. Comunicar o trabalho em linguagem científica, no formato SBC.

## Requisitos do conjunto de dados

- Público ou de uso autorizado, com **fonte e licença citadas no artigo**.
- Tamanho suficiente para treino, validação e teste (referência: 500+ instâncias tabulares).
- Sem dados pessoais identificáveis sem anonimização e autorização (LGPD).
- Não pode ser conjunto didático (Iris, Titanic, MNIST) como problema principal.

---

## Parte 1: o modelo (35% da nota)

Treinar e comparar **pelo menos três abordagens**, uma delas um baseline simples, e justificar o
modelo final com evidência experimental.

1. **Formulação:** tipo de tarefa, alvo, atributos e métrica de sucesso, justificados pelo domínio.
2. **Análise exploratória:** distribuição do alvo, ausentes, outliers e correlações, com **pelo menos duas figuras**.
3. **Pré-processamento:** limpeza, codificação de categóricas, normalização e desbalanceamento.
4. **Protocolo:** treino/validação/teste ou validação cruzada estratificada, **semente fixa** e
   **busca de hiperparâmetros documentada**.
5. **Comparação:** um baseline (classe majoritária, regressão linear/logística, árvore rasa) e
   **pelo menos dois modelos de famílias diferentes** (ensemble de árvores, SVM, k-NN, rede neural...).
6. **Avaliação no teste:** para classificação, F1, precisão, revocação, AUC-ROC e matriz de confusão
   (média e desvio padrão se houver validação cruzada).
7. **Interpretação:** importância de atributos, SHAP, LIME ou análise de erros: o que o modelo
   aprendeu e onde falha.
8. **Persistência:** modelo e pipeline salvos (joblib, pickle, ONNX...) para uso pela aplicação.

**Regras:**
- Todo pré-processamento ajustado **apenas no treino** (sem vazamento de dados).
- Modelos pré-treinados só com fine-tuning ou como extratores de características.
- Treinamento reprodutível por notebook ou script no repositório.

---

## Parte 2: a aplicação (25% da nota)

A aplicação carrega o modelo salvo e entrega predições por uma interface. **Não basta um notebook.**
Formatos aceitos: web (Streamlit, Gradio, Flask, FastAPI com front, Django), API REST documentada com
cliente, app móvel/desktop, bot ou extensão de navegador.

1. **Entrada de dados** com validação de entradas inválidas e mensagens de erro claras.
2. **Saída interpretável:** predição + confiança/probabilidade e, se possível, uma explicação
   (atributos mais influentes).
3. **Mesmo pipeline:** aplicar exatamente o pré-processamento do treino, **carregado do arquivo
   salvo**, não reimplementado.
4. **Contexto de uso:** dizer para quem é, o que faz e quais as limitações.
5. **Execução documentada:** README que funcione em máquina limpa (requirements.txt, conda ou Dockerfile).

**Bônus (até +1,0 ponto):**
- Implantação pública por link (Hugging Face Spaces, Streamlit Community Cloud, Render...).
- Registro das predições e coleta de feedback do usuário.
- Testes automatizados do pipeline ou da API.

---

## Parte 3: o artigo SBC (30% da nota)

Artigo de **8 a 12 páginas, incluindo referências**, no modelo oficial da SBC (LaTeX ou Word;
no Overleaf: "SBC Conferences Template").

| Seção | Conteúdo mínimo | Extensão sugerida |
|---|---|---|
| Título, autores, resumo, abstract | Resumo (PT) e abstract (EN), até 10 linhas cada | 0,5 p. |
| 1. Introdução | Contexto, problema, motivação, objetivo, contribuições | 1 p. |
| 2. Fundamentação teórica | Algoritmos e métricas, com referências | 1–1,5 p. |
| 3. Trabalhos relacionados | **Pelo menos 3**, comparados em tabela | 1 p. |
| 4. Metodologia | Dados, EDA, pré-processamento, modelos, protocolo | 2 p. |
| 5. Resultados e discussão | Tabelas e figuras, comparação, interpretação, análise de erros | 2 p. |
| 6. A aplicação | Arquitetura (com diagrama), tecnologias, capturas de tela | 1 p. |
| 7. Ética e limitações | Vieses, privacidade, uso indevido, limites de generalização | 0,5 p. |
| 8. Conclusão | Resposta ao objetivo e próximos passos | 0,5 p. |
| Referências | Formato SBC, **mínimo 8**, sendo **pelo menos 4 artigos científicos** | – |

**Regras:**
- Não alterar margens, fonte ou espaçamento do modelo.
- Toda figura e tabela referenciada e discutida no texto; resultados numéricos em tabelas, com a
  melhor métrica destacada.
- Link do repositório em nota de rodapé ou na Metodologia.
- Afirmações de desempenho apoiadas nos experimentos do próprio grupo.
- Seção curta **"Declaração de uso de IA"** (obrigatória): ferramentas, finalidade e onde foram usadas.

---

## Vídeo (10% da nota)

Até **5 minutos**, mostrando a aplicação em uso e um resumo dos resultados do modelo.

---

## O que entregar

- Artigo em PDF no modelo SBC.
- Link do repositório (GitHub ou GitLab, público ou com acesso para a docente).
- Vídeo de demonstração.

**O repositório deve conter:**
- `README.md` com descrição, instruções de execução e **link do vídeo**;
- `notebooks/` ou `src/` com o treinamento reprodutível;
- `app/` com a aplicação;
- o modelo salvo, ou instrução para regerá-lo se exceder o limite de tamanho;
- `requirements.txt`, ambiente conda ou Dockerfile.

---

## Critérios de avaliação

| Componente | Peso | O que é avaliado |
|---|---|---|
| Modelo de AM | 35% | Formulação, rigor do protocolo (sem vazamento), comparação com baseline, métricas corretas, interpretação |
| Artigo SBC | 30% | Aderência ao modelo, clareza, encadeamento, figuras e tabelas, trabalhos relacionados, discussão crítica e ética, referências |
| Aplicação | 25% | Funcionamento, uso do pipeline salvo, validação de entradas, saída interpretável, usabilidade, reprodutibilidade |
| Vídeo | 10% | Clareza da aplicação em uso e dos resultados, dentro de 5 minutos |

| Nível | Fração do peso | Descrição |
|---|---|---|
| Excelente | 100% | Atende a tudo com rigor e **vai além em algum aspecto** |
| Bom | 75% | Atende com falhas pontuais |
| Regular | 50% | Atende parcialmente; lacunas metodológicas relevantes |
| Insuficiente | ≤ 25% | Requisitos centrais ausentes ou incorretos |

A docente pode convocar a equipe para **arguição** sobre o código e o artigo; qualquer membro pode
ser questionado sobre qualquer parte. A nota pode ser individualizada pela arguição e por
autoavaliação e coavaliação.

---

## Uso de IA generativa e integridade

- **Permitido:** tirar dúvidas, depurar código, revisar a escrita, sugerir estrutura.
- **Obrigatório:** a seção "Declaração de uso de IA" no artigo.
- **Proibido:** gerar resultados, tabelas ou referências que não correspondam a experimentos e fontes reais.
- **Referências inexistentes ou experimentos não reproduzíveis anulam o componente correspondente.**
- Plágio resulta em nota zero.
- Dúvidas de escopo devem ser esclarecidas até 26/11/2026.

---

## Lista de verificação antes de entregar

**Modelo**
- [ ] 1. O pré-processamento foi ajustado apenas no treino?
- [ ] 2. Há um baseline e pelo menos dois modelos de famílias diferentes?
- [ ] 3. As métricas foram calculadas no teste, separado desde o início?
- [ ] 4. A semente está fixa e o treinamento reproduz a partir do repositório?
- [ ] 5. O modelo final e o pipeline foram salvos em arquivo?

**Aplicação**
- [ ] 6. A aplicação carrega o pipeline salvo, sem reimplementar o pré-processamento?
- [ ] 7. Entradas inválidas geram mensagens de erro claras?
- [ ] 8. A saída mostra a confiança ou probabilidade?
- [ ] 9. A interface informa o propósito e as limitações?
- [ ] 10. O README permite executar a aplicação em uma máquina limpa?

**Artigo**
- [ ] 11. Modelo SBC sem alterações de formatação, entre 8 e 12 páginas?
- [ ] 12. Resumo e abstract presentes?
- [ ] 13. Todas as figuras e tabelas referenciadas e discutidas no texto?
- [ ] 14. Todas as referências existem e foram efetivamente consultadas?
- [ ] 15. Há seções de ética e de declaração de uso de IA, e o link do repositório?
- [ ] 16. O vídeo tem até 5 minutos e mostra a aplicação em uso?
