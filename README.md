# Sistema de Avaliação de Conformidade ABNT para Artigos Científicos

Sistema de Processamento de Linguagem Natural que avalia artigos científicos em
português quanto à conformidade **estrutural** (ABNT NBR 6022:2018) e de
**citações** (NBR 10520). Produz um score de conformidade, diagnóstico seção a
seção e **feedback textual gerado por LLM** em duas variantes comparáveis:
*baseline* (sem RAG) e *fundamentado nas normas* (com RAG).

> Projeto de mestrado (PPGEC). Este repositório contém **código e documentação**;
> o corpus de artigos (PDFs e textos completos) **não é versionado** — ver
> [Dados](#dados-não-versionados).

---

## Principais componentes

- **Detecção estrutural em 6 camadas** — regex → fuzzy → spaCy (cabeçalho) →
  spaCy (conteúdo) → BERTimbau (conteúdo) → zero-shot NLI. Localiza 20 elementos
  ABNT mesmo em artigos de estrutura não convencional (relatos de caso etc.).
- **Validação semântica híbrida** — combina score léxico e cosseno de embeddings
  BERTimbau-STS (peso 0,15/0,85) para classificar as seções obrigatórias.
- **Análises complementares** — BoW/TF-IDF por seção, coerência semântica entre
  seções de corpo, citações NBR 10520, NER de autores/instituições (spaCy PER/ORG
  na zona de autoria) e termos científicos (SciERC, no abstract) e BERTScore.
- **Feedback via LLM** — Llama-3.1-8B (Hugging Face Inference API) com fallback
  local (Ollama). Inclui sinal de coesão e sugestões de reescrita "Antes /
  Sugestão". Amostragem: `temperature=0.3`, `top_p=0.90`, `max_tokens=1200`
  (calibrados para evitar respostas incoerentes).
- **RAG** — recupera trechos de manuais de normalização ABNT relevantes aos
  problemas detectados e fundamenta o feedback citando a norma. **O usuário final
  sempre recebe a versão com RAG**; a versão sem RAG é gerada apenas nos scripts
  de avaliação (comparação pareada em `feedback_llm`, coluna `rag_ativo`).
- **Front-end Streamlit** — upload de PDF/DOCX/TXT, diagnóstico visual, barra de
  progresso e histórico de análises persistido em SQLite.

## Estrutura do repositório

```
PLN front-end/                 # Aplicação Streamlit
├── main.py, db.py             # entrada + persistência SQLite
├── analisador.py              # adaptador: pipeline -> contrato do front
├── carregar_modelos.py        # carregamento cacheado dos modelos
├── pipeline_abnt.py           # módulo extraído do notebook de funções
├── rag_abnt.py                # indexação e recuperação (RAG)
├── pages/                     # up_artigo, artigos, resultado
└── utils/styles.py

PLN_BACKEND/
├── PLN_PROJETO_FINAL/
│   ├── pipeline_abnt_funcoes_oficial.ipynb       # biblioteca de funções (fonte da verdade)
│   ├── pipeline_abnt_apresentacao_oficial.ipynb  # análise/apresentação
│   └── gabarito_*.csv                            # dados de avaliação
├── PLN_SciELO_API_3.ipynb     # coleta SciELO/ArticleMeta
└── normas_rag/                # manuais ABNT do RAG (PDFs não versionados)

resultados/                    # notebooks EXECUTADOS (com gráficos/tabelas visíveis no GitHub)
```

O notebook `pipeline_abnt_funcoes_oficial.ipynb` é a **fonte da verdade**; o
módulo `PLN front-end/pipeline_abnt.py` é extraído dele. Os notebooks na raiz
estão com os outputs limpos — as versões executadas (com resultados) estão em
[`resultados/`](resultados/), que o GitHub renderiza com gráficos e tabelas.

## Como executar o front-end

```bash
# 1. Dependências (Python 3.13)
pip install -r PLN_BACKEND/requirements.txt
python -m spacy download pt_core_news_lg

# 2. Token da Hugging Face para o feedback via LLM
#    Crie PLN_BACKEND/PLN_PROJETO_FINAL/.env com:
#    HUGGINGFACE_API_KEY=seu_token_aqui
#    (gere em https://huggingface.co/settings/tokens)
#    Sem token, o feedback usa o fallback local via Ollama.

# 3. Rodar o app
cd "PLN front-end"
python -m streamlit run main.py
```

A **primeira análise** carrega os modelos (~2-4 min); as seguintes usam cache.

### Reconstruir o índice do RAG

Os PDFs dos manuais e o índice não são versionados. Coloque os manuais em
`PLN_BACKEND/normas_rag/` e reconstrua uma vez:

```python
import rag_abnt
from sentence_transformers import SentenceTransformer
modelo = SentenceTransformer("rufimelo/Legal-BERTimbau-sts-base")
rag_abnt.construir_indice(modelo=modelo)
```

## Avaliação quantitativa

Detecção de presença de seções, sobre dois gabaritos:

**Gabarito assistido/estrutural** (25 artigos × 19 seções — anotação assistida com
verificação humana):

| Agregado | Acurácia | Precisão | Recall | F1 |
|----------|----------|----------|--------|-----|
| Micro    | 0.92     | 0.90     | 0.96   | **0.93** |
| Macro    | —        | 0.76     | 0.76   | 0.75 |

**Gabarito de conteúdo** (18 artigos — anotação humana **independente**, por
conteúdo mesmo sem heading). Concordância entre anotadores: **Cohen's kappa =
0.70** (substancial). Contra esse gabarito, as correções do pipeline elevaram o
detector de **F1 0.885 → 0.938** (recall 0.807 → 0.901), com ganhos concentrados
em metodologia e resultados (busca por conteúdo quando não há heading ABNT).

Reprodução na seção 17 do notebook de apresentação.

## Dados não versionados

Excluídos via `.gitignore` (regenerar/obter à parte):

- **Corpus SciELO** (`PLN_BACKEND/PLN_SCIELO/`) — PDFs e CSVs de textos completos
  (~130 MB); regenerável com `PLN_SciELO_API_3.ipynb`.
- **PDFs** dos manuais de normalização do RAG.
- **Banco** `base_abnt.db` — criado automaticamente ao rodar o app.
- **Índice RAG** (`.npy`/`.json`) — reconstruído com `rag_abnt.construir_indice`.
- **`.env`** com o token da Hugging Face.

## Modelos utilizados

spaCy `pt_core_news_lg` (NER de autores/instituições + similaridade de headings) ·
BERTimbau-STS `rufimelo/Legal-BERTimbau-sts-base` (validação semântica, coerência,
embeddings do RAG) · NER científico `RJuro/SciNERTopic` (SciERC, no abstract) ·
zero-shot `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` ·
LLM `meta-llama/Llama-3.1-8B-Instruct` (fallback Ollama).

## Limitações

Pesos/thresholds fixados a priori (sem calibração); base do RAG são manuais
universitários (não o texto oficial da ABNT); avaliação da qualidade do feedback
LLM ainda qualitativa; a seção Discussão tem detecção parcial quando fundida em
"Resultados e discussão".
