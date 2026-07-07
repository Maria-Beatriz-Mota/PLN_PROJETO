# Pipeline de Análise de Artigos Científicos — Explicação Completa

> Documentação técnica do `pipeline_abnt_corrigido.ipynb` e do notebook de aprendizado `aprendizado_pipeline_PLN.ipynb`.

---

## Visão Geral

O pipeline analisa artigos científicos em português e verifica:
- **Estrutura** conforme NBR 6022 (norma ABNT para artigos)
- **Coerência temática** entre as seções
- **Vocabulário relevante** por seção
- **Entidades nomeadas** (autores, instituições, datas)
- **Sugestões de melhoria** geradas por LLM local

```
Texto bruto (PDF)
       │
       ▼
[1] Pré-processamento ──────── re, nltk
       │
       ▼
[2] Detecção de Seções ─────── REGEX → rapidfuzz → spaCy
       │
       ▼
[3] Validação ABNT ─────────── Python puro (NBR 6022)
       │
       ▼
[4] Coerência Temática ──────── BERTimbau + cosseno
       │
       ▼
[5] Análise Léxica ──────────── TF-IDF + BoW (scikit-learn)
       │
       ▼
[6] NER ─────────────────────── spaCy pt_core_news_lg
       │
       ▼
[7] Relatório Final ─────────── matplotlib
       │
       ▼
[8] Feedback LLM (opcional) ─── ollama + llama3
```

---

## Etapa 0 — Configuração do Ambiente

### O que faz
Carrega todas as bibliotecas, baixa recursos de linguagem e lê o dataset de artigos.

### Bibliotecas principais

| Biblioteca | Função no pipeline |
|---|---|
| `pandas` | Leitura e manipulação do CSV de artigos |
| `spacy` | Detecção de seções (fallback) e NER |
| `nltk` | Stopwords em português |
| `transformers` | BERTimbau para embeddings |
| `torch` | Backend de computação para BERTimbau |
| `sklearn` | TF-IDF e Bag of Words |
| `rapidfuzz` | Fuzzy matching para títulos de seção |
| `matplotlib` | Visualizações do relatório |
| `ollama` | Interface com LLM local |

### Decisão de projeto
O modelo spaCy `pt_core_news_lg` é carregado com fallback suave — se não estiver instalado, o pipeline continua sem o fallback semântico (Etapa 2 passo 3) e sem NER (Etapa 6).

---

## Etapa 1 — Pré-processamento de Texto

### O que faz
Limpa o texto bruto extraído de PDFs para remover ruído que prejudicaria as análises.

### Funções implementadas

#### `preprocessar_texto(texto, remover_stopwords=False)`
Limpeza completa para análises léxicas e NER. Passos:
1. **Junta hifenização de PDF** — `apren-\ndizado` → `aprendizado`
2. **Minúsculas** — normaliza caixa
3. **Remove URLs e e-mails** — ruído frequente em artigos copiados de web
4. **Remove citações** — `[1,2]` e `(Silva, 2020)` não são conteúdo semântico
5. **Remove caracteres especiais** — mantém letras acentuadas via `\w` Unicode
6. **Colapsa repetições** — `---` → `-`
7. **Normaliza espaços** — remove espaços duplos e quebras extras
8. **Remove stopwords** (opcional) — apenas quando `remover_stopwords=True`

#### `preparar_texto_para_estrutura(texto)`
Normalização suave que **preserva quebras de linha**, necessária para a detecção de seções (os títulos de seção estão em linhas isoladas).

### Por que separar as duas funções?
- `preprocessar_texto` é para análise de **conteúdo** (léxico, NER, BERTimbau)
- `preparar_texto_para_estrutura` é para análise de **formato** (posições de títulos)

### Stopwords — quando remover?

| Remover | Não remover |
|---|---|
| TF-IDF e BoW (Etapa 5) | BERTimbau (Etapa 4) |
| Visualizações de termos | NER com spaCy (Etapa 6) |
| Contagem de frequência | Detecção de seções (Etapa 2) |

Palavras como "de", "que", "em" não carregam significado léxico, mas são necessárias para modelos que dependem de contexto gramatical.

---

## Etapa 2 — Detecção de Seções

### O que faz
Identifica onde cada seção começa no texto e retorna um dicionário `{ label: posição_char }`.

### Estratégia em Cascata

A cascata ativa cada nível apenas se o anterior não encontrou todas as seções, evitando custo desnecessário:

```
REGEX (O(n), grátis)
   │
   ▼ [seções não encontradas]
rapidfuzz (O(n × variantes), barato)
   │
   ▼ [seções ainda faltando]
spaCy similarity (O(n × linhas), caro)
```

#### Passo 1 — REGEX
Expressões regulares que buscam padrões exatos de títulos. Exemplos:
- `(?i)^\s*\d*\.?\s*introdução\s*$` — "Introdução", "1. Introdução", "1.1 Introdução"
- `(?i)^\s*(referências(\s+bibliográficas)?)\s*$` — "Referências", "Referências Bibliográficas"

**Por que `MULTILINE`?** O flag `re.MULTILINE` faz `^` e `$` funcionar em cada linha, não só no início/fim do texto inteiro.

#### Passo 2 — rapidfuzz
Para títulos com erros de digitação ou variações não previstas no REGEX.

```python
fuzz.ratio('introducao', 'introdução')  # → 93 (score 0-100)
fuzz.ratio('metodlogia', 'metodologia')  # → 90 (typo tolerado)
```

- `fuzz.ratio` — compara sequências caractere a caractere
- `fuzz.token_sort_ratio` — ignora ordem das palavras (útil para "Materiais e Métodos" vs "Métodos e Materiais")
- Threshold padrão: **80** — evita falsos positivos

#### Passo 3 — spaCy similarity
Fallback semântico: compara vetores de palavras usando o modelo `pt_core_news_lg`.

```python
nlp("introdução").similarity(nlp("contextualização"))  # > 0.75 → aceita
```

**Threshold padrão: 0.75** — mais restritivo que o fuzzy para evitar ruído semântico.

### Estrutura de saída
```python
{
    "resumo": 45,        # offset em caracteres no texto
    "introducao": 320,
    "metodologia": 1200,
    "_violacoes": {      # seções não encontradas por severidade
        "critico": ["conclusao"],
        "aviso": [],
        "observacao": ["agradecimentos"]
    }
}
```

### Seções NBR 6022 monitoradas

| Grupo | Seções |
|---|---|
| Pré-textuais | título, autores, resumo, abstract, palavras-chave, data de submissão, DOI |
| Textuais | introdução, referencial teórico, metodologia, resultados, discussão, implicações, conclusão |
| Pós-textuais | referências, agradecimentos, apêndice, anexo, glossário |

---

## Etapa 3 — Validação ABNT (NBR 6022)

### O que faz
Verifica se o artigo segue a estrutura exigida pela norma NBR 6022 e calcula um **score de conformidade** de 0 a 100.

### Verificações realizadas

1. **Presença** — cada seção está no texto?
2. **Ordem** — as seções aparecem na sequência correta?
3. **Duplicatas** — alguma seção foi detectada mais de uma vez no mesmo offset?

### Score de Conformidade

```
Score = 100 - penalidades
```

| Problema | Penalidade por ocorrência |
|---|---|
| Seção crítica ausente | -15 pontos |
| Seção fora de ordem | -10 pontos |
| Seção duplicada | -5 pontos |
| Seção recomendada ausente | -3 pontos |

Mínimo: 0 pontos.

### Severidade das seções

| Crítico (obrigatório) | Aviso (recomendado) | Observação (opcional) |
|---|---|---|
| título, autores, resumo | abstract, metodologia | data_submissao, doi |
| palavras_chave | resultados | referencial_teorico |
| introducao, conclusao | | agradecimentos, apendice |
| referencias | | glossario, anexo |

### Estrutura de saída
```python
{
    "resumo"    : "ok",
    "introducao": "ok",
    "conclusao" : "critico",   # ausente e obrigatório
    "metodologia": "fora_de_ordem",
    "_resumo": {
        "conforme_nbr6022"  : False,
        "score_conformidade": 55,
        "criticos"          : ["conclusao"],
        "fora_de_ordem"     : ["metodologia"],
        "duplicados"        : [],
        "avisos"            : ["abstract"],
    }
}
```

---

## Etapa 4 — Coerência Temática (BERTimbau)

### O que faz
Verifica se as seções do artigo tratam do mesmo tema central, usando embeddings semânticos.

### O que é BERTimbau?
[BERTimbau](https://github.com/neuralmind-ai/portuguese-bert) é o BERT pré-treinado em textos em **português brasileiro** pela NeuralMind.  
Modelo: `neuralmind/bert-base-portuguese-cased` (768 dimensões, 12 camadas).

### Como gera embeddings?

#### Problema: BERT aceita máximo 512 tokens
Seções de artigos científicos têm frequentemente mais de 512 tokens.

#### Solução: Sliding Window
1. Divide o texto em janelas sobrepostas de 510 tokens (stride=256)
2. Gera embedding para cada janela
3. Tira a **média** dos embeddings das janelas

```
Texto:  [token_1 ... token_800]
Janela 1: [CLS] token_1..token_510 [SEP]
Janela 2: [CLS] token_255..token_765 [SEP]  ← sobreposição de 255 tokens
Janela 3: [CLS] token_511..token_800 [SEP]
Embedding final = média(embed_1, embed_2, embed_3)
```

#### Por que excluir [CLS] e [SEP]?
- `[CLS]` é treinado para **Next Sentence Prediction** (NSP), não representa conteúdo
- `[SEP]` é marcador de separação de segmentos
- **Mean pooling** dos tokens internos captura melhor o significado semântico

### Similaridade de Cosseno

$$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

- Valor entre 0 (sem relação) e 1 (idêntico)
- Threshold padrão: **0.50** — seções abaixo são sinalizadas

#### Por que cosseno e não distância euclidiana?
O cosseno mede o **ângulo** entre vetores, independente de magnitude.  
Textos longos geram vetores com maior norma (magnitude maior).  
A distância euclidiana penalizaria textos longos artificialmente.  
O cosseno normaliza esse efeito.

### Saída
```python
{
    "labels"          : ["resumo", "introducao", "metodologia", ...],
    "matriz"          : [[1.0, 0.82, 0.75, ...], ...],  # NxN float
    "media_por_secao" : {"resumo": 0.78, "introducao": 0.81, ...},
    "secoes_problemas": ["discussao"],  # abaixo do threshold
    "threshold"       : 0.50,
}
```

---

## Etapa 5 — Análise Léxica (TF-IDF e BoW)

### O que faz
Extrai os termos mais relevantes do artigo e por seção, usando dois métodos complementares.

### Bag of Words (BoW)
Conta a frequência bruta de cada termo.

```
Texto: "aprendizado máquina aprendizado deep learning"
BoW:   {"aprendizado": 2, "máquina": 1, "deep": 1, "learning": 1}
```

**Limitação:** palavras comuns ("artigo", "estudo") recebem alta frequência sem serem relevantes.

### TF-IDF (Term Frequency — Inverse Document Frequency)

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

- **TF** (Term Frequency): frequência do termo no documento
- **IDF** (Inverse Document Frequency): penaliza termos que aparecem em muitos documentos
- **N**: total de documentos (seções, neste caso)

**Estratégia inter-seções:** cada seção é um "documento" do corpus.  
Termos exclusivos de uma seção têm IDF alto → mais relevantes.  
Termos que aparecem em todas as seções têm IDF baixo → menos informativos.

### Parâmetros do `TfidfVectorizer`

| Parâmetro | Valor | Motivo |
|---|---|---|
| `max_features` | 500 | Limita vocabulário, evita esparsidade |
| `ngram_range` | (1, 2) | Inclui bigramas ("aprendizado máquina") |
| `sublinear_tf` | True | `log(1 + tf)` — suaviza textos muito longos |

### Saída
```python
{
    "artigo_completo": {
        "tfidf": [("aprendizado máquina", 0.32), ("redes neurais", 0.28), ...],
        "bow"  : {"aprendizado": 45, "dados": 32, ...}
    },
    "por_secao": {
        "metodologia": {
            "tfidf": [("deep learning", 0.41), ...],
            "bow"  : {"modelo": 12, ...}
        },
        ...
    }
}
```

---

## Etapa 6 — NER (Reconhecimento de Entidades Nomeadas)

### O que faz
Identifica automaticamente nomes de pessoas, organizações, locais e datas no texto.

### Ferramenta: spaCy `pt_core_news_lg`
Modelo treinado em textos em português com capacidade de NER.

### Tipos de entidade detectados

| Tipo | Descrição | Exemplos |
|---|---|---|
| `PER` | Pessoas | "João Silva", "Smith et al." |
| `ORG` | Organizações | "Universidade de São Paulo", "CNPq" |
| `LOC` | Locais | "Brasil", "São Paulo" |
| `DATE` | Datas e períodos | "2020", "janeiro de 2021" |
| `MISC` | Demais entidades | Outros tipos reconhecidos pelo modelo |

### Processamento em blocos

Problema: textos científicos longos (100k+ caracteres) podem sobrecarregar a memória do spaCy.

Solução:
- Limite: 100.000 caracteres por artigo
- Blocos: 50.000 caracteres
- Sobreposição: 500 caracteres (evita perder entidades no ponto de corte)

```
[0 ... 50000] bloco 1
      [49500 ... 99500] bloco 2  ← sobreposição de 500 chars
```

### Saída
```python
{
    "PER"      : ["João Silva", "Maria Santos"],
    "ORG"      : ["USP", "FAPESP"],
    "LOC"      : ["Brasil", "São Paulo"],
    "DATE"     : ["2020", "março de 2021"],
    "MISC"     : [...],
    "_contagem": {"PER": 2, "ORG": 2, "LOC": 2, "DATE": 2, "MISC": 0}
}
```

---

## Etapa 7 — Relatório Final

### O que faz
Consolida os resultados de todas as etapas em um relatório textual + visualizações matplotlib.

### Componentes

#### 1. Bloco ABNT
- Score numérico e barra de progresso ASCII
- Status (CONFORME / NÃO CONFORME)
- Lista de seções críticas ausentes, fora de ordem, duplicadas

#### 2. Heatmap de Coerência
- Matriz NxN de similaridade de cosseno entre seções
- Escala de cores YlOrRd (amarelo=baixo, vermelho=alto)
- Valores numéricos em cada célula

#### 3. Barras TF-IDF por Seção
- Gráfico de barras horizontais para cada seção textual principal
- Mostra os top-N termos mais relevantes da seção

#### 4. Entidades NER
- Lista de entidades por tipo (PER, ORG, LOC, DATE)
- Limitado às primeiras 5-6 por tipo para legibilidade

---

## Etapa 8 — Feedback Textual (LLM Local)

### O que faz
Gera sugestões de melhoria em linguagem natural usando um modelo de linguagem local via `ollama`.

### Por que LLM local?
- **Privacidade**: artigos científicos não saem da sua máquina
- **Custo zero**: sem API keys ou chamadas pagas
- **Controle**: você escolhe o modelo

### Pré-requisitos
1. Instalar o servidor ollama: https://ollama.com/download
2. Baixar o modelo: `ollama pull llama3` (no terminal, ~4GB)
3. Manter o servidor rodando (inicia automaticamente no Windows)

### Estratégia de Prompt
O prompt é montado estruturando os problemas encontrados nas etapas anteriores:

```
"Você é especialista em redação científica e normas ABNT.
Com base nos problemas identificados, forneça sugestões acionáveis:

- Seções obrigatórias ausentes: conclusao, referencias.
- Score de conformidade: 40/100.
- Baixa coerência temática: discussao (sim=0.38).
- Principais termos: aprendizado, redes neurais, dados.

Sugestões de melhoria:"
```

---

## Etapa 9 — Pipeline Completo

### O que faz
Função `executar_pipeline()` que orquestra todas as etapas em sequência para um único artigo.

### Parâmetros configuráveis

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `texto_bruto` | — | Texto do artigo (string) |
| `titulo` | `"Artigo"` | Nome exibido no relatório |
| `top_n_lexico` | `10` | Número de termos TF-IDF por seção |
| `threshold_coerencia` | `0.50` | Limiar de similaridade para sinalizar problemas |
| `gerar_fb` | `False` | Se True, chama ollama ao final |
| `modelo_llm` | `"llama3"` | Modelo ollama a usar |

### Retorno
Dicionário com todos os resultados intermediários:
```python
{
    "titulo"   : "Artigo de exemplo",
    "secoes"   : {...},    # Etapa 2
    "validacao": {...},    # Etapa 3
    "coerencia": {...},    # Etapa 4
    "lexico"   : {...},    # Etapa 5
    "entidades": {...},    # Etapa 6
    "feedback" : "..."     # Etapa 8 (se gerar_fb=True)
}
```

---

## Decisões Técnicas e Trade-offs

### 1. Cascata de detecção (REGEX → fuzzy → spaCy)
**Por quê?** Balanceia precisão e performance. REGEX é O(n) e determinístico; spaCy é caro mas tolerante a variações semânticas.

### 2. BERTimbau vs modelos menores
**Por quê?** BERTimbau foi pré-treinado especificamente em português brasileiro. Modelos multilíngues (mBERT, XLM-R) performam pior em PT-BR para tarefas de similaridade.

### 3. TF-IDF inter-seções vs inter-artigos
**Por quê?** Com muitos artigos do mesmo domínio, o IDF inter-artigos seria quase zero para todos os termos técnicos. Usar as seções como corpus dá relevância relativa dentro do próprio artigo.

### 4. Chunking com sobreposição (stride=256)
**Por quê?** Stride menor (= mais sobreposição) gera embeddings mais estáveis para textos com transições temáticas graduais, ao custo de mais processamento.

### 5. Score ABNT com penalidades lineares
**Por quê?** Simplicidade interpretável. Alternativas (pontuação geométrica, ponderada) seriam mais difíceis de justificar aos usuários sem validação empírica.

---

## Referências

- Souza, F.; Nogueira, R.; Lotufo, R. (2020). **BERTimbau: Pretrained BERT Models for Brazilian Portuguese**. BRACIS 2020. [GitHub](https://github.com/neuralmind-ai/portuguese-bert)
- ABNT NBR 6022:2018 — Informação e documentação — Artigo em publicação periódica científica — Apresentação
- Devlin, J. et al. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL-HLT 2019.
- Honnibal, M. & Montani, I. (2017). **spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing**.
