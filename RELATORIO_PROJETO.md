# Relatório Técnico — Sistema de Avaliação de Conformidade ABNT para Artigos Científicos

**Data:** 06/07/2026 · **Autora do projeto:** Maria Beatriz · Documento de apoio para redação do artigo

---

## 1. Visão geral

Sistema de Processamento de Linguagem Natural que avalia artigos científicos em português quanto à conformidade **estrutural** (ABNT NBR 6022:2018) e de **citações** (NBR 10520), produzindo: score de conformidade (0-100), diagnóstico seção a seção, ocorrências pontuais e **feedback textual gerado por LLM** — em duas variantes comparáveis: baseline (sem RAG) e fundamentado em normas via RAG. Inclui front-end web (Streamlit) com persistência de análises e avaliação quantitativa do detector contra gabarito anotado.

**Pergunta de pesquisa implícita na comparação:** o feedback de LLM fundamentado por recuperação de trechos normativos (RAG) é mais preciso/útil que o feedback do mesmo modelo sem fundamentação?

## 2. Corpus

- **Fonte:** SciELO via ArticleMeta (coleta própria, `PLN_SCIELO/artigos_validos.csv`): 5.930 documentos com PDF e texto extraído (PyMuPDF).
- **Filtro de idioma:** classificação por blocos de texto (stopwords PT/EN/ES + fallback langdetect); mantidos documentos com ≥70% de blocos em PT e ≥2 blocos PT → **662 documentos elegíveis**.
- **Característica importante:** o corpus é heterogêneo — além de artigos, contém **resenhas, erratas, editoriais e listas de pareceristas** (na amostra de avaliação, 8 de 30 documentos não eram artigos). Isso motivou tanto as camadas de robustez do detector quanto decisões de anotação (ver §9).

## 3. Arquitetura

| Componente | Papel |
|---|---|
| `pipeline_abnt_funcoes_oficial.ipynb` | **Fonte da verdade** do pipeline (biblioteca de funções) |
| `pipeline_abnt_apresentacao_oficial.ipynb` | Notebook de análise/apresentação (17 seções, executado ponta a ponta sem erros) |
| `PLN front-end/pipeline_abnt.py` | Módulo Python **extraído automaticamente** do notebook de funções (escopo de import limpo, sem carregamento de modelos em import-time) |
| `PLN front-end/` (Streamlit) | `main.py`, `pages/up_artigo.py` (upload PDF/DOCX/TXT + toggle RAG), `pages/artigos.py` (histórico com score), `pages/resultado.py` (diagnóstico + feedback LLM), `analisador.py` (adaptador), `carregar_modelos.py` (cache de modelos), `rag_abnt.py` (RAG) |
| SQLite `base_abnt.db` | Tabelas `artigos` (texto + resultado JSON — reabertura instantânea), `erros_abnt`, `feedback_llm` (com flag **`rag_ativo`** 0/1 para comparação pareada) |

Modelos utilizados (todos abertos): spaCy `pt_core_news_lg`; **BERTimbau-STS** `rufimelo/Legal-BERTimbau-sts-base` (sentence-transformers, dim 768); NER `pierreguillou/ner-bert-base-cased-pt-lenerbr` (LeNER-BR) e `RJuro/SciNERTopic` (SciERC, só no abstract); zero-shot `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`; LLM `meta-llama/Llama-3.1-8B-Instruct` (HF Inference API) com fallback local Ollama (`llama3.2:3b`, depois `mistral`).

## 4. Detecção estrutural de seções — 6 camadas

Detecta a posição de 20 elementos ABNT (pré-textuais, textuais e pós-textuais). Camadas em cascata — cada uma só atua sobre o que as anteriores não acharam:

1. **Regex** de headings canônicos (linha própria; resumo/abstract também em forma inline "Resumo: ..."). Título tem heurística anti-cabeçalho editorial (ignora datas de recebimento, DOI, banners de tipo, bios; guarda de "início de prosa").
2. **Fuzzy matching** (rapidfuzz ≥80) com variantes por seção (ex.: metodologia ≈ "Materiais e Métodos", "Relato de caso"). Guardas: headings do corpo só **após** a posição do resumo (evita o banner "Relato de Caso" da capa); linhas terminadas em pontuação de frase são descartadas (evita fragmentos de texto corrido).
3. **spaCy cabeçalho** — similaridade vetorial de linhas candidatas vs rótulo de referência (limiar **0.85**, calibrado empiricamente: headings reais exatos ≈1.0; falsos positivos observados 0.76-0.82); linhas já reivindicadas por outra seção são excluídas.
4. **spaCy conteúdo** — similaridade de blocos de texto vs frase de referência da seção (só seções críticas faltantes).
5. **BERTimbau conteúdo** — mesmo princípio com embeddings BERTimbau-STS.
6. **Zero-shot NLI** (mDeBERTa multilingual) — classifica blocos sobrepostos contra descrições em PT das seções; último recurso para estrutura não convencional (ex.: relatos de caso).

**Decisão de projeto:** a validação de **ordem** das seções usa apenas posições de camadas de heading (1-2) — posições de camadas de conteúdo (4-6) são âncoras de bloco aproximadas e geravam falsos "fora de ordem".

## 5. Validação semântica híbrida (seções obrigatórias NBR 6022)

Para introdução, metodologia, resultados, conclusão e referências: score híbrido = **0,15·léxico + 0,85·semântico** (cosseno BERTimbau-STS entre o texto da seção e frase de referência; pesos fixados a priori — limitação declarada, ver §10). Classificação em 4 status: *Contém seção* (heading + semântica ok), *Contém com observação* (conteúdo compatível sem heading), *Requer revisão* (heading sem coerência semântica), *Não contém*.

**Score global** = 100 − 15·críticos − 5·avisos − 2·observações; conforme se ≥70 sem críticos. Elementos **opcionais** da norma (glossário, apêndice, anexo, agradecimentos, título em outro idioma) **não geram penalidade** quando ausentes.

## 6. Análises complementares

- **Léxico:** BoW/TF-IDF top-termos por seção; Jaccard vs termos de referência com indicador qualitativo.
- **Coerência semântica entre seções:** matriz de cosseno BERTimbau par a par; média por seção; seções abaixo do limiar (0.50) são sinalizadas.
- **Citações NBR 10520:** regex para diretas (aspas + autor-data, exigência de página) e indiretas; status + alertas.
- **NER:** entidades PESSOA/ORGANIZAÇÃO (LeNER-BR) com validação cruzada opcional contra metadados; termos científicos (SciERC) apenas no abstract (inglês).
- **BERTScore** por seção (bert-base-multilingual).

## 7. Feedback via LLM

`chamar_llm_analise_abnt()` monta um prompt com: score, status das seções obrigatórias, contagem de citações, **coesão semântica por seção** (médias + seções problemáticas), início do artigo e — para até 2 seções de prosa com baixa coesão — o **trecho real** com instrução de propor reescrita no formato "Antes / Sugestão de reescrita". Ordem de tentativa: HF Inference API → Ollama local (2 modelos). Todo o caminho degrada graciosamente (feedback indisponível não quebra a análise).

Nota de implementação: o modelo hospedado original (Mistral-7B-Instruct-v0.3) deixou de ser servido como chat pela HF Inference API em 2026; substituído por Llama-3.1-8B-Instruct após verificação empírica.

## 8. RAG — feedback fundamentado em normas (v1)

- **Base de conhecimento:** 3 manuais **públicos** de normalização de bibliotecas universitárias (UFC — artigos NBR 6022; PUC Minas — guia completo com NBR 10520; UFABC — guia 2026), em `PLN_BACKEND/normas_rag/`. *Ressalva metodológica: são resumos fiéis publicados abertamente, não o texto oficial da ABNT.*
- **Indexação:** ~984 chunks de ~700 caracteres (corte preferencial em fim de frase, sobreposição 150), embeddings BERTimbau-STS normalizados (mesmo modelo do pipeline — nenhum modelo extra).
- **Recuperação:** consultas derivadas **dos problemas detectados** (seção obrigatória ausente/duvidosa → regra daquela seção; citação direta sem página → regra da NBR 10520; ordem incorreta → estrutura). Artigo sem problemas recebe consultas genéricas de fallback (sem isso o contexto seria vazio e o par de comparação, inválido). Top-k por cosseno com deduplicação e score mínimo.
- **Uso:** o bloco recuperado entra no prompt com instrução de **citar a norma correspondente**. No site: checkbox "Fundamentar o feedback nas normas ABNT (RAG)".
- **Persistência pareada:** tabela `feedback_llm` com `rag_ativo` 0/1 — o mesmo artigo pode ter as duas versões. **Par de referência: artigo id=7** (`SELECT * FROM feedback_llm WHERE artigo_id=7 ORDER BY rag_ativo`): a versão sem RAG é genérica; a com RAG cita explicitamente NBR 6022, NBR 10520:2023 e capítulo do manual recuperado.

## 9. Avaliação quantitativa do detector de seções

**Metodologia — anotação assistida com verificação humana** (declarar assim no artigo): amostra de 30 documentos (`random_state=42`, reprodutível) com predições do pipeline pré-preenchidas → primeira revisão por agente de IA baseada em inventário de headings e sondas de padrão extraídos de cada texto → **arbitragem humana final** das discordâncias e células incertas (74 células revisadas). Convenções: "seção presente" = heading identificável ou equivalente inequívoco (ex.: "Caso clínico" conta como metodologia em relato de caso; "Resultados e discussão" conta para ambas; "Lições e reflexões" aceito como conclusão); coluna `titulo_outro` excluída (não verificável com confiança no texto extraído); **5 documentos excluídos por não serem artigos** (2 erratas, 2 resenhas, 1 lista de pareceristas) — decisão de anotação da autora; 3 editoriais mantidos como negativos.

**Resultados (25 artigos × 19 seções, avaliação binária de presença):**

| Agregado | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| **Micro** | **0.920** | 0.899 | 0.956 | **0.927** |
| Macro | — | 0.758 | 0.756 | 0.746 |

Por seção (F1): título 1.00 · abstract 1.00 · anexo 1.00 · agradecimentos 1.00 · autores 0.98 · referências 0.98 · data de submissão 0.97 · DOI 0.96 · resumo 0.94 · palavras-chave 0.94 · metodologia 0.90 · introdução 0.86 · conclusão 0.84 · resultados 0.77 · **discussão 0.55** · referencial teórico 0.50 (apenas 2 positivos) · implicações/glossário/apêndice: 0 positivos na amostra (acurácia 0.96-1.0 mede taxa de falso positivo).

**Análise de erros (material rico para o artigo):**
1. *Recall baixo em discussão (0.38):* o pipeline perde discussões **fundidas** em "Resultados e discussão" — o heading é atribuído a resultados.
2. *Precisão menor em introdução/conclusão (0.72-0.76):* as camadas de conteúdo (4-6) "encontram" essas seções em **editoriais** — o trade-off cobertura×precisão dos fallbacks em documentos atípicos.
3. *Erros de resumo/palavras-chave* concentram-se onde a **extração do PDF** perdeu o heading (limitação de extração, não do detector).

## 10. Limitações e trabalhos futuros

- Pesos da validação híbrida (0,15/0,85) e thresholds fixados a priori, sem calibração via busca/validação cruzada.
- Base do RAG composta por manuais universitários, não pelo texto oficial ABNT (trocável: basta substituir os PDFs e reindexar).
- **Avaliação da qualidade do feedback LLM (com vs sem RAG) ainda é qualitativa** — próximo passo natural: avaliação humana cega dos pares armazenados em `feedback_llm`.
- Amostra de avaliação com 25 artigos; seções raras (referencial teórico, implicações) com pouca estatística.
- Jaccard semântico via FastText desativado (cc.pt.300.bin requer ~13 GB de RAM na carga — inviável no hardware do projeto; decisão documentada).
- Extração de PDF (PyMuPDF) ocasionalmente perde headings ou desordena colunas — afeta detector e anotação.

## 11. Reprodutibilidade

- Execução do site: `py -3.13 -m streamlit run main.py` em `PLN front-end/` (dependências em `requirements.txt` + `python -m spacy download pt_core_news_lg`; token HF em `.env`, chave `HUGGINGFACE_API_KEY`).
- Notebook de apresentação executado ponta a ponta (25 células de código, 0 erros) com outputs salvos.
- Gabarito e avaliação: `gabarito_secoes_template.csv` (predições), `gabarito_secoes.csv` (gold revisado), `gabarito_revisao_pendente.csv.notas-usuaria.bak` (arbitragens), `gabarito_artigos_pdfs.csv` (mapeamento para PDFs); métricas reproduzíveis na seção 17 do notebook.
- Amostras determinísticas (`random_state=42`); índice RAG versionado junto aos PDFs-fonte.
