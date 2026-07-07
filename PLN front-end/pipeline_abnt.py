"""
Modulo extraido de pipeline_abnt_funcoes_oficial.ipynb (celulas de codigo,
na ordem original). Biblioteca de funcoes do pipeline ABNT: deteccao
estrutural em 6 camadas, validacao semantica hibrida (BERTimbau-STS),
analise lexica, NER, citacoes NBR 10520 e agregacao de indicadores.
"""
from typing import Optional
import os
import re
import numpy as np
import pandas as pd
import torch
import spacy
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
except Exception:
    detect = None

try:
    import ollama
    OLLAMA_DISPONIVEL = True
except Exception:
    ollama = None
    OLLAMA_DISPONIVEL = False

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_HUB_DISPONIVEL = True
except Exception:
    InferenceClient = None
    HUGGINGFACE_HUB_DISPONIVEL = False

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_DISPONIVEL = True
except Exception:
    bert_score_fn = None
    BERTSCORE_DISPONIVEL = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_DISPONIVEL = True
except Exception:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_DISPONIVEL = False

try:
    from gensim.models.fasttext import load_facebook_vectors
    GENSIM_DISPONIVEL = True
except Exception:
    load_facebook_vectors = None
    GENSIM_DISPONIVEL = False

try:
    from keybert import KeyBERT
    KEYBERT_DISPONIVEL = True
except Exception:
    KeyBERT = None
    KEYBERT_DISPONIVEL = False

# Modelo BERTimbau "cru" (mantido para compatibilidade/comparacao).
MODEL_NAME_BERTIMBAU = "neuralmind/bert-base-portuguese-cased"

# Modelo BERTimbau fine-tuned para STS (Similaridade Textual Semantica) --
# substitui o BERTimbau cru na validacao semantica hibrida, deteccao por
# conteudo (camada 5) e coerencia entre secoes. O BERTimbau cru nao foi
# treinado para similaridade textual, o que produz um espaco de embeddings
# anisotropico/comprimido (por isso os thresholds antigos eram tao baixos,
# ~0.10). O modelo STS abaixo foi fine-tuned nos benchmarks padrao de STS em
# portugues (assin, assin2, stsb_multi_mt), produzindo cossenos mais
# discriminativos e interpretaveis.
MODEL_NAME_BERTIMBAU_STS = "rufimelo/Legal-BERTimbau-sts-base"

# NER fine-tuned no LeNER-BR (corpus juridico brasileiro) -- mais preciso que
# o NER generico do spaCy para PESSOA/ORGANIZACAO em texto formal/academico
# em portugues, especialmente para afiliacoes com sigla.
MODEL_NAME_NER_LENERBR = "pierreguillou/ner-bert-base-cased-pt-lenerbr"

# NER fine-tuned no dataset SciERC (Task/Method/Metric/Material/
# OtherScientificTerm). Treinado em ingles -- por isso e aplicado apenas
# na secao "abstract" (que e a unica secao do artigo genuinamente em
# ingles, exigida pela NBR 6022), nao no corpo do artigo em portugues.
MODEL_NAME_NER_SCIERC = "RJuro/SciNERTopic"

# LLM para o feedback textual (etapa final) -- via Hugging Face Inference API
# (hospedada), com fallback para Ollama local se a chamada hospedada falhar
# (sem token, rate limit, modelo indisponivel, rede). Llama-3.1-8B-Instruct:
# o Mistral-7B-Instruct-v0.3 deixou de ser servido como chat model pela
# Inference API (testado em 2026-07: model_not_supported); o Llama 3.1 foi
# o modelo verificado funcionando com o token do projeto, com bom portugues.
MODEL_NAME_LLM_HF = "meta-llama/Llama-3.1-8B-Instruct"

CHARS_PULAR = 800
CHARS_LER   = 4500
MIN_CHARS_DETECCAO = 80

print("Imports OK")
print(f"BERTScore disponivel: {BERTSCORE_DISPONIVEL}")
print(f"sentence-transformers disponivel: {SENTENCE_TRANSFORMERS_DISPONIVEL}")
print(f"gensim disponivel: {GENSIM_DISPONIVEL}")
print(f"KeyBERT disponivel: {KEYBERT_DISPONIVEL}")
print(f"huggingface_hub disponivel: {HUGGINGFACE_HUB_DISPONIVEL}")
print(f"Ollama (fallback local) disponivel: {OLLAMA_DISPONIVEL}")

# Zero-shot classifier (camada 6 da deteccao de secoes).
# Modelo: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli -- multilingual, gratuito,
# sem gating de acesso, bom suporte a portugues via NLI (Natural Language
# Inference). Nao precisa de palavras-chave pre-definidas: classifica o
# trecho diretamente nos labels canonicos do pipeline.
# USAR_ZERO_SHOT = False desativa a camada sem remover o codigo, util para
# depuracao ou ambientes com memoria limitada.
USAR_ZERO_SHOT = True
MODEL_NAME_ZERO_SHOT = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

def _montar_prompt_llm_analise(texto_artigo: str, resultado_sistema: dict,
                               rag_contexto=None) -> str:
    """Monta o prompt do feedback textual via LLM.

    Alem do resumo numerico do sistema (score, secoes, citacoes) e do inicio
    do artigo, inclui -- quando disponiveis no resultado -- dois blocos que
    tornam o feedback cirurgico em vez de generico:
      1. Coesao semantica entre secoes (analisar_coerencia_semantica): medias
         por secao e quais ficaram abaixo do limiar.
      2. O trecho REAL das secoes com baixa coesao (ate 2 secoes, 800 chars
         cada), com instrucao para propor reescrita "Antes / Sugestao".
    Sem esses dados (ex.: BERTimbau indisponivel), o prompt degrada para a
    versao original.

    `rag_contexto`: bloco de texto opcional com trechos recuperados de
    manuais/normas ABNT (ver rag_abnt.py no front-end). Quando presente,
    entra no prompt com instrucao para fundamentar as recomendacoes na
    norma correspondente -- e o que diferencia o feedback com RAG
    (rag_ativo=1) da baseline sem RAG (rag_ativo=0).
    """
    ind = resultado_sistema.get("_indicadores", {})
    score = ind.get("score_abnt_heuristico", "n/d")
    secoes_info = resultado_sistema.get("secoes", {})
    secoes_resumo = {
        label: sec.get("status", "?")
        for label, sec in secoes_info.items()
    }
    cit = resultado_sistema.get("citacoes", {})

    coer = resultado_sistema.get("_coerencia", {}) or {}
    secoes_problemas = coer.get("secoes_problemas", []) or []
    media_por_secao = coer.get("media_por_secao", {}) or {}
    contexto_coesao = ""
    if media_por_secao:
        medias_fmt = ", ".join(f"{l}={v:.2f}" for l, v in media_por_secao.items())
        contexto_coesao = (
            f"\nCoesão semântica média por seção (cosseno BERTimbau): {medias_fmt}"
        )
        if secoes_problemas:
            limiar = coer.get("threshold", 0.5)
            contexto_coesao += (
                f"\nSeções com BAIXA coesão (média abaixo do limiar {limiar}): "
                + ", ".join(secoes_problemas)
            )

    trechos_problematicos = ""
    texto_estrut = resultado_sistema.get("_texto_estruturado", "") or ""
    secoes_raw = resultado_sistema.get("_secoes_raw", {}) or {}
    if texto_estrut and secoes_raw:
        # So secoes de PROSA do corpo (TEXTUAIS) recebem sugestao de
        # reescrita -- "reescrever" DOI, titulo, autores ou a lista de
        # referencias nao faz sentido; o sinal de coesao delas continua
        # indo no bloco acima, so nao pedimos reescrita.
        _candidatas_reescrita = [l for l in secoes_problemas if l in TEXTUAIS]
        for label in _candidatas_reescrita[:2]:
            if isinstance(secoes_raw.get(label), int):
                trecho = _extrair_texto_secao(texto_estrut, secoes_raw, label)[:800]
                if trecho.strip():
                    trechos_problematicos += (
                        f"\nTrecho da seção '{label}' (baixa coesão):\n"
                        f'\"\"\"{trecho}\"\"\"\n'
                    )

    bloco_rag = f"\n{rag_contexto}\n" if rag_contexto else ""
    instrucao_rag = (
        "- Fundamente as recomendações nos trechos de manuais ABNT fornecidos, "
        "citando a norma correspondente (NBR 6022 ou NBR 10520) quando aplicável.\n"
        if rag_contexto else ""
    )

    instrucao_reescrita = (
        "- Para cada trecho de baixa coesão fornecido acima, escolha 1-2 frases "
        "problemáticas e proponha uma reescrita no formato \"Antes: ...\" / "
        "\"Sugestão de reescrita: ...\" "
        "(as reescritas podem vir em tópicos, além dos parágrafos).\n"
        if trechos_problematicos else ""
    )

    return f"""Você é um especialista em análise de artigos científicos seguindo normas ABNT/NBR 6022.

Com base nos dados abaixo, gere uma análise textual complementar ao relatório do sistema.

Score ABNT heurístico: {score}/100
Seções obrigatórias: {secoes_resumo}
Citações detectadas: diretas={cit.get('diretas', {}).get('count', 0)}, indiretas={cit.get('indiretas', {}).get('count', 0)}{contexto_coesao}

Trecho do artigo (primeiros 1500 chars):
{str(texto_artigo)[:1500]}
{trechos_problematicos}{bloco_rag}
Instruções:
- Não contradiga os dados do sistema acima.
- Identifique pontos positivos e pontos de melhoria.
- Dê sugestões práticas e objetivas.
{instrucao_reescrita}{instrucao_rag}- Máximo de 4 parágrafos.
"""


def chamar_llm_analise_abnt(
    texto_artigo: str,
    resultado_sistema: dict,
    modelo_hf: str = MODEL_NAME_LLM_HF,
    modelo_local_principal: str = "llama3.2:3b",
    modelo_local_fallback: str = "mistral",
    hf_token: Optional[str] = None,
    rag_contexto: Optional[str] = None,
) -> dict:
    """
    Recebe o texto do artigo e o resultado do sistema heurístico/semântico.
    Gera uma análise textual complementar via LLM.

    Ordem de tentativa:
      1. Hugging Face Inference API (hospedada) -- preferência principal.
      2. Ollama local (llama3.2:3b, depois mistral) -- fallback, usado só se
         a chamada hospedada falhar (sem token, rate limit, modelo
         indisponível, problema de rede).

    `hf_token`: se não fornecido, usa a variável de ambiente HF_TOKEN.

    Retorna um dicionário com status, origem (huggingface/ollama_local),
    modelo usado e resposta da LLM.
    """
    prompt = _montar_prompt_llm_analise(texto_artigo, resultado_sistema,
                                        rag_contexto=rag_contexto)

    def _tentar_huggingface(modelo: str) -> tuple:
        """Tenta chamar o modelo via Hugging Face Inference API. Retorna (sucesso, resposta_ou_erro)."""
        if not HUGGINGFACE_HUB_DISPONIVEL:
            return False, "Biblioteca huggingface_hub não instalada. Execute: pip install huggingface_hub"
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            return False, "Token Hugging Face não configurado (defina a variável de ambiente HF_TOKEN)."
        try:
            client = InferenceClient(model=modelo, token=token)
            resposta = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
            )
            conteudo = resposta.choices[0].message.content
            if conteudo and conteudo.strip():
                return True, conteudo.strip()
            return False, "Resposta vazia do modelo."
        except Exception as e:
            return False, str(e)

    def _tentar_ollama(modelo: str) -> tuple:
        """Tenta chamar o modelo Ollama local. Retorna (sucesso, resposta_ou_erro)."""
        if not OLLAMA_DISPONIVEL:
            return False, "Biblioteca ollama não instalada. Execute: pip install ollama"
        try:
            resposta = ollama.chat(
                model=modelo,
                messages=[{"role": "user", "content": prompt}],
            )
            conteudo = resposta.get("message", {}).get("content", "")
            if conteudo and conteudo.strip():
                return True, conteudo.strip()
            return False, "Resposta vazia do modelo."
        except Exception as e:
            return False, str(e)

    erros = {}

    # 1) Hugging Face Inference API -- preferência principal
    ok, resp = _tentar_huggingface(modelo_hf)
    if ok:
        return {
            "usada": True,
            "modelo": modelo_hf,
            "origem": "huggingface",
            "status": "ok",
            "resposta": resp,
        }
    erros["huggingface"] = resp
    print(f"  LLM Hugging Face '{modelo_hf}' falhou: {resp}. Tentando fallback local (Ollama)...")

    # 2) Ollama local -- fallback (só chega aqui se a Hugging Face falhar)
    ok2, resp2 = _tentar_ollama(modelo_local_principal)
    if ok2:
        return {
            "usada": True,
            "modelo": modelo_local_principal,
            "origem": "ollama_local",
            "status": "ok_fallback_local",
            "resposta": resp2,
        }
    erros["ollama_local_principal"] = resp2

    ok3, resp3 = _tentar_ollama(modelo_local_fallback)
    if ok3:
        return {
            "usada": True,
            "modelo": modelo_local_fallback,
            "origem": "ollama_local",
            "status": "ok_fallback_local_2",
            "resposta": resp3,
        }
    erros["ollama_local_fallback"] = resp3

    return {
        "usada": False,
        "modelo": None,
        "origem": None,
        "status": "erro",
        "resposta": None,
        "erro": erros,
    }


print("chamar_llm_analise_abnt carregada.")
print(f"  Principal (hospedado) : Hugging Face -> {MODEL_NAME_LLM_HF}")
print("  Fallback (local)      : Ollama -> llama3.2:3b, depois mistral")

def carregar_modelo_spacy(model_name: str = "pt_core_news_lg"):

    try:

        return spacy.load(model_name)

    except OSError:

        print(f"Modelo spaCy '{model_name}' nao encontrado. Execute: python -m spacy download {model_name}")

        return None



def carregar_bertimbau(model_name: str = MODEL_NAME_BERTIMBAU):

    try:

        tok = AutoTokenizer.from_pretrained(model_name)

        mdl = AutoModel.from_pretrained(model_name)

        mdl.eval()

        print(f"BERTimbau carregado: {model_name}")

        return tok, mdl

    except Exception as e:

        print(f"BERTimbau nao carregado: {e}")

        return None, None


def carregar_bertimbau_sts(model_name: str = MODEL_NAME_BERTIMBAU_STS):
    """Carrega o BERTimbau fine-tuned para STS (sentence-transformers).

    Retorna (None, modelo) -- por convencao, `tokenizer=None` sinaliza pros
    demais componentes do pipeline (`_chunk_encode` etc.) que `modelo` e um
    objeto SentenceTransformer (tokenizacao e pooling ja embutidos), em vez
    de um par (AutoTokenizer, AutoModel) cru.
    """
    if not SENTENCE_TRANSFORMERS_DISPONIVEL:
        print("sentence-transformers nao instalado. Execute: pip install sentence-transformers")
        return None, None
    try:
        mdl = SentenceTransformer(model_name)
        print(f"BERTimbau-STS carregado: {model_name}  (dim={mdl.get_sentence_embedding_dimension()})")
        return None, mdl
    except Exception as e:
        print(f"BERTimbau-STS nao carregado: {e}")
        return None, None


def carregar_fasttext_pt(caminho_modelo: str):
    """Carrega vetores FastText pre-treinados em portugues (formato .bin do NILC/fastText.cc).

    Usa gensim.load_facebook_vectors, que preserva o lookup por subpalavras do
    FastText (cobre palavras fora do vocabulario, ao contrario de vetores
    word2vec/.txt). Recomendado: vetores pre-treinados (NILC ou fastText.cc),
    nao treinar do zero -- o corpus de artigos e pequeno demais pra aprender
    uma geometria vetorial estavel.
    """
    if not GENSIM_DISPONIVEL:
        print("gensim nao instalado. Execute: pip install gensim")
        return None
    if not os.path.isfile(caminho_modelo):
        print(f"Arquivo de vetores FastText nao encontrado: {caminho_modelo}")
        return None
    try:
        wv = load_facebook_vectors(caminho_modelo)
        print(f"FastText PT carregado: {caminho_modelo}  (dim={wv.vector_size}, vocab={len(wv)})")
        return wv
    except Exception as e:
        print(f"FastText PT nao carregado: {e}")
        return None


def carregar_keybert(sts_model=None):
    """Carrega o KeyBERT, reaproveitando o modelo BERTimbau-STS ja carregado
    como motor de embeddings (evita carregar mais um modelo do zero)."""
    if not KEYBERT_DISPONIVEL:
        print("KeyBERT nao instalado. Execute: pip install keybert")
        return None
    try:
        kw_model = KeyBERT(model=sts_model) if sts_model is not None else KeyBERT()
        print("KeyBERT carregado" + (" (usando BERTimbau-STS)" if sts_model is not None else " (modelo padrao)"))
        return kw_model
    except Exception as e:
        print(f"KeyBERT nao carregado: {e}")
        return None


def carregar_ner_lenerbr(model_name: str = MODEL_NAME_NER_LENERBR):
    """Carrega o pipeline de NER fine-tuned no LeNER-BR (PESSOA/ORGANIZACAO/LOCAL)."""
    try:
        from transformers import pipeline as hf_pipeline
        tok = AutoTokenizer.from_pretrained(model_name)
        ner_pipe = hf_pipeline(
            "ner", model=model_name, tokenizer=tok, aggregation_strategy="simple"
        )
        print(f"NER LeNER-BR carregado: {model_name}")
        return ner_pipe
    except Exception as e:
        print(f"NER LeNER-BR nao carregado: {e}")
        return None


def carregar_ner_scierc(model_name: str = MODEL_NAME_NER_SCIERC):
    """Carrega o pipeline de NER fine-tuned no SciERC (Task/Method/Metric/
    Material/OtherScientificTerm). Em ingles -- usar apenas na secao abstract."""
    try:
        from transformers import pipeline as hf_pipeline
        ner_pipe = hf_pipeline(
            "ner", model=model_name, aggregation_strategy="simple"
        )
        print(f"NER SciERC carregado: {model_name}")
        return ner_pipe
    except Exception as e:
        print(f"NER SciERC nao carregado: {e}")
        return None


def carregar_zero_shot(
    model_name: str = MODEL_NAME_ZERO_SHOT,
    usar: bool = True,
):
    """Carrega o pipeline de zero-shot classification para a camada 6 de
    deteccao de secoes.

    Modelo: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
      - Multilingual (cobre portugues nativamente via NLI).
      - Sem gating de acesso (diferente de varios modelos Llama/Mistral).
      - Classifica um trecho de texto diretamente nos labels canonicos do
        pipeline, sem precisar de nenhuma palavra-chave pre-definida.

    Se `usar=False` (USAR_ZERO_SHOT=False no topo do notebook), retorna None
    sem carregar nada -- util pra desativar a camada sem remover o codigo,
    por exemplo em ambientes com pouca memoria ou para depuracao.
    """
    if not usar:
        print("Zero-shot classifier desativado (USAR_ZERO_SHOT=False).")
        return None
    try:
        from transformers import pipeline as hf_pipeline
        zs_pipe = hf_pipeline(
            "zero-shot-classification",
            model=model_name,
            multi_label=False,
        )
        print(f"Zero-shot classifier carregado: {model_name}")
        return zs_pipe
    except Exception as e:
        print(f"Zero-shot classifier nao carregado: {e}")
        return None



def extrair_texto_pdf(caminho_pdf: str) -> str:

    """Extrai texto de PDF com PyMuPDF (fitz), em linha com a arquitetura do sistema."""

    if not os.path.isfile(caminho_pdf):

        return ""

    try:

        import fitz

        paginas = []

        with fitz.open(caminho_pdf) as pdf:

            for pagina in pdf:

                paginas.append(pagina.get_text("text") or "")

        return "\n".join(paginas).strip()

    except Exception as e:

        print(f"Erro ao extrair PDF com PyMuPDF: {e}")

        return ""



def extrair_texto_docx(caminho_docx: str) -> str:

    """Extrai texto de arquivos .docx usando python-docx."""

    if not os.path.isfile(caminho_docx):

        return ""

    try:

        from docx import Document

        doc = Document(caminho_docx)

        paragrafos = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]

        return "\n".join(paragrafos).strip()

    except Exception as e:

        print(f"Erro ao extrair DOCX: {e}")

        return ""



def extrair_texto_arquivo(caminho_arquivo: str) -> str:

    """Extrai texto de PDF ou DOCX conforme a extensao do arquivo."""

    if not os.path.isfile(caminho_arquivo):

        return ""

    ext = os.path.splitext(caminho_arquivo)[1].lower()

    if ext == ".pdf":

        return extrair_texto_pdf(caminho_arquivo)

    if ext == ".docx":

        return extrair_texto_docx(caminho_arquivo)

    print(f"Formato nao suportado para extracao: {ext}")

    return ""




def extrair_texto_arquivo_usuario(caminho_arquivo: str) -> str:
    """
    Recebe o caminho de um arquivo PDF, DOCX ou TXT.
    Extrai o texto e retorna uma string.
    Essa função será usada futuramente pela interface Streamlit.
    """
    if not os.path.isfile(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    ext = os.path.splitext(caminho_arquivo)[1].lower()

    if ext == ".pdf":
        texto = extrair_texto_pdf(caminho_arquivo)
        if not texto.strip():
            raise ValueError(
                "Nenhum texto extraído do PDF. "
                "Se for PDF escaneado, OCR será necessário (extensão futura)."
            )
        return texto

    if ext == ".docx":
        texto = extrair_texto_docx(caminho_arquivo)
        if not texto.strip():
            raise ValueError("Nenhum texto extraído do DOCX.")
        return texto

    if ext == ".txt":
        with open(caminho_arquivo, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    raise ValueError(
        f"Formato '{ext}' não suportado. "
        "Use: .pdf, .docx ou .txt"
    )


def analisar_arquivo_usuario(
    caminho_arquivo: str,
    usar_llm: bool = False,
    nlp=None,
    tokenizer=None,
    model=None,
    stopwords_pt=None,
    ft_model=None,
    kw_model=None,
    ner_pipeline_lenerbr=None,
    ner_pipeline_scierc=None,
    zs_pipeline=None,
) -> dict:
    """
    Recebe um arquivo do usuário em PDF, DOCX ou TXT.
    Extrai o texto.
    Executa o pipeline ABNT.
    Gera a saída padrão com gerar_saida_analise.
    Se usar_llm=True, chama a LLM e adiciona o resultado no campo 'llm'.
    Retorna a saída final consolidada.

    Essa função será usada depois pela interface Streamlit.
    """
    texto = extrair_texto_arquivo_usuario(caminho_arquivo)

    resultado = analisar_artigo(
        texto,
        nlp=nlp,
        tokenizer=tokenizer,
        model=model,
        stopwords_pt=stopwords_pt,
        ft_model=ft_model,
        kw_model=kw_model,
        ner_pipeline_lenerbr=ner_pipeline_lenerbr,
        ner_pipeline_scierc=ner_pipeline_scierc,
        zs_pipeline=zs_pipeline,
    )

    saida = gerar_saida_analise(resultado)

    if usar_llm:
        saida["llm"] = chamar_llm_analise_abnt(texto, resultado)
    else:
        saida["llm"] = {
            "usada": False,
            "modelo": None,
            "status": "desativada",
            "resposta": None,
        }

    return saida


print("extrair_texto_arquivo_usuario + analisar_arquivo_usuario carregadas.")
print("  Prontas para uso futuro com interface Streamlit.")


STOP_PT = {'de','da','do','das','dos','e','em','para','com','na','no',
           'que','como','por','uma','um','os','as','ao','aos','nao'}
STOP_EN = {'the','and','of','to','in','for','with','on','is','are',
           'was','were','that','this','from','by','as','an','or','be'}
STOP_ES = {'de','la','el','y','en','para','con','los','las','que',
           'por','una','un','como','del','al','es','son','se','su'}

def _linhas_relevantes_para_idioma(texto, min_palavras_linha=8, max_linhas=100):
    linhas = []
    for linha in str(texto).splitlines():
        l = re.sub(r"\s+", " ", linha).strip()
        if not l:
            continue
        if re.match(r"(?i)^(abstract|resumo|title|titulo|keywords?|palavras[- ]?chave)\s*:?$", l):
            continue
        if len(l.split()) >= min_palavras_linha:
            linhas.append(l)
        if len(linhas) >= max_linhas:
            break
    return linhas

def dividir_em_blocos(texto, tamanho_bloco=1000, min_palavras=60,
                      min_palavras_linha=8, linhas_por_bloco=6):
    linhas = _linhas_relevantes_para_idioma(texto, min_palavras_linha)
    blocos_linha = []
    if len(linhas) >= linhas_por_bloco:
        for i in range(0, len(linhas), linhas_por_bloco):
            bloco = " ".join(linhas[i:i+linhas_por_bloco]).strip()
            if len(bloco.split()) >= min_palavras:
                blocos_linha.append(bloco)
    if blocos_linha:
        return blocos_linha
    palavras = str(texto).split()
    return [" ".join(palavras[i:i+tamanho_bloco])
            for i in range(0, len(palavras), tamanho_bloco)
            if len(palavras[i:i+tamanho_bloco]) >= min_palavras]

def _score_sw(tokens, vocab):
    return sum(1 for t in tokens if t in vocab)

def detectar_idioma_bloco(texto, usar_langdetect_fallback=True):
    tokens = re.findall(r"\b[\wÀ-ÖØ-öø-ÿ]+\b", str(texto).lower())
    if len(tokens) < 20:
        return "indefinido"
    scores = {'pt': _score_sw(tokens, STOP_PT),
              'en': _score_sw(tokens, STOP_EN),
              'es': _score_sw(tokens, STOP_ES)}
    idioma = max(scores, key=scores.get)
    if scores[idioma] == 0 and usar_langdetect_fallback and detect is not None:
        try:
            ld = detect(texto)
            if ld in ('pt','en','es'):
                return ld
        except Exception:
            pass
        return "indefinido"
    return idioma

def aplicar_filtro_idioma_percentual(df, coluna_texto='texto', limiar_pt=0.70,
                                     min_blocos_pt=2, tamanho_bloco=1000, min_palavras=60):
    if coluna_texto not in df.columns:
        raise ValueError(f'Coluna nao encontrada: {coluna_texto}')
    registros = []
    for idx, row in df.iterrows():
        texto = str(row[coluna_texto]) if pd.notna(row[coluna_texto]) else ""
        blocos = dividir_em_blocos(texto, tamanho_bloco=tamanho_bloco, min_palavras=min_palavras)
        preds  = [detectar_idioma_bloco(b) for b in blocos] if blocos else []
        total  = len(preds)
        n_pt   = sum(p == 'pt' for p in preds)
        n_en   = sum(p == 'en' for p in preds)
        n_es   = sum(p == 'es' for p in preds)
        pct_pt = n_pt / total if total else 0.0
        pct_en = n_en / total if total else 0.0
        pct_es = n_es / total if total else 0.0
        idioma_pred = max({'pt': pct_pt, 'en': pct_en, 'es': pct_es}, key=lambda k: {'pt': pct_pt, 'en': pct_en, 'es': pct_es}[k]) if total else 'indefinido'
        manter = (pct_pt >= limiar_pt) and (n_pt >= min_blocos_pt)
        registros.append({
            'indice_original': idx,
            'idioma_predominante': idioma_pred,
            'pct_pt': round(pct_pt, 4),
            'pct_en': round(pct_en, 4),
            'pct_es': round(pct_es, 4),
            'n_blocos': total,
            'blocos_pt': n_pt,
            'manter_dataset_pt': manter,
        })
    diag = pd.DataFrame(registros).set_index('indice_original')
    df_diag = df.join(diag, how='left')
    df_pt = df_diag[df_diag['manter_dataset_pt'] == True].copy()
    df_nao_pt = df_diag[df_diag['manter_dataset_pt'] != True].copy()
    return df_diag, df_pt, df_nao_pt


def diagnosticar_idioma_por_percentual(texto, limiar_pt=0.70, min_blocos_pt=2,
                                       tamanho_bloco=1000, min_palavras=60):
    """Versao de aplicar_filtro_idioma_percentual para um unico texto (nao DataFrame).

    Usada por analisar_artigo() e _resultado_analise_vazio() para classificar o
    idioma predominante de um artigo individual, com a mesma logica de blocos
    aplicada ao dataset completo.
    """
    blocos = dividir_em_blocos(texto, tamanho_bloco=tamanho_bloco, min_palavras=min_palavras)
    preds  = [detectar_idioma_bloco(b) for b in blocos] if blocos else []
    total  = len(preds)
    n_pt   = sum(p == 'pt' for p in preds)
    n_en   = sum(p == 'en' for p in preds)
    n_es   = sum(p == 'es' for p in preds)
    pct_pt = n_pt / total if total else 0.0
    pct_en = n_en / total if total else 0.0
    pct_es = n_es / total if total else 0.0
    idioma_pred = (
        max({'pt': pct_pt, 'en': pct_en, 'es': pct_es},
            key=lambda k: {'pt': pct_pt, 'en': pct_en, 'es': pct_es}[k])
        if total else 'indefinido'
    )
    manter = (pct_pt >= limiar_pt) and (n_pt >= min_blocos_pt)
    return {
        'idioma_predominante': idioma_pred,
        'pct_pt': round(pct_pt, 4),
        'pct_en': round(pct_en, 4),
        'pct_es': round(pct_es, 4),
        'n_blocos': total,
        'blocos_pt': n_pt,
        'manter_dataset_pt': manter,
    }


print("Funcoes de idioma carregadas.")

def _normalizar_token(tok: str) -> str:
    mapa = str.maketrans("áéíóúâêîôûãõàçü", "aeiouaeiouaoacu")
    return tok.lower().translate(mapa)

def preprocessar_texto_lexico(texto, stopwords_lexico=None, min_len=3, manter_nao=True):
    """Limpeza reforcada exclusiva para BoW e TF-IDF."""
    _EDITORIAIS = {
        "id","doi","issn","pmid","isbn","eissn","epub","rev","bras","braz","journal",
        "article","artigo","available","disponivel","org","www","http",
        "https","pdf","vol","volume","num","numero","pp","pag","pagina",
        "fig","figure","figura","table","tabela","fonte","copyright",
        "license","licenca","open","access","creative","commons","et","al",
        "sim","nao","ou","and","or","of","the","deste","desta","neste",
        "nesta","trabalho","estudo","universidade","federal"
    }
    if not isinstance(texto, str):
        texto = str(texto) if texto is not None else ""
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\S+@\S+\.\S+", " ", texto)
    texto = re.sub(r"\b(doi|issn|pmid|isbn|eissn)\s*[:\-]?\s*[\d\.\-/Xx]+",
                   " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"10\.\d{4,}/\S+", " ", texto)
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúâêîôûãõàçü\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    sw = set(stopwords_lexico) if stopwords_lexico else set()
    if manter_nao:
        sw.discard("nao")
        sw.discard("não")
    tokens = []
    for tok in texto.split():
        if len(tok) < min_len or tok in _EDITORIAIS or tok in sw:
            continue
        tok = _normalizar_token(tok)
        if tok in _EDITORIAIS or tok in sw or len(tok) < min_len:
            continue
        tokens.append(tok)
    return " ".join(tokens)

def preparar_texto_para_estrutura(texto):
    if not isinstance(texto, str):
        texto = str(texto) if texto is not None else ""
    texto = texto.replace("\r\n","\n").replace("\r","\n").replace("\xad","")
    texto = re.sub(r"([A-Za-zÀ-ÿ]+)\s*-\s*\n\s*([A-Za-zÀ-ÿ]+)", r"\1\2", texto)
    texto = re.sub(r"\t+| +", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()

def preparar_texto_para_ner(texto):
    if not isinstance(texto, str):
        texto = str(texto) if texto is not None else ""
    texto = texto.replace("\r\n","\n").replace("\r","\n").replace("\xad","")
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\S+@\S+\.\S+", " ", texto)
    texto = re.sub(r"([A-Za-zÀ-ÿ]+)\s*-\s*\n\s*([A-Za-zÀ-ÿ]+)", r"\1\2", texto)
    texto = re.sub(r"\t+| +", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()

print("Pre-processamento carregado.")

PRE_TEXTUAIS = {
    "titulo":        r"(?m)^[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÇÜ][^\n]{5,100}$",
    "titulo_outro":  r"(?i)(title|titre)\s*[:\-]?",
    "autores":       r"(?i)(autor(es)?|author(s)?)\s*[:\-]?",
    "resumo":        r"(?i)^\s*resumo\s*([:\-–—]|$)",
    "abstract":      r"(?i)^\s*(abstract|résumé|resumen)\s*([:\-–—]|$)",
    "palavras_chave":r"(?i)(palavras[- ]chave|keywords?)\s*[:\-]?",
    "data_submissao":r"(?i)(receb|submet|submiss|aprovad).{0,40}\d{4}",
    "doi_disponib":  r"(?i)(doi|disponível|available|http)",
}
TEXTUAIS = {
    "introducao":        r"(?i)^\s*\d*\.?\s*introdu[cç][aã]o\s*$",
    "referencial_teorico":r"(?i)^\s*\d*\.?\s*(referencial\s+te[oó]rico|revis[aã]o\s+de\s+literatura|fundamenta[cç][aã]o\s+te[oó]rica|marco\s+te[oó]rico)\s*$",
    "metodologia":       r"(?i)^\s*\d*\.?\s*(metodologia|m[eé]todo(s)?|materiais?\s*e\s*m[eé]todo(s)?)\s*$",
    "resultados":        r"(?i)^\s*\d*\.?\s*(resultados?(\s*e\s*discuss[õo][eê]s?)?)\s*$",
    "discussao":         r"(?i)^\s*\d*\.?\s*discuss[aã]o\s*$",
    "implicacoes":       r"(?i)^\s*\d*\.?\s*implica[cç][õo]es\b",
    "conclusao":         r"(?i)^\s*\d*\.?\s*(conclus[aã]o|conclus[õo]es|considera[cç][õo]es\s+finais)\s*$",
}
POS_TEXTUAIS = {
    "referencias":    r"(?i)^\s*(refer[eê]ncias(\s+bibliogr[aá]ficas)?)\s*$",
    "agradecimentos": r"(?i)^\s*agradecimentos?\s*$",
    "apendice":       r"(?i)^\s*ap[eê]ndice\s*",
    "anexo":          r"(?i)^\s*anexo\s*",
    "glossario":      r"(?i)^\s*gloss[aá]rio\s*$",
}

ORDEM_NBR6022 = [
    "titulo","autores","resumo","palavras_chave","abstract","titulo_outro",
    "data_submissao","doi_disponib","introducao","referencial_teorico",
    "metodologia","resultados","discussao","implicacoes","conclusao",
    "referencias","glossario","apendice","anexo","agradecimentos"
]

# Secoes das obrigacoes NBR 6022 que receberao validacao semantica hibrida
SECOES_OBRIGATORIAS_NBR = {"introducao", "metodologia", "resultados", "conclusao", "referencias"}

# Severidade base usada apenas para secoes nao cobertas pela validacao semantica
SEVERIDADE_SECAO_BASE = {
    "titulo":"critico","autores":"critico","resumo":"critico","palavras_chave":"critico",
    "abstract":"aviso","titulo_outro":"aviso","data_submissao":"observacao","doi_disponib":"observacao",
    "introducao":"critico","referencial_teorico":"observacao","metodologia":"aviso",
    "resultados":"aviso","discussao":"aviso","implicacoes":"observacao",
    "conclusao":"critico","referencias":"critico","agradecimentos":"observacao",
    "apendice":"observacao","anexo":"observacao","glossario":"observacao"
}

# Elementos opcionais da NBR 6022: a ausencia deles NAO gera aviso nem
# penalidade de score (quando presentes, continuam reportados normalmente).
SECOES_OPCIONAIS_NBR = {
    "glossario", "apendice", "anexo", "agradecimentos", "titulo_outro",
}

CANDIDATOS_FUZZY = {
    "resumo":        ["Resumo", "Resumo expandido", "Resumo estruturado"],
    "abstract":      ["Abstract", "Résumé", "Resumen", "Summary"],
    "introducao":    ["Introdução", "Contextualização", "Apresentação",
                      "Contexto", "Antecedentes"],
    "metodologia":   ["Metodologia", "Método", "Materiais e Métodos",
                      "Material e Método", "Materiais e métodos",
                      # relatos de caso e estudos clínicos
                      "Caso clínico", "Relato de caso", "Apresentação do caso",
                      "Descrição do caso", "Relato", "Caso",
                      # estudos observacionais/epidemiológicos
                      "Delineamento do estudo", "Delineamento",
                      "Abordagem metodológica", "Procedimentos",
                      "Procedimentos metodológicos", "Coleta de dados",
                      # computação/engenharia
                      "Abordagem experimental", "Experimentos",
                      "Configuração experimental", "Implementação",
                      "Desenvolvimento", "Proposta"],
    "resultados":    ["Resultados", "Resultados e Discussão",
                      "Achados", "Findings", "Resultados obtidos",
                      # relatos de caso
                      "Evolução", "Evolução clínica", "Seguimento",
                      "Desfecho", "Outcomes"],
    "discussao":     ["Discussão", "Discussão dos resultados",
                      "Análise e discussão", "Discussão e conclusão"],
    "conclusao":     ["Conclusão", "Considerações finais",
                      "Conclusões", "Considerações", "Fechamento",
                      "Conclusão e perspectivas"],
    "referencias":   ["Referências", "Referências bibliográficas",
                      "Referência", "Bibliography", "References"],
    "palavras_chave":["Palavras-chave", "Keywords", "Descritores",
                      "Unitermos", "Termos indexadores"],
}

# Frases de referencia para validacao semantica via BERTimbau-STS.
# Cada entrada usa multiplos contextos separados por ponto e virgula pra
# o embedding medio cobrir mais variacoes de vocabulario por area/tipo de artigo.
CONTEUDO_REFERENCIA = {
    "resumo": (
        "Este artigo apresenta brevemente os objetivos, métodos e principais resultados. "
        "O resumo descreve de forma concisa o problema investigado e as conclusões."
    ),
    "abstract": (
        "This paper briefly presents the objectives, methods and main results. "
        "The abstract summarizes the research problem, methodology and findings."
    ),
    "introducao": (
        "Este trabalho apresenta o contexto, a motivação e os objetivos da pesquisa. "
        "A introdução discute o problema de pesquisa, a justificativa e os objetivos do estudo. "
        "O presente estudo tem como objetivo investigar o fenômeno descrito na literatura."
    ),
    "referencial_teorico": (
        "A revisão da literatura aborda os principais conceitos e teorias do estudo. "
        "Os autores discutem os trabalhos relacionados e o embasamento teórico da pesquisa."
    ),
    "metodologia": (
        # pesquisa clássica
        "Os dados foram coletados e analisados utilizando métodos quantitativos e qualitativos. "
        "A amostra foi composta por participantes selecionados segundo critérios de inclusão. "
        # relato de caso clínico (ex.: oftalmologia, cardiologia, etc.)
        "O paciente foi submetido ao procedimento cirúrgico e acompanhado no pós-operatório. "
        "Foi realizada a cirurgia e o exame clínico conforme protocolo institucional aprovado. "
        "O caso clínico foi conduzido seguindo os critérios éticos e o consentimento foi obtido. "
        # estudos observacionais/epidemiológicos
        "O delineamento do estudo foi transversal, retrospectivo ou prospectivo. "
        # computação/engenharia
        "A abordagem experimental envolveu a implementação do algoritmo e a configuração dos testes."
    ),
    "resultados": (
        "Os resultados obtidos demonstram que a hipótese foi confirmada pelos dados coletados. "
        "Os achados indicam que houve melhora significativa após a intervenção realizada. "
        "A análise estatística revelou diferenças significativas entre os grupos estudados. "
        # relato de caso
        "A paciente evoluiu bem e relatou melhora substancial após o procedimento realizado. "
        "O exame pós-operatório revelou resultados satisfatórios com redução dos sintomas."
    ),
    "discussao": (
        "Os resultados são discutidos em relação à literatura e às implicações para o campo. "
        "Os achados corroboram estudos anteriores e apontam limitações da pesquisa."
    ),
    "implicacoes": (
        "As implicações práticas e teóricas dos resultados são discutidas. "
        "O estudo contribui para o campo com recomendações para pesquisas futuras."
    ),
    "conclusao": (
        "Conclui-se que os objetivos foram alcançados, com perspectivas para trabalhos futuros. "
        "O estudo demonstrou que a intervenção foi eficaz e os resultados foram satisfatórios. "
        "As conclusões apontam para a necessidade de novos estudos e aplicações práticas."
    ),
    "referencias": (
        "SILVA, João. Título do trabalho. Revista Brasileira, v. 1, n. 1, p. 1-10, 2020. "
        "Lista de referências bibliográficas no formato ABNT, APA ou Vancouver."
    ),
    "agradecimentos": (
        "Os autores agradecem ao apoio financeiro e à colaboração dos participantes. "
        "Este trabalho foi financiado pela agência de fomento e aprovado pelo comitê de ética."
    ),
}

print("Padroes e constantes carregados.")

PISTAS_SEMANTICAS_SECOES = {
    "introducao": [
        "objetivo","objetivos","contexto","problema","literatura",
        "pesquisa","investigar","analisar","tema","contexto historico",
        "no brasil","no mundo","area de saude","em educacao","no pais",
        "desafios","motiva","justifica","relevancia","importancia",
        "justificativa","este trabalho","este estudo","presente estudo",
        "presente pesquisa","objetiva-se","tem como objetivo","busca analisar",
        "nos ultimos anos","torna-se necessario","e necessario","lacuna",
        "problematiza","tendencia","contextualiza","introduz","apresenta",
    ],
    "metodologia": [
        "metodo","metodologia","procedimento","procedimentos",
        "amostra","coleta","dados","analise","questionario",
        "criterio","participantes","instrumento",
        "foram coletados","foi utilizado","utilizou-se","estatistica",
        "delineamento","estudo transversal","estudo longitudinal",
        "ensaio clinico","revisao sistematica","coorte","caso-controle",
    ],
    "resultados": [
        "resultado","resultados","observou-se","foi observado",
        "foram observados","apresentou","apresentaram","tabela",
        "figura","percentual","prevalencia",
        "diferenca","associacao",
        "p <","valor de p","significativo","media","desvio",
        "frequencia","proporcao","taxa",
    ],
    "discussao": [
        "discussao","os achados","estes achados",
        "literatura","estudos anteriores","corrobora","corroboram",
        "semelhante","divergente","sugere","sugerem","explica",
        "pode ser explicado","comparado","em comparacao",
        "de acordo com","em concordancia","em consonancia",
        "por outro lado","entretanto","no entanto",
        "limitacao","limitacoes","hipotese","evidencia",
        "consistente com","diferentemente de","contrasta com",
    ],
    "conclusao": [
        "conclusao","conclui-se","concluimos",
        "portanto","assim","dessa forma","recomenda-se",
        "diante do exposto","diante desse contexto",
        "sugere-se","os resultados indicam","evidenciam",
        "em sintese","por fim","identificou",
        "este estudo apresentou","em resumo","em conclusao",
        "foi possivel observar","foi possivel constatar",
        "o objetivo deste estudo foi","pesquisas futuras","estudos futuros",
        "contribui para","contribuicao deste estudo",
    ],
    "referencias": [
        "et al","disponivel em","acesso em","doi","issn","v.","n.",
        "p.","ed.","editora","revista","journal","anais","in:",
        "apud","idem","ibid",
    ],
}


def _score_heuristica_secao(texto_secao: str, label: str) -> float:
    """Retorna score 0-1 baseado em pistas lexicais para o label informado."""
    texto_l = str(texto_secao).lower()
    texto_l = re.sub(r"\s+", " ", texto_l).strip()
    # Normaliza acentos para casar com as pistas em PISTAS_SEMANTICAS_SECOES,
    # que estao escritas sem acento (ex.: "discussao", "hipotese", "conclusao").
    _mapa_acentos = str.maketrans("áéíóúâêîôûãõàçü", "aeiouaeiouaoacu")
    texto_l = texto_l.translate(_mapa_acentos)
    pistas = PISTAS_SEMANTICAS_SECOES.get(label, [])
    if not pistas:
        return 0.0
    encontradas = sum(
        1 for p in pistas
        if re.search(r"\b" + re.escape(p.lower()) + r"\b", texto_l)
    )
    return round(encontradas / len(pistas), 4)


print("Pistas semanticas e heuristica carregadas.")

def _chunk_encode_sentence_transformer(texto: str, model, janela_palavras: int = 90, sobreposicao: int = 25) -> np.ndarray:
    """Janela deslizante em nivel de palavras para modelos SentenceTransformer.

    Modelos sentence-transformers tem max_seq_length proprio (geralmente
    menor que o BERT base, ~128-256 tokens), e ja embutem tokenizacao e
    pooling em `model.encode()` -- por isso a janela aqui e em palavras, nao
    em tokens, e nao precisamos lidar com CLS/SEP manualmente.
    """
    palavras = texto.split()
    if not palavras:
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
    janelas, inicio = [], 0
    passo = max(1, janela_palavras - sobreposicao)
    while inicio < len(palavras):
        fim = min(inicio + janela_palavras, len(palavras))
        janelas.append(" ".join(palavras[inicio:fim]))
        if fim == len(palavras):
            break
        inicio += passo
    embs = model.encode(janelas, show_progress_bar=False, convert_to_numpy=True)
    return np.mean(embs, axis=0)


def _chunk_encode(texto: str, tokenizer, model, stride: int = 256) -> np.ndarray:
    """Codifica texto longo por janela deslizante e retorna embedding medio.

    Suporta dois tipos de `model`, pelo valor de `tokenizer`:
      - tokenizer=None  -> `model` e um SentenceTransformer (STS), que ja
        embute tokenizacao e pooling; usa janela deslizante em palavras.
      - tokenizer != None -> caminho original (AutoTokenizer + AutoModel
        "crus"), com janela deslizante em tokens e mean-pooling manual.
    """
    if tokenizer is None:
        return _chunk_encode_sentence_transformer(texto, model)

    ids = tokenizer.encode(texto, add_special_tokens=False)
    if len(ids) == 0:
        return np.zeros(768, dtype=np.float32)
    max_tokens, janelas, inicio = 510, [], 0
    while inicio < len(ids):
        fim = min(inicio + max_tokens, len(ids))
        janelas.append([tokenizer.cls_token_id] + ids[inicio:fim] + [tokenizer.sep_token_id])
        if fim == len(ids):
            break
        inicio += stride
    emb = []
    for chunk in janelas:
        tensor = torch.tensor([chunk])
        with torch.no_grad():
            out = model(tensor).last_hidden_state[0]
        if out.shape[0] > 2:
            out = out[1:-1]
        emb.append(out.mean(dim=0).cpu().numpy())
    return np.mean(emb, axis=0)


def _cosseno(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _score_bert_secao(texto_secao: str, label: str, tokenizer, model) -> float:
    """Similaridade cosseno entre embedding do texto da secao e frase-referencia do label."""
    frase_ref = CONTEUDO_REFERENCIA.get(label)
    if not frase_ref or model is None:
        return np.nan
    texto_curto = str(texto_secao)[:2000]
    if len(texto_curto.split()) < 10:
        return np.nan
    emb_secao = _chunk_encode(texto_curto, tokenizer, model)
    emb_ref   = _chunk_encode(frase_ref, tokenizer, model)
    return round(_cosseno(emb_secao, emb_ref), 4)


print("BERTimbau scoring carregado.")

# -- Pesos ajustaveis por parametro -------------------------------------------
# DECISAO: pesos fixados a priori em 0.15 (heuristica) / 0.85 (BERT), por causa
# do prazo -- a calibracao via RandomizedSearchCV + validacao cruzada sobre o
# dataset anotado manualmente NAO foi feita. Os thresholds tambem permanecem
# como estimativa inicial (nao calibrados), escolhidos para a escala de
# similaridade do BERTimbau-STS (MODEL_NAME_BERTIMBAU_STS), que e diferente
# da escala do BERTimbau cru usado antes (cossenos mais discriminativos,
# faixa mais alta e mais espalhada). Documentar como limitacao/trabalho
# futuro na dissertacao.
PESO_HEURISTICA = 0.15
PESO_BERT       = 0.85

THRESHOLDS_HIBRIDO = {
    "introducao":  0.30,
    "metodologia": 0.30,
    "resultados":  0.30,
    "conclusao":   0.30,
    "referencias": 0.25,
}
THRESHOLD_PADRAO = 0.30

# Limiares do indicador lexical (Bloco 2)
LIMIAR_LEXICO_ADEQUADO = 0.50
LIMIAR_LEXICO_PARCIAL  = 0.25

STATUS_CONTEM             = "Contém seção"
STATUS_CONTEM_OBS         = "Contém seção com observação"
STATUS_REQUER_REVISAO     = "Requer revisão"
STATUS_NAO_CONTEM         = "Não contém seção"

PENALIDADE_STATUS = {
    STATUS_NAO_CONTEM:     15,
    STATUS_REQUER_REVISAO:  5,
    STATUS_CONTEM_OBS:      2,
    STATUS_CONTEM:          0,
}


def _status_para_nivel_legado(status: str) -> str:
    return {
        STATUS_CONTEM:         "ok",
        STATUS_CONTEM_OBS:     "observacao",
        STATUS_REQUER_REVISAO: "aviso",
        STATUS_NAO_CONTEM:     "critico",
    }.get(status, "critico")


def combinar_score_validacao_semantica(
    score_heur: float,
    score_bert: float,
    peso_heur: float = PESO_HEURISTICA,
    peso_bert: float = PESO_BERT,
) -> float:
    if np.isnan(score_bert):
        return float(score_heur)
    return round(peso_heur * score_heur + peso_bert * score_bert, 4)


_score_semantico_ponderado = combinar_score_validacao_semantica


def validar_secao_obrigatoria_semantica(
    texto_completo: str,
    secoes_detectadas: dict,
    label: str,
    tokenizer=None,
    model=None,
    peso_heuristica: float = PESO_HEURISTICA,
    peso_bert_param: float = PESO_BERT,
) -> dict:
    """Valida semanticamente uma secao obrigatoria apos a deteccao estrutural.

    `tem_lexico` indica se houve deteccao estrutural do heading.
    `tem_semantico` indica se o conteudo se mostrou compativel com a secao.

    Classificacao:
      tem_lexico=True  + tem_semantico=True  -> Contem secao
      tem_lexico=False + tem_semantico=True  -> Contem secao com observacao
      tem_lexico=True  + tem_semantico in {False, None} -> Requer revisao
      tem_lexico=False + tem_semantico in {False, None} -> Nao contem secao
    """
    bert_disponivel = model is not None
    tem_lexico = label in secoes_detectadas and not label.startswith("_")

    if tem_lexico:
        inicio = secoes_detectadas[label]
        proximas = sorted(
            [p for l, p in secoes_detectadas.items()
             if not l.startswith("_") and isinstance(p, int) and p > inicio]
        )
        fim = proximas[0] if proximas else len(texto_completo)
        texto_secao = texto_completo[inicio:fim].strip()
    else:
        texto_secao = texto_completo

    score_heur = _score_heuristica_secao(texto_secao, label)
    score_bert = (
        _score_bert_secao(texto_secao, label, tokenizer, model)
        if bert_disponivel else np.nan
    )
    score_sem = combinar_score_validacao_semantica(
        score_heur, score_bert, peso_heuristica, peso_bert_param
    )
    threshold = THRESHOLDS_HIBRIDO.get(label, THRESHOLD_PADRAO)

    if bert_disponivel:
        tem_semantico = bool(score_sem >= threshold)
    else:
        tem_semantico = None

    metodo_deteccao_estrutural = secoes_detectadas.get(f"_metodo_{label}") if tem_lexico else None
    metodo_validacao_semantica = []
    if bert_disponivel:
        metodo_validacao_semantica.append("bertimbau")
    elif score_heur > 0:
        metodo_validacao_semantica.append("heuristica_lexical")

    sem_positivo = (tem_semantico is True)
    sem_pendente = (tem_semantico is None)

    if tem_lexico and sem_positivo:
        status = STATUS_CONTEM
        mensagem = (
            f"Seção '{label}' detectada estruturalmente e validada semanticamente "
            f"(score={score_sem:.3f})."
        )
    elif not tem_lexico and sem_positivo:
        status = STATUS_CONTEM_OBS
        mensagem = (
            f"O conteúdo é semanticamente compatível com '{label}', "
            f"mas o título da seção não foi detectado estruturalmente "
            f"(score={score_sem:.3f}). Considere adicionar o heading para conformidade NBR 6022."
        )
    elif tem_lexico and not sem_positivo:
        status = STATUS_REQUER_REVISAO
        if sem_pendente:
            mensagem = (
                f"Heading da seção '{label}' detectado estruturalmente, "
                f"mas a validação semântica está pendente "
                f"(BERTimbau não carregado, score heurístico={score_sem:.3f})."
            )
        else:
            mensagem = (
                f"O heading sugere a seção '{label}', mas o conteúdo não apresentou "
                f"coerência semântica suficiente (score={score_sem:.3f} < {threshold}). "
                f"Verifique se a seção está desenvolvida."
            )
    else:
        status = STATUS_NAO_CONTEM
        if sem_pendente:
            mensagem = (
                f"Heading da seção '{label}' não detectado estruturalmente. "
                f"Validação semântica pendente (BERTimbau não carregado)."
            )
        else:
            mensagem = (
                f"Não foram encontradas evidências estruturais nem semânticas para '{label}' "
                f"(score={score_sem:.3f})."
            )

    metodos_compativeis = []
    if metodo_deteccao_estrutural:
        metodos_compativeis.append(metodo_deteccao_estrutural)
    metodos_compativeis.extend(metodo_validacao_semantica)

    return {
        "label": label,
        "tem_lexico": tem_lexico,
        "tem_semantico": tem_semantico,
        "status": status,
        "score_lexical": score_heur,
        "score_bert": float(score_bert) if not np.isnan(score_bert) else None,
        "score_semantico": score_sem,
        "score_final": score_sem,
        "threshold": threshold,
        "metodo_deteccao_estrutural": metodo_deteccao_estrutural,
        "metodo_validacao_semantica": metodo_validacao_semantica,
        "metodo_deteccao": metodos_compativeis,
        "mensagem": mensagem,
        "titulo_presente": tem_lexico,
        "conteudo_presente": tem_semantico,
        "nivel": _status_para_nivel_legado(status),
    }


def validar_secoes_obrigatorias_semanticas(
    texto_completo: str,
    secoes_detectadas: dict,
    tokenizer=None,
    model=None,
    peso_heuristica: float = PESO_HEURISTICA,
    peso_bert_param: float = PESO_BERT,
) -> pd.DataFrame:
    """Roda a validacao semantica para todas as secoes obrigatorias NBR 6022."""
    linhas = []
    for label in SECOES_OBRIGATORIAS_NBR:
        res = validar_secao_obrigatoria_semantica(
            texto_completo, secoes_detectadas, label,
            tokenizer, model, peso_heuristica, peso_bert_param
        )
        linhas.append(res)
    return pd.DataFrame(linhas)


avaliar_secao_hibrida = validar_secao_obrigatoria_semantica
avaliar_secoes_hibridas = validar_secoes_obrigatorias_semanticas

print("Validacao semantica das secoes obrigatorias carregada.")
print(f"  Pesos padrao -> lexical={PESO_HEURISTICA}, semantico={PESO_BERT}")

# Linhas de cabecalho editorial de revista -- nunca sao o titulo do artigo:
# datas de recebimento/aprovacao, DOI/ISSN/URLs, creditos de editor, licenca,
# banners de tipo ("Artigo", "Relato de Caso"), referencias de volume/pagina,
# e-locators (e025003), bios de autores (Doutoranda...).
_RE_LINHA_EDITORIAL = re.compile(
    r"(?i)(receb|aprovad|revisad|submet|submiss|receipt|accepted|editor|issn|"
    r"doi|http|www\.|copyright|licen[cç]a|creative commons|licensed|all the contents|"
    r"journal of|v\.\s*\d+|vol\.\s*\d+|n\.\s*\d+|p{2}?\.\s*\d+|\d{4}\s*;|\d{2}/\d{2}/\d{4}|"
    r"e\d{6,}|\d+\s*\(\d+\)\s*:|\d+\s*:\s*e?\d{4,}|"
    r"doutorand|mestrand|graduand|bolsist|"
    r"^\s*\w{1,5}\s+é\s|"
    r"^\s*artigo(\s+original)?\s*$|^\s*(relato|estudo)\s+de\s+caso\s*$)"
)


def _detectar_titulo_estrutural(texto, max_linhas=40):
    """Primeira linha 'com cara de titulo' fora do cabecalho editorial.

    Guarda de inicio de prosa: uma linha comecando com minuscula e com 4+
    palavras indica texto corrido ja quebrado em colunas -- o titulo nunca
    vem depois disso. Se nenhum candidato passar nos filtros, retorna None
    e o chamador mantem o match do regex generico (comportamento anterior).
    """
    linhas = texto.splitlines()
    idx_prosa = max_linhas
    for i, linha in enumerate(linhas[:max_linhas]):
        l = linha.strip()
        if l and l[0].islower() and len(l.split()) >= 4:
            idx_prosa = i
            break
    pos_acum = 0
    for i, linha in enumerate(linhas):
        if i >= min(max_linhas, idx_prosa):
            break
        l = linha.strip()
        pos_linha = pos_acum
        pos_acum += len(linha) + 1
        if not (6 <= len(l) <= 150):
            continue
        if not l[0].isupper():
            continue
        if len(l.split()) < 4:
            continue
        if l.endswith((".", ",", ";")):
            continue
        if ";" in l:
            continue
        if _RE_LINHA_EDITORIAL.search(l):
            continue
        return pos_linha, l
    return None, None


def detectar_secoes_estrutural_regex(texto):
    """Camada 1: regex. Retorna (posicoes, metodos, titulos_detectados)."""
    todas = {**PRE_TEXTUAIS, **TEXTUAIS, **POS_TEXTUAIS}
    out, metodos, titulos = {}, {}, {}
    for label, padrao in todas.items():
        m = re.search(padrao, texto, re.MULTILINE)
        if m:
            out[label] = m.start()
            metodos[label] = "regex"
            titulos[label] = m.group(0).strip()
    # Titulo: o padrao generico aceita qualquer linha iniciada em maiuscula
    # (pega "Recebido em: ..." etc.); a heuristica anti-editorial encontra a
    # linha real do titulo quando ela existe no texto extraido.
    pos_titulo, linha_titulo = _detectar_titulo_estrutural(texto)
    if pos_titulo is not None:
        out["titulo"] = pos_titulo
        metodos["titulo"] = "regex"
        titulos["titulo"] = linha_titulo
    return out, metodos, titulos


def detectar_secoes_estrutural_fuzzy(labels_candidatos, texto, threshold=80,
                                      pos_minima_por_label=None):
    """Camada 2: fuzzy matching para recuperar headings ausentes no regex.

    `pos_minima_por_label` (dict label -> posicao em caracteres) descarta
    linhas candidatas antes dessa posicao. Usado para impedir que headings de
    secoes do corpo casem com o banner de tipo de artigo que as revistas
    imprimem no topo da pagina (ex.: a 1a linha "Relato de Caso" casa 100%
    com a variante de metodologia "Relato de caso").
    """
    out = {}
    pos_minima_por_label = pos_minima_por_label or {}
    linhas_todas = texto.splitlines()
    pos_linha, acum = [], 0
    for l in linhas_todas:
        pos_linha.append(acum)
        acum += len(l) + 1
    # Headings nao terminam com pontuacao de frase: descarta fragmentos de
    # texto corrido (ex.: "dos dados, discussão dos resultados." casava 80
    # com a variante "Discussão dos resultados").
    linhas = [(i, l.strip()) for i, l in enumerate(linhas_todas)
              if 2 <= len(l.strip()) <= 60
              and not l.strip().endswith((".", ",", ";", ":"))]
    for label in labels_candidatos:
        pos_min = pos_minima_por_label.get(label, 0)
        variantes = CANDIDATOS_FUZZY.get(label, [label])
        best_score, best_idx, best_titulo = 0, -1, ""
        for i, linha in linhas:
            if pos_linha[i] < pos_min:
                continue
            for v in variantes:
                s = fuzz.ratio(linha.lower(), v.lower())
                if s > best_score:
                    best_score, best_idx, best_titulo = s, i, linha
        if best_score >= threshold and best_idx >= 0:
            out[label] = {
                "pos": pos_linha[best_idx],
                "titulo_detectado": best_titulo,
                "score_fuzzy": round(best_score / 100, 4),
            }
    return out


def detectar_secoes_estrutural_spacy_cabecalho(labels_faltantes, texto, nlp, threshold=0.85,
                                                posicoes_ocupadas=None):
    """Camada 3: similaridade vetorial spaCy aplicada a cabecalhos.

    threshold=0.85: calibrado empiricamente -- headings reais exatos pontuam
    ~1.0, enquanto falsos-positivos observados pontuaram 0.76-0.82 (ex.:
    "REFERÊNCIAS" vs "implicações" = 0.762, "Estruturas regulatórias" vs
    "implicações" = 0.817). Um heading real composto ("Implicações do estudo")
    pontua so 0.58, entao esta camada so e confiavel no quase-exato.

    `posicoes_ocupadas`: posicoes (em caracteres) de linhas ja reivindicadas
    por outra secao -- evita a mesma linha ser contada duas vezes (ex.: linha
    "REFERÊNCIAS" detectada como referencias E como implicacoes).
    """
    if nlp is None:
        return {}
    posicoes_ocupadas = posicoes_ocupadas or set()
    LABEL_REF = {
        "titulo": "título", "autores": "autores", "resumo": "resumo",
        "abstract": "abstract", "palavras_chave": "palavras-chave",
        "data_submissao": "data de submissão", "doi_disponib": "doi",
        "introducao": "introdução", "referencial_teorico": "referencial teórico",
        "metodologia": "metodologia", "resultados": "resultados",
        "discussao": "discussão", "implicacoes": "implicações",
        "conclusao": "conclusão", "referencias": "referências",
        "agradecimentos": "agradecimentos", "apendice": "apêndice",
        "anexo": "anexo", "glossario": "glossário",
    }
    linhas_todas = texto.splitlines()
    pos_linha, _acum = [], 0
    for l in linhas_todas:
        pos_linha.append(_acum)
        _acum += len(l) + 1
    linhas = [(i, l.strip()) for i, l in enumerate(linhas_todas)
              if 2 <= len(l.strip()) <= 60
              and pos_linha[i] not in posicoes_ocupadas
              and not l.strip().endswith((".", ",", ";", ":"))]
    out = {}
    for label in labels_faltantes:
        ref = nlp(LABEL_REF.get(label, label))
        if not ref.has_vector:
            continue
        melhor_sim, melhor_i, melhor_linha = 0.0, -1, ""
        for i, linha in linhas:
            doc = nlp(linha.lower())
            if not doc.has_vector:
                continue
            sim = ref.similarity(doc)
            if sim > melhor_sim:
                melhor_sim, melhor_i, melhor_linha = sim, i, linha
        if melhor_sim >= threshold and melhor_i >= 0:
            out[label] = {"pos": pos_linha[melhor_i], "titulo_detectado": melhor_linha}
    return out


def detectar_secoes_estrutural_spacy_conteudo(labels_faltantes, texto, nlp, threshold=0.70):
    """Camada 4: similaridade semantica spaCy sobre blocos de conteudo."""
    if nlp is None:
        return {}
    blocos_raw = re.split(r"\n{2,}", texto.strip())
    blocos, pos_acum = [], 0
    for bloco in blocos_raw:
        if len(bloco.split()) >= 30:
            blocos.append((pos_acum, bloco))
        pos_acum += len(bloco) + 2
    if not blocos:
        return {}
    vetores = []
    for pos, bloco in blocos:
        doc = nlp(bloco[:1000])
        if doc.has_vector:
            vetores.append((pos, doc.vector))
    if not vetores:
        return {}
    out = {}
    for label in labels_faltantes:
        frase_ref = CONTEUDO_REFERENCIA.get(label)
        if not frase_ref:
            continue
        ref_doc = nlp(frase_ref)
        if not ref_doc.has_vector:
            continue
        ref_vec = ref_doc.vector
        melhor_sim, melhor_pos = 0.0, -1
        for pos, vec in vetores:
            nn = np.linalg.norm(ref_vec) * np.linalg.norm(vec)
            if nn == 0:
                continue
            sim = float(np.dot(ref_vec, vec) / nn)
            if sim > melhor_sim:
                melhor_sim, melhor_pos = sim, pos
        if melhor_sim >= threshold and melhor_pos >= 0:
            out[label] = {"pos": melhor_pos}
    return out


def detectar_secoes_estrutural_bertimbau_conteudo(labels_faltantes, texto, tokenizer, model):
    """Camada 5: classificacao contextual via BERTimbau sobre blocos de conteudo.

    Fallback final, executado apos a camada spaCy de conteudo (camada 4) para
    as secoes criticas que ainda nao foram localizadas por nenhuma camada
    anterior. Reaproveita o mesmo encoder (`_chunk_encode`/`_cosseno`) e as
    mesmas frases de referencia (`CONTEUDO_REFERENCIA`) ja usados na validacao
    semantica hibrida — aqui, porem, para *detectar* a posicao de uma secao
    ausente a partir da similaridade semantica do conteudo, e nao para validar
    uma secao cujo heading ja foi encontrado.

    Usa os thresholds calibrados em THRESHOLDS_HIBRIDO/THRESHOLD_PADRAO, pois
    compara o mesmo tipo de score (cosseno de embeddings BERTimbau) usado na
    validacao semantica hibrida — diferente da escala de similaridade dos
    vetores spaCy usada na camada 4.
    """
    if model is None:
        return {}
    blocos_raw = re.split(r"\n{2,}", texto.strip())
    blocos, pos_acum = [], 0
    for bloco in blocos_raw:
        if len(bloco.split()) >= 30:
            blocos.append((pos_acum, bloco))
        pos_acum += len(bloco) + 2
    if not blocos:
        return {}

    vetores = []
    for pos, bloco in blocos:
        try:
            emb = _chunk_encode(bloco[:1000], tokenizer, model)
        except Exception:
            continue
        vetores.append((pos, emb))
    if not vetores:
        return {}

    out = {}
    for label in labels_faltantes:
        frase_ref = CONTEUDO_REFERENCIA.get(label)
        if not frase_ref:
            continue
        try:
            ref_emb = _chunk_encode(frase_ref, tokenizer, model)
        except Exception:
            continue
        threshold = THRESHOLDS_HIBRIDO.get(label, THRESHOLD_PADRAO)
        melhor_sim, melhor_pos = 0.0, -1
        for pos, emb in vetores:
            sim = _cosseno(emb, ref_emb)
            if sim > melhor_sim:
                melhor_sim, melhor_pos = sim, pos
        if melhor_sim >= threshold and melhor_pos >= 0:
            out[label] = {"pos": melhor_pos, "score_bertimbau_conteudo": round(melhor_sim, 4)}
    return out


# Labels canonicos expostos para o zero-shot classifier (camada 6).
# Cada label tem uma descricao em portugues usada como hipotese NLI --
# mais informativa que o nome curto isolado ("metodologia" sozinha e ambigua;
# "descricao dos metodos e procedimentos utilizados no estudo" nao e).
ZERO_SHOT_LABELS = {
    "introducao":         "introdução e contextualização do problema de pesquisa",
    "metodologia":        "descrição dos métodos, procedimentos, caso clínico ou experimentos realizados",
    "resultados":         "apresentação dos resultados, achados, dados e desfechos obtidos",
    "discussao":          "discussão e interpretação dos resultados em relação à literatura",
    "conclusao":          "conclusão, considerações finais e perspectivas do estudo",
    "referencias":        "lista de referências bibliográficas citadas no trabalho",
    "resumo":             "resumo ou abstract do artigo com objetivos e síntese dos resultados",
    "referencial_teorico":"revisão da literatura e embasamento teórico do estudo",
}
ZERO_SHOT_THRESHOLD = 0.50   # confianca minima para aceitar a classificacao


def detectar_secoes_estrutural_zero_shot(
    labels_faltantes: list,
    texto: str,
    zs_pipeline,
    threshold: float = ZERO_SHOT_THRESHOLD,
    tamanho_bloco: int = 400,
    sobreposicao: int = 100,
) -> dict:
    """Camada 6: zero-shot classification via NLI (mDeBERTa multilingual).

    Divide o texto em blocos sobrepostos e classifica cada bloco contra os
    labels faltantes. Para cada label, toma o bloco com maior score -- se
    esse score superar o threshold, considera a secao encontrada naquela
    posicao.

    Vantagem sobre as camadas anteriores: nao precisa de heading, palavras-
    chave ou frase de referencia -- o modelo decide pelo significado direto
    do trecho (Natural Language Inference). Util especificamente para artigos
    com estrutura nao convencional (relatos de caso, estudos de caso, artigos
    de area que usam terminologia muito diferente do vocabulario medio).

    So e chamada se USAR_ZERO_SHOT=True e o pipeline foi carregado com
    sucesso. Caso contrario, retorna {} sem quebrar o pipeline.
    """
    if zs_pipeline is None or not labels_faltantes:
        return {}

    # Monta blocos de texto sobrepostos
    palavras = texto.split()
    if not palavras:
        return {}
    blocos = []
    pos_acum = 0
    passo = max(1, tamanho_bloco - sobreposicao)
    i = 0
    while i < len(palavras):
        fim = min(i + tamanho_bloco, len(palavras))
        bloco_texto = " ".join(palavras[i:fim])
        blocos.append((pos_acum, bloco_texto))
        pos_acum += len(" ".join(palavras[i:i+passo])) + 1
        if fim == len(palavras):
            break
        i += passo

    if not blocos:
        return {}

    # Descricoes em portugues para o NLI (mais informativas que os labels curtos)
    labels_desc = [
        ZERO_SHOT_LABELS.get(l, l) for l in labels_faltantes
    ]
    label_map = dict(zip(labels_desc, labels_faltantes))

    # Score maximo por label em todos os blocos
    melhor = {l: {"score": 0.0, "pos": -1} for l in labels_faltantes}
    for pos, bloco in blocos:
        if len(bloco.split()) < 15:
            continue
        try:
            res = zs_pipeline(bloco[:1000], candidate_labels=labels_desc)
        except Exception:
            continue
        for desc, score in zip(res["labels"], res["scores"]):
            label = label_map.get(desc)
            if label and score > melhor[label]["score"]:
                melhor[label] = {"score": score, "pos": pos}

    out = {}
    for label, info in melhor.items():
        if info["score"] >= threshold and info["pos"] >= 0:
            out[label] = {
                "pos": info["pos"],
                "score_zero_shot": round(info["score"], 4),
            }
    return out


def detectar_secoes_estrutural_hibrida(texto, nlp=None, tokenizer=None, model=None, zs_pipeline=None):
    """Deteccao estrutural em 6 camadas: regex -> fuzzy -> spaCy cabecalho ->
    spaCy conteudo -> BERTimbau conteudo -> zero-shot NLI.

    As camadas 1-4 nao usam BERTimbau. A camada 5 (fallback final) usa
    BERTimbau para localizar secoes criticas que nenhuma camada anterior
    conseguiu encontrar, comparando blocos de conteudo com as frases de
    referencia de `CONTEUDO_REFERENCIA`. A validacao semantica posterior
    (independente desta deteccao) ocorre em `validar_secao_obrigatoria_semantica()`.
    """
    sec, metodos, titulos_det = detectar_secoes_estrutural_regex(texto)

    faltantes_fuzzy = [s for s in ORDEM_NBR6022
                       if s not in sec and s in CANDIDATOS_FUZZY]
    # Headings do corpo (textuais/pos-textuais) nao podem estar antes do
    # resumo ja localizado por regex -- barra o banner de tipo de artigo
    # ("Relato de Caso" etc.) impresso pela revista no topo da 1a pagina.
    # Se o resumo nao foi detectado, pos_min fica 0 e nada muda.
    _pos_resumo = sec.get("resumo")
    _labels_corpo = set(TEXTUAIS) | set(POS_TEXTUAIS)
    _pos_min_fuzzy = (
        {s: _pos_resumo for s in _labels_corpo}
        if isinstance(_pos_resumo, int) else {}
    )
    fuzzy_res = detectar_secoes_estrutural_fuzzy(
        faltantes_fuzzy, texto, threshold=80,
        pos_minima_por_label=_pos_min_fuzzy,
    )
    for label, info in fuzzy_res.items():
        sec[label] = info["pos"]
        metodos[label] = "fuzzy"
        titulos_det[label] = info["titulo_detectado"]
        sec[f"_score_fuzzy_{label}"] = info["score_fuzzy"]

    faltantes = [s for s in ORDEM_NBR6022 if s not in sec]
    if faltantes and nlp is not None:
        _ocupadas = {v for k, v in sec.items()
                     if not k.startswith("_") and isinstance(v, int)}
        spacy_res = detectar_secoes_estrutural_spacy_cabecalho(
            faltantes, texto, nlp=nlp, posicoes_ocupadas=_ocupadas)
        for label, info in spacy_res.items():
            sec[label] = info["pos"]
            metodos[label] = "spacy_cabecalho"
            titulos_det[label] = info.get("titulo_detectado", "")

    faltantes_criticos = [s for s in ORDEM_NBR6022
                          if s not in sec and SEVERIDADE_SECAO_BASE.get(s) == "critico"]
    if faltantes_criticos and nlp is not None:
        conteudo_res = detectar_secoes_estrutural_spacy_conteudo(faltantes_criticos, texto, nlp=nlp)
        for label, info in conteudo_res.items():
            sec[label] = info["pos"]
            metodos[label] = "spacy_conteudo"

    # Camada 5 (fallback final): BERTimbau sobre blocos de conteudo, apenas
    # para as secoes criticas que regex, fuzzy, spaCy cabecalho e spaCy
    # conteudo ainda nao localizaram.
    faltantes_criticos_bert = [s for s in ORDEM_NBR6022
                               if s not in sec and SEVERIDADE_SECAO_BASE.get(s) == "critico"]
    if faltantes_criticos_bert and model is not None:
        bert_res = detectar_secoes_estrutural_bertimbau_conteudo(
            faltantes_criticos_bert, texto, tokenizer=tokenizer, model=model
        )
        for label, info in bert_res.items():
            sec[label] = info["pos"]
            metodos[label] = "bertimbau_conteudo"
            sec[f"_score_bertimbau_conteudo_{label}"] = info["score_bertimbau_conteudo"]

    # Camada 6: zero-shot NLI (mDeBERTa multilingual) -- ultimo recurso para
    # secoes criticas que NENHUMA camada anterior localizou. Especialmente
    # util para artigos com estrutura nao convencional (relatos de caso,
    # terminologia de area muito especifica, ausencia de headings formais).
    # So executa se USAR_ZERO_SHOT=True e zs_pipeline foi carregado.
    faltantes_criticos_zs = [s for s in ORDEM_NBR6022
                             if s not in sec and SEVERIDADE_SECAO_BASE.get(s) == "critico"]
    if faltantes_criticos_zs and zs_pipeline is not None:
        zs_res = detectar_secoes_estrutural_zero_shot(
            faltantes_criticos_zs, texto, zs_pipeline=zs_pipeline
        )
        for label, info in zs_res.items():
            sec[label] = info["pos"]
            metodos[label] = "zero_shot_nli"
            sec[f"_score_zero_shot_{label}"] = info["score_zero_shot"]

    for label, metodo in metodos.items():
        sec[f"_metodo_{label}"] = metodo
    for label, titulo in titulos_det.items():
        sec[f"_titulo_detectado_{label}"] = titulo

    violacoes_raw = {"critico": [], "aviso": [], "observacao": []}
    for s in ORDEM_NBR6022:
        if s not in sec and s not in SECOES_OPCIONAIS_NBR:
            violacoes_raw[SEVERIDADE_SECAO_BASE.get(s, "observacao")].append(s)
    sec["_violacoes_raw"] = violacoes_raw
    return sec


def detalhar_deteccao_estrutural_secoes(secoes: dict, texto_completo: str) -> dict:
    """Converte o dict interno de secoes para uma saida estruturada da deteccao."""
    resultado = {}
    secoes_validas = {k: v for k, v in secoes.items()
                      if not k.startswith("_") and isinstance(v, int)}
    for label, inicio in sorted(secoes_validas.items(), key=lambda x: x[1]):
        proximas = sorted(
            [p for l, p in secoes_validas.items() if p > inicio]
        )
        fim = proximas[0] if proximas else len(texto_completo)
        metodo = secoes.get(f"_metodo_{label}", "desconhecido")
        titulo = secoes.get(f"_titulo_detectado_{label}", "")
        score_f = secoes.get(f"_score_fuzzy_{label}", None)
        score_bert_conteudo = secoes.get(f"_score_bertimbau_conteudo_{label}", None)
        resultado[label] = {
            "secao": label,
            "titulo_detectado": titulo,
            "metodo_deteccao": [metodo],
            "score_fuzzy": score_f,
            "score_bertimbau_conteudo": score_bert_conteudo,
            "inicio": inicio,
            "fim": fim,
            "texto_secao": texto_completo[inicio:fim].strip()[:500],
        }
    return resultado


detectar_secoes_regex = detectar_secoes_estrutural_regex
detectar_secoes_fuzzy = detectar_secoes_estrutural_fuzzy
detectar_secoes_spacy = detectar_secoes_estrutural_spacy_cabecalho
detectar_secoes_por_conteudo = detectar_secoes_estrutural_spacy_conteudo
detectar_secoes_por_conteudo_bertimbau = detectar_secoes_estrutural_bertimbau_conteudo
detectar_secoes_zero_shot = detectar_secoes_estrutural_zero_shot
detectar_secoes = detectar_secoes_estrutural_hibrida
detectar_secoes_detalhado = detalhar_deteccao_estrutural_secoes

print("Deteccao estrutural de secoes carregada: regex + fuzzy + spaCy cabecalho + spaCy conteudo + BERTimbau conteudo + zero-shot NLI.")

def _extrair_texto_secao(texto: str, secoes: dict, label: str) -> str:
    """Extrai o texto de uma secao delimitada pelas posicoes no dict de secoes."""
    inicio = secoes[label]
    proximas = sorted(
        [p for l, p in secoes.items()
         if not l.startswith("_") and isinstance(p, int) and p > inicio]
    )
    fim = proximas[0] if proximas else len(texto)
    return texto[inicio:fim].strip()


ORDEM_VALIDACAO_SEQUENCIA = [
    "introducao", "referencial_teorico", "metodologia",
    "resultados", "discussao", "implicacoes", "conclusao", "referencias",
]


def validar_ordem_secoes(secoes: dict, secoes_para_checar: list = None) -> dict:
    """Verifica se as secoes textuais detectadas aparecem no texto na ordem
    esperada pela NBR 6022 (ex.: metodologia antes de resultados, resultados
    antes de conclusao).

    Compara apenas pares de secoes onde ambas foram detectadas estruturalmente
    (presentes como posicao inteira em `secoes`). Secoes pre-textuais (titulo,
    resumo, palavras-chave etc.) tem ordem editorial mais flexivel na pratica e
    nao entram nessa checagem; o foco e o corpo do artigo, na ordem definida em
    ORDEM_VALIDACAO_SEQUENCIA.
    """
    if secoes_para_checar is None:
        secoes_para_checar = ORDEM_VALIDACAO_SEQUENCIA

    # Somente posicoes vindas de deteccao por HEADING entram na checagem de
    # ordem. As camadas de conteudo (spacy_conteudo, bertimbau_conteudo,
    # zero_shot_nli) devolvem a posicao do bloco de texto mais parecido com a
    # secao -- uma ancora aproximada, nao o inicio real da secao -- e usa-las
    # aqui gera falsos "fora de ordem" (duas secoes podem ate cair no mesmo
    # bloco). Se "_metodo_<label>" nao existir (ex.: dict de teste), assume
    # heading para manter o comportamento anterior.
    # spacy_cabecalho fica de fora: mesmo com threshold alto, e a camada com
    # mais falsos-positivos de posicao -- e um unico heading errado gera
    # varios pares "fora de ordem" espurios.
    _METODOS_HEADING = {"regex", "fuzzy"}
    detectadas = [
        (s, secoes[s]) for s in secoes_para_checar
        if s in secoes and isinstance(secoes[s], int)
        and secoes.get(f"_metodo_{s}", "regex") in _METODOS_HEADING
    ]

    pares_fora_de_ordem = []
    for i in range(len(detectadas)):
        for j in range(i + 1, len(detectadas)):
            label_i, pos_i = detectadas[i]
            label_j, pos_j = detectadas[j]
            # label_i deveria vir antes de label_j na ordem canonica;
            # se a posicao de label_i for maior, a secao esta fora de ordem.
            if pos_i > pos_j:
                pares_fora_de_ordem.append((label_i, label_j))

    return {
        "ordem_correta": len(pares_fora_de_ordem) == 0,
        "pares_fora_de_ordem": pares_fora_de_ordem,
        "secoes_consideradas": [s for s, _ in detectadas],
    }


def validar_estrutura_abnt(
    secoes: dict,
    texto_completo: str = "",
    tokenizer=None,
    model=None,
    peso_heuristica: float = PESO_HEURISTICA,
    peso_bert_param: float = PESO_BERT,
) -> dict:
    """Validacao estrutural em dois passos.

    Passo 1 - secoes nao obrigatorias: usa mapa fixo SEVERIDADE_SECAO_BASE.
    Passo 2 - secoes obrigatorias NBR: usa validacao semantica apos a
              deteccao estrutural do heading.

    Score final: 100 - (criticos*15) - (avisos*5) - (observacoes*2).
    Conforme: score >= 70 e sem criticos.
    """
    criticos, avisos, observacoes = [], [], []
    detalhes_hibrido = {}
    secoes_resultado = {}

    secoes_fixas = [s for s in ORDEM_NBR6022 if s not in SECOES_OBRIGATORIAS_NBR]
    violacoes_raw = secoes.get("_violacoes_raw", {})
    for nivel, lista in violacoes_raw.items():
        for s in lista:
            if s in secoes_fixas:
                if nivel == "critico":
                    criticos.append(s)
                elif nivel == "aviso":
                    avisos.append(s)
                else:
                    observacoes.append(s)

    if texto_completo:
        for label in SECOES_OBRIGATORIAS_NBR:
            res = validar_secao_obrigatoria_semantica(
                texto_completo, secoes, label,
                tokenizer, model, peso_heuristica, peso_bert_param
            )
            detalhes_hibrido[label] = res

            secao_saida = {
                "presente": res["status"] in (STATUS_CONTEM, STATUS_CONTEM_OBS),
                "status": res["status"],
                "tem_lexico": res["tem_lexico"],
                "tem_semantico": res["tem_semantico"],
                "score_lexical": res["score_lexical"],
                "score_semantico": res["score_semantico"],
                "score_final": res["score_final"],
                "metodo_deteccao_estrutural": res["metodo_deteccao_estrutural"],
                "metodo_validacao_semantica": res["metodo_validacao_semantica"],
                "metodo_deteccao": res["metodo_deteccao"],
                "mensagem": res["mensagem"],
            }
            if res["status"] == STATUS_CONTEM_OBS:
                secao_saida["observacao"] = res["mensagem"]
            secoes_resultado[label] = secao_saida

            penalidade = PENALIDADE_STATUS.get(res["status"], 0)
            if penalidade >= 15:
                criticos.append(label)
            elif penalidade >= 5:
                avisos.append(label)
            elif penalidade >= 2:
                observacoes.append(label)
    else:
        for nivel, lista in violacoes_raw.items():
            for s in lista:
                if s in SECOES_OBRIGATORIAS_NBR:
                    if nivel == "critico":
                        criticos.append(s)
                    elif nivel == "aviso":
                        avisos.append(s)
                    else:
                        observacoes.append(s)

    # Verificacao de ordem das secoes textuais (NBR 6022): alem da presenca,
    # confere se as secoes do corpo do artigo aparecem na sequencia esperada.
    ordem_info = validar_ordem_secoes(secoes)
    if not ordem_info["ordem_correta"]:
        avisos.append("ordem_secoes")

    score = max(0, 100 - len(criticos) * 15 - len(avisos) * 5 - len(observacoes) * 2)
    conforme = (score >= 70 and len(criticos) == 0)

    return {
        "secoes": secoes_resultado,
        "_resumo": {
            "score_conformidade": score,
            "conforme_nbr6022": conforme,
            "criticos": criticos,
            "avisos": avisos,
            "observacoes": observacoes,
            "detalhes_hibrido": detalhes_hibrido,
            "ordem_secoes": ordem_info,
        },
    }


print("validar_estrutura_abnt carregada com separacao entre deteccao estrutural e validacao semantica.")

NGRAM_RANGE_PADRAO = (1, 2)   # testar (2,2) para bigramas puros

# Termos esperados por secao para Jaccard + indicador lexical (Bloco 2)
TERMOS_ESPERADOS_JACCARD = {
    "introducao":    {"contexto", "problema", "objetivo", "justificativa",
                      "pesquisa", "estudo", "tema", "motivacao", "literatura",                      "objetivos"},
    "metodologia":   {"metodo", "metodologia", "amostra", "dados",
                      "procedimento", "analise", "coleta", "instrumento",
                      "questionario", "criterio"},
    "resultados":    {"resultado", "obtido", "dados", "analise",
                      "tabela", "figura", "observou", "percentual",
                      "media", "frequencia"},
    "conclusao":     {"conclusao", "consideracoes", "final", "contribuicao",
                      "limitacao", "futuro", "sintese", "alcancado",
                      "objetivo", "recomenda"},
    "referencias":   {"autor", "ano", "revista", "editora", "doi",
                      "disponivel", "acesso", "journal", "volume",
                      "publicacao"},
}


def _indicador_lexical(jaccard: float,
                        limiar_adequado: float = LIMIAR_LEXICO_ADEQUADO,
                        limiar_parcial: float  = LIMIAR_LEXICO_PARCIAL) -> str:
    """Classifica o score Jaccard em adequado / parcial / baixo."""
    if jaccard >= limiar_adequado:
        return "adequado"
    if jaccard >= limiar_parcial:
        return "parcial"
    return "baixo"


def calcular_jaccard_secoes(secoes_por_secao: dict) -> dict:
    """Jaccard Similarity + indicador lexical por secao.

    Jaccard = |interseccao| / |uniao|

    Retorna dict label -> {jaccard, indicador_lexical, intersecao,
                          n_esperados, n_encontrados}.
    """
    resultado = {}
    for label, dados in secoes_por_secao.items():
        esperados = TERMOS_ESPERADOS_JACCARD.get(label)
        if not esperados:
            continue
        bow = dados.get("bow", {})
        encontrados = set(bow.keys())
        intersecao  = esperados & encontrados
        uniao       = esperados | encontrados
        jaccard     = round(len(intersecao) / len(uniao), 4) if uniao else 0.0
        resultado[label] = {
            "jaccard":          jaccard,
            "indicador_lexical": _indicador_lexical(jaccard),
            "intersecao":       sorted(intersecao),
            "n_esperados":      len(esperados),
            "n_encontrados":    len(encontrados),
        }
    return resultado


def _melhor_similaridade_fasttext(termo: str, termos_candidatos, ft_model) -> float:
    """Maior similaridade de cosseno entre `termo` e qualquer termo em
    `termos_candidatos`, usando vetores FastText (com fallback de subpalavra
    para termos fora do vocabulario)."""
    try:
        v1 = ft_model.get_vector(termo)
    except KeyError:
        return 0.0
    melhor = 0.0
    for cand in termos_candidatos:
        try:
            v2 = ft_model.get_vector(cand)
        except KeyError:
            continue
        sim = _cosseno(v1, v2)
        if sim > melhor:
            melhor = sim
    return melhor


def calcular_jaccard_semantico_secoes(secoes_por_secao: dict, ft_model, threshold_sim: float = 0.65) -> dict:
    """Jaccard semantico: complementa calcular_jaccard_secoes aceitando
    sinonimos via similaridade de vetores FastText, nao so match exato de
    string.

    O Jaccard lexico original nao reconhece "dataset" e "conjunto de dados"
    como equivalentes, mesmo sendo sinonimos -- isso penaliza artigos que
    usam vocabulario diferente do esperado, mas semanticamente adequado.
    Aqui, um termo esperado conta como encontrado se ele proprio estiver no
    texto OU se houver um termo do texto com similaridade >= threshold_sim.
    """
    if ft_model is None:
        return {}
    resultado = {}
    for label, dados in secoes_por_secao.items():
        esperados = TERMOS_ESPERADOS_JACCARD.get(label)
        if not esperados:
            continue
        bow = dados.get("bow", {})
        encontrados = set(bow.keys())

        matches_semanticos = set()
        for esperado in esperados:
            if esperado in encontrados:
                matches_semanticos.add(esperado)
                continue
            sim = _melhor_similaridade_fasttext(esperado, encontrados, ft_model)
            if sim >= threshold_sim:
                matches_semanticos.add(esperado)

        uniao = esperados | encontrados
        jaccard_sem = round(len(matches_semanticos) / len(uniao), 4) if uniao else 0.0
        resultado[label] = {
            "jaccard_semantico":           jaccard_sem,
            "matches_semanticos":          sorted(matches_semanticos),
            "indicador_lexical_semantico": _indicador_lexical(jaccard_sem),
        }
    return resultado


def extrair_termos_chave_keybert(texto: str, kw_model, top_n: int = 10, ngram_range: tuple = (1, 2)) -> list:
    """Extrai termos-chave via KeyBERT (similaridade de embeddings de
    sentenca entre candidatos n-grama e o texto completo).

    Diferente do top-N por score TF-IDF (estatistico, baseado em raridade
    no corpus), o KeyBERT escolhe termos por proximidade semantica ao
    significado geral do texto -- captura sinonimos e relacoes conceituais
    que o TF-IDF nao ve. Os dois sao mantidos lado a lado (nao um substitui
    o outro) porque servem propositos diferentes: TF-IDF tambem alimenta a
    comparacao de similaridade lexical entre secoes do mesmo artigo.
    """
    if kw_model is None:
        return []
    texto_limpo = str(texto)[:3000]
    if len(texto_limpo.split()) < 5:
        return []
    try:
        pares = kw_model.extract_keywords(
            texto_limpo, keyphrase_ngram_range=ngram_range,
            stop_words=None, top_n=top_n,
        )
        return [termo for termo, _score in pares]
    except Exception:
        return []


def analisar_lexico(
    texto_completo: str,
    secoes: dict,
    stopwords_pt: set | None = None,
    top_n: int = 20,
    ngram_range: tuple = NGRAM_RANGE_PADRAO,
    ft_model=None,
    kw_model=None,
) -> dict:
    """Analise lexical por secao com BoW, TF-IDF, Jaccard, e (opcional)
    Jaccard semantico (FastText) e termos-chave via KeyBERT.

    Cada secao retorna:
      bow_top_terms     : list[str]  - top N termos mais frequentes (BoW)
      tfidf_top_terms    : list[str]  - top N termos mais relevantes (TF-IDF)
      keybert_top_terms  : list[str]  - top N termos por similaridade semantica (se kw_model fornecido)
      bow                : dict       - contagens brutas (para Jaccard interno)
      tfidf              : list[tuple]- (termo, score) completo
      jaccard_score      : float      - sobreposicao lexical (match exato) com termos esperados
      jaccard_semantico   : float      - sobreposicao semantica via FastText (se ft_model fornecido)
      indicador_lexical   : str        - adequado / parcial / baixo

    ngram_range padrao (1,2). Testar (2,2) para bigramas puros:
      'revisao bibliografica', 'analise dos dados', 'materiais e metodos'.
    """
    secoes_validas = {k: v for k, v in secoes.items()
                      if not k.startswith("_") and isinstance(v, int)}
    textos_secoes = {}
    for label in sorted(secoes_validas, key=lambda x: secoes_validas[x]):
        txt = _extrair_texto_secao(texto_completo, secoes_validas, label)
        txt_limpo = preprocessar_texto_lexico(txt, stopwords_lexico=stopwords_pt, manter_nao=True)
        if len(txt_limpo.split()) >= 5:
            textos_secoes[label] = txt_limpo

    if not textos_secoes:
        return {"artigo_completo": {}, "por_secao": {},
                "similaridade_lexical_media": np.nan, "jaccard_por_secao": {},
                "jaccard_semantico_por_secao": {}}

    labels  = list(textos_secoes.keys())
    corpus  = [textos_secoes[l] for l in labels]
    n_docs  = len(corpus)
    min_df  = 1 if n_docs <= 3 else 2
    max_df_tfidf = 1.0 if n_docs <= 3 else 0.85
    max_df_bow   = 1.0 if n_docs <= 3 else 0.95

    tfidf_vec = TfidfVectorizer(
        max_features=2000, ngram_range=ngram_range,
        sublinear_tf=True, min_df=min_df, max_df=max_df_tfidf,
        token_pattern=r"(?u)\b\w+\b"
    )
    bow_vec = CountVectorizer(
        max_features=2000, ngram_range=ngram_range,
        min_df=min_df, max_df=max_df_bow,
        token_pattern=r"(?u)\b\w+\b"
    )
    try:
        X_tfidf = tfidf_vec.fit_transform(corpus)
        X_bow   = bow_vec.fit_transform(corpus)
    except ValueError as e:
        por_secao_vazio = {
            label: {
                "bow_top_terms": [],
                "tfidf_top_terms": [],
                "keybert_top_terms": [],
                "tfidf": [],
                "bow": {},
                "jaccard_score": None,
                "jaccard_semantico": None,
                "indicador_lexical": None,
            }
            for label in labels
        }
        return {
            "artigo_completo": {},
            "por_secao": por_secao_vazio,
            "similaridade_lexical_media": np.nan,
            "ngram_range_usado": str(ngram_range),
            "jaccard_por_secao": {},
            "jaccard_semantico_por_secao": {},
            "_erro": f"analisar_lexico: {e}"
        }

    tfidf_features = tfidf_vec.get_feature_names_out()
    bow_features   = bow_vec.get_feature_names_out()

    por_secao = {}
    for i, label in enumerate(labels):
        tfidf_scores = X_tfidf[i].toarray()[0]
        bow_counts   = X_bow[i].toarray()[0]
        top_t = tfidf_scores.argsort()[::-1][:top_n]
        top_b = bow_counts.argsort()[::-1][:top_n]

        tfidf_full  = [(tfidf_features[j], float(tfidf_scores[j]))
                       for j in top_t if tfidf_scores[j] > 0]
        bow_dict    = {bow_features[j]: int(bow_counts[j])
                       for j in top_b if bow_counts[j] > 0}

        por_secao[label] = {
            "bow_top_terms":   [t for t in bow_dict.keys()],
            "tfidf_top_terms": [t for t, _ in tfidf_full],
            "tfidf":           tfidf_full,
            "bow":             bow_dict,
        }

    sim_lex = cosine_similarity(X_tfidf)
    sims = [sim_lex[i, j] for i in range(len(labels)) for j in range(i + 1, len(labels))]
    sim_media = float(np.mean(sims)) if sims else np.nan

    # Jaccard + indicador lexical por secao (match exato de string)
    jaccard_por_secao = calcular_jaccard_secoes(por_secao)
    for label, jdata in jaccard_por_secao.items():
        por_secao[label]["jaccard_score"] = jdata["jaccard"]
        por_secao[label]["indicador_lexical"] = jdata["indicador_lexical"]

    # Jaccard semantico (FastText) -- opcional, complementa o lexico acima
    jaccard_semantico_por_secao = calcular_jaccard_semantico_secoes(por_secao, ft_model)
    for label, jdata in jaccard_semantico_por_secao.items():
        por_secao[label]["jaccard_semantico"] = jdata["jaccard_semantico"]
        por_secao[label]["matches_semanticos"] = jdata["matches_semanticos"]

    # Termos-chave via KeyBERT -- opcional, ao lado do tfidf_top_terms (nao o substitui)
    if kw_model is not None:
        for label in labels:
            por_secao[label]["keybert_top_terms"] = extrair_termos_chave_keybert(
                textos_secoes[label], kw_model, top_n=top_n, ngram_range=ngram_range
            )

    return {
        "artigo_completo":            {},
        "por_secao":                  por_secao,
        "similaridade_lexical_media": sim_media,
        "ngram_range_usado":          str(ngram_range),
        "jaccard_por_secao":          jaccard_por_secao,
        "jaccard_semantico_por_secao": jaccard_semantico_por_secao,
    }


print("analisar_lexico (bow_top_terms, tfidf_top_terms, indicador_lexical) carregada.")

def analisar_coerencia_semantica(texto, secoes, tokenizer, model, threshold=0.50):
    """Similaridade coseno entre pares de seções (BERTimbau). Inalterado."""
    secoes_validas = {k: v for k, v in secoes.items()
                      if not k.startswith('_') and isinstance(v, int)}
    if len(secoes_validas) < 2:
        return {"erro": "Menos de 2 seções válidas", "labels": [], "matriz": [],
                "media_por_secao": {}, "secoes_problemas": []}
    embeddings = {}
    for label in sorted(secoes_validas, key=lambda l: secoes_validas[l]):
        txt = _extrair_texto_secao(texto, secoes_validas, label)
        if len(txt.split()) < 10:
            continue
        embeddings[label] = _chunk_encode(txt, tokenizer, model)
    if len(embeddings) < 2:
        return {"erro": "Seções úteis insuficientes", "labels": list(embeddings.keys()),
                "matriz": [], "media_por_secao": {}, "secoes_problemas": []}
    labels = list(embeddings.keys())
    matriz = cosine_similarity(np.vstack([embeddings[l] for l in labels]))
    media = {lb: float(np.mean([matriz[i, j] for j in range(len(labels)) if j != i]))
             for i, lb in enumerate(labels)}
    probs = [l for l, m in media.items() if m < threshold]
    return {"labels": labels, "matriz": matriz.tolist(),
            "media_por_secao": media, "secoes_problemas": probs, "threshold": threshold}

print("analisar_coerencia_semantica carregado.")

def calcular_bertscore_secoes(
    texto_completo: str,
    secoes: dict,
    lang: str = "pt",
) -> dict:
    """BERTScore (Precision, Recall, F1) para cada seção obrigatória da NBR 6022.

    Compara o texto extraído de cada seção com a frase-âncora de referência
    (CONTEUDO_REFERENCIA). O F1 é a métrica principal de similaridade semântica.

    Requer: pip install bert-score
    """
    if not BERTSCORE_DISPONIVEL:
        return {"_erro": "bert-score não instalado. Execute: pip install bert-score"}

    secoes_validas = {k: v for k, v in secoes.items()
                      if not k.startswith("_") and isinstance(v, int)}
    resultado = {}

    for label in SECOES_OBRIGATORIAS_NBR:
        frase_ref = CONTEUDO_REFERENCIA.get(label)
        if not frase_ref or label not in secoes_validas:
            resultado[label] = {
                "precision": None, "recall": None, "f1": None,
                "observacao": "seção não detectada ou sem referência",
            }
            continue

        texto_secao = _extrair_texto_secao(texto_completo, secoes_validas, label)
        if len(texto_secao.split()) < 10:
            resultado[label] = {
                "precision": None, "recall": None, "f1": None,
                "observacao": "texto da seção muito curto (< 10 tokens)",
            }
            continue

        try:
            P, R, F = bert_score_fn(
                [texto_secao[:1000]],
                [frase_ref],
                lang=lang,
                verbose=False,
            )
            resultado[label] = {
                "precision": round(float(P[0]), 4),
                "recall":    round(float(R[0]), 4),
                "f1":        round(float(F[0]), 4),
            }
        except Exception as e:
            resultado[label] = {
                "precision": None, "recall": None, "f1": None,
                "erro": str(e),
            }

    return resultado


print("calcular_bertscore_secoes carregada.")
print(f"  BERTScore disponível: {BERTSCORE_DISPONIVEL}")

_MESES_PT = (
    "janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|"
    "setembro|outubro|novembro|dezembro"
)
_RE_DATA_EXTENSO = re.compile(
    rf"\b\d{{1,2}}\s+de\s+(?:{_MESES_PT})\s+de\s+\d{{4}}\b", re.IGNORECASE
)
_RE_DATA_MES_ANO = re.compile(
    rf"\b(?:{_MESES_PT})\s+de\s+\d{{4}}\b", re.IGNORECASE
)
_RE_DATA_NUMERICA = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")


def extrair_datas_regex(texto: str) -> list:
    """Extrai datas do texto via regex.

    O modelo spaCy pt_core_news_lg usa o esquema de rotulos PER/LOC/ORG/MISC
    e nao possui um rotulo DATE dedicado, entao datas nao aparecem como
    categoria propria no NER nativo do spaCy. Esta funcao complementa
    extrair_entidades() com uma categoria "DATA" real, cobrindo os formatos
    mais comuns em artigos cientificos (extenso, mes/ano e numerico).

    Datas mais especificas (ex.: "12 de março de 2023") tem prioridade sobre
    trechos mais curtos que estejam contidos nelas (ex.: "março de 2023"), que
    sao descartados para evitar duplicidade.
    """
    spans = []
    for padrao in (_RE_DATA_EXTENSO, _RE_DATA_MES_ANO, _RE_DATA_NUMERICA):
        for m in padrao.finditer(texto):
            spans.append((m.start(), m.end(), m.group(0).strip()))

    # Remove spans inteiramente contidos em outro span mais longo
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    selecionados = []
    for inicio, fim, valor in spans:
        if any(i0 <= inicio and fim <= f0 and (i0, f0) != (inicio, fim)
               for i0, f0, _ in selecionados):
            continue
        selecionados.append((inicio, fim, valor))

    encontradas = []
    vistos = set()
    for _, _, valor in sorted(selecionados, key=lambda s: s[0]):
        chave = valor.lower()
        if chave not in vistos:
            vistos.add(chave)
            encontradas.append(valor)
    return encontradas


_LENERBR_LABEL_MAP = {
    "PESSOA": "PER",
    "ORGANIZACAO": "ORG",
    "LOCAL": "LOC",
    "TEMPO": "MISC",
    "LEGISLACAO": "MISC",
    "JURISPRUDENCIA": "MISC",
}


def extrair_entidades_lenerbr(texto: str, ner_pipeline) -> dict:
    """Extrai PER/ORG/LOC via BERTimbau fine-tuned no LeNER-BR.

    Mais preciso que o NER generico do spaCy para nomes de pessoas e
    instituicoes em texto formal/academico em portugues -- especialmente
    para afiliacoes com sigla, onde o spaCy generico erra mais.
    """
    if ner_pipeline is None:
        return {}
    try:
        resultados = ner_pipeline(texto[:5000])
    except Exception:
        return {}
    entidades = {}
    for ent in resultados:
        label_raw = ent.get("entity_group", ent.get("entity", ""))
        label = _LENERBR_LABEL_MAP.get(label_raw, label_raw)
        palavra = str(ent.get("word", "")).strip()
        if palavra:
            entidades.setdefault(label, []).append(palavra)
    return entidades


def extrair_entidades(texto: str, nlp, ner_pipeline_lenerbr=None) -> dict:
    """Extrai entidades nomeadas, complementado por extracao de datas via
    regex (ver extrair_datas_regex).

    PER/ORG/LOC: preferencialmente via LeNER-BR (`ner_pipeline_lenerbr`), se
    fornecido -- mais preciso que o spaCy generico para esse contexto. Sem
    LeNER-BR, cai para o NER nativo do spaCy (PER/LOC/ORG/MISC).
    """
    texto_ner = texto[:5000]
    entidades = {}

    if ner_pipeline_lenerbr is not None:
        entidades = extrair_entidades_lenerbr(texto_ner, ner_pipeline_lenerbr)

    if not entidades:
        if nlp is None:
            if ner_pipeline_lenerbr is None:
                return {"erro": "Nenhum modelo de NER disponível (spaCy e LeNER-BR ausentes)"}
        else:
            try:
                doc = nlp(texto_ner)
                for ent in doc.ents:
                    entidades.setdefault(ent.label_, []).append(ent.text)
            except Exception as e:
                return {"erro": str(e)}

    datas = extrair_datas_regex(texto_ner)
    if datas:
        entidades["DATA"] = datas
    return entidades


def extrair_entidades_cientificas_abstract(texto_completo: str, secoes: dict, ner_pipeline_scierc) -> dict:
    """Extrai termos cientificos (Task/Method/Metric/Material/
    OtherScientificTerm) da secao "abstract", via NER fine-tuned no SciERC.

    Escopo deliberadamente limitado ao abstract: o SciERC e treinado em
    ingles, e o abstract e a unica secao do artigo genuinamente em ingles
    (exigida pela NBR 6022 mesmo em artigos em portugues) -- aplicar o
    modelo no corpo do artigo (em portugues) nao funcionaria sem tradução.
    Se a secao abstract nao foi detectada, ou o pipeline nao esta
    disponivel, retorna {} (degradacao graciosa, sem quebrar o pipeline).
    """
    if ner_pipeline_scierc is None:
        return {}
    secoes_validas = {k: v for k, v in secoes.items()
                      if not k.startswith("_") and isinstance(v, int)}
    if "abstract" not in secoes_validas:
        return {}
    texto_abstract = _extrair_texto_secao(texto_completo, secoes_validas, "abstract")
    if len(texto_abstract.split()) < 5:
        return {}
    try:
        resultados = ner_pipeline_scierc(texto_abstract[:2000])
    except Exception:
        return {}
    entidades = {}
    for ent in resultados:
        label = ent.get("entity_group", ent.get("entity", "OUTRO"))
        palavra = str(ent.get("word", "")).strip()
        if palavra:
            entidades.setdefault(label, []).append(palavra)
    return entidades


def validar_ner_com_metadados(
    entidades: dict,
    metadados: Optional[dict] = None,
) -> dict:
    """Valida entidades identificadas pelo NER contra metadados fornecidos.

    metadados pode conter:
        "autores"      : list[str] — autores esperados
        "instituicoes" : list[str] — instituições esperadas

    Usa fuzzy matching (token_sort_ratio) para tolerar variações de escrita.

    Retorna:
        autores_detectados      : entidades PER identificadas no texto
        organizacoes_detectadas : entidades ORG/LOC identificadas
        validacao_com_metadados : cruzamento com metadados (se fornecidos)
    """
    autores_ner = entidades.get("PER", []) if isinstance(entidades, dict) else []
    orgs_ner    = (entidades.get("ORG", []) + entidades.get("LOC", [])
                   if isinstance(entidades, dict) else [])

    resultado = {
        "entidades_identificadas": entidades,
        "autores_detectados":      list(set(autores_ner)),
        "organizacoes_detectadas": list(set(orgs_ner)),
        "validacao_com_metadados": {},
    }

    if not metadados or not isinstance(metadados, dict):
        return resultado

    # ── Cruza autores esperados x detectados ──────────────────────────────
    autores_esperados = metadados.get("autores", [])
    validacao_autores = {}
    for autor_esp in autores_esperados:
        melhor_match, melhor_score = None, 0
        for autor_ner in autores_ner:
            s = fuzz.token_sort_ratio(autor_esp.lower(), autor_ner.lower())
            if s > melhor_score:
                melhor_score, melhor_match = s, autor_ner
        validacao_autores[autor_esp] = {
            "encontrado_no_texto":    melhor_score >= 75,
            "melhor_correspondencia": melhor_match,
            "score_fuzzy":            melhor_score,
        }

    # ── Cruza instituições esperadas x detectadas ─────────────────────────
    inst_esperadas = metadados.get("instituicoes", [])
    validacao_inst = {}
    for inst_esp in inst_esperadas:
        melhor_match, melhor_score = None, 0
        for org_ner in orgs_ner:
            s = fuzz.token_sort_ratio(inst_esp.lower(), org_ner.lower())
            if s > melhor_score:
                melhor_score, melhor_match = s, org_ner
        validacao_inst[inst_esp] = {
            "encontrada_no_texto":    melhor_score >= 70,
            "melhor_correspondencia": melhor_match,
            "score_fuzzy":            melhor_score,
        }

    resultado["validacao_com_metadados"] = {
        "autores":      validacao_autores,
        "instituicoes": validacao_inst,
    }
    return resultado


print("NER + validação cruzada com metadados carregado.")

def agregar_indicadores(
    validacao: dict,
    coerencia: dict,
    lexico: dict,
    ner_resultado: dict,
    bertscore: Optional[dict] = None,
    entidades_cientificas_abstract: Optional[dict] = None,
) -> dict:
    """Agrega métricas de todas as etapas do pipeline num único dict de indicadores."""
    resumo   = validacao.get("_resumo", {})
    det_hibr = resumo.get("detalhes_hibrido", {})

    # Similaridade semântica média entre seções (BERTimbau pairwise)
    media_sem = np.nan
    if isinstance(coerencia, dict) and coerencia.get("media_por_secao"):
        media_sem = float(np.mean(list(coerencia["media_por_secao"].values())))

    # Score semântico médio das seções obrigatórias (validação híbrida)
    scores_hib = [v["score_semantico"] for v in det_hibr.values()
                  if isinstance(v, dict) and "score_semantico" in v]
    score_sem_hibrido = round(float(np.mean(scores_hib)), 4) if scores_hib else np.nan

    # BERTScore F1 médio
    bertscore_f1_medio = np.nan
    if bertscore and isinstance(bertscore, dict):
        f1s = [v["f1"] for v in bertscore.values()
               if isinstance(v, dict) and v.get("f1") is not None]
        if f1s:
            bertscore_f1_medio = round(float(np.mean(f1s)), 4)

    # Jaccard médio (lexico, match exato)
    jaccard_medio = np.nan
    jaccard_por_secao = lexico.get("jaccard_por_secao", {}) if isinstance(lexico, dict) else {}
    if jaccard_por_secao:
        jacs = [v["jaccard"] for v in jaccard_por_secao.values()
                if isinstance(v, dict) and "jaccard" in v]
        if jacs:
            jaccard_medio = round(float(np.mean(jacs)), 4)

    # Jaccard semântico médio (FastText, aceita sinônimos)
    jaccard_semantico_medio = np.nan
    jaccard_sem_por_secao = lexico.get("jaccard_semantico_por_secao", {}) if isinstance(lexico, dict) else {}
    if jaccard_sem_por_secao:
        jacs_sem = [v["jaccard_semantico"] for v in jaccard_sem_por_secao.values()
                    if isinstance(v, dict) and "jaccard_semantico" in v]
        if jacs_sem:
            jaccard_semantico_medio = round(float(np.mean(jacs_sem)), 4)

    # Termos científicos extraídos do abstract via SciERC (Task/Method/Metric/Material)
    n_termos_cientificos_abstract = (
        sum(len(v) for v in entidades_cientificas_abstract.values())
        if isinstance(entidades_cientificas_abstract, dict) else 0
    )

    return {
        "score_abnt_heuristico":          resumo.get("score_conformidade", np.nan),
        "conforme_nbr6022":               resumo.get("conforme_nbr6022", False),
        "n_criticos":                     len(resumo.get("criticos", [])),
        "n_avisos":                       len(resumo.get("avisos", [])),
        "n_observacoes":                  len(resumo.get("observacoes", [])),
        "score_semantico_hibrido_medio":  score_sem_hibrido,
        "n_secoes_problemas_semantica":   len(coerencia.get("secoes_problemas", []))
                                          if isinstance(coerencia, dict) else np.nan,
        "media_similaridade_semantica":   media_sem,
        "similaridade_lexical_media":     lexico.get("similaridade_lexical_media", np.nan)
                                          if isinstance(lexico, dict) else np.nan,
        "jaccard_medio":                  jaccard_medio,
        "bertscore_f1_medio":             bertscore_f1_medio,
        "n_entidades_per":                len(ner_resultado.get("autores_detectados", []))
                                          if isinstance(ner_resultado, dict) else np.nan,
        "n_entidades_org":                len(ner_resultado.get("organizacoes_detectadas", []))
                                          if isinstance(ner_resultado, dict) else np.nan,
        "n_entidades_data":               len(ner_resultado.get("entidades_identificadas", {}).get("DATA", []))
                                          if isinstance(ner_resultado, dict) else np.nan,
        "ordem_secoes_correta":           resumo.get("ordem_secoes", {}).get("ordem_correta", None),
        "ordem_secoes_pares_fora_de_ordem": resumo.get("ordem_secoes", {}).get("pares_fora_de_ordem", []),
        "jaccard_semantico_medio":         jaccard_semantico_medio,
        "n_termos_cientificos_abstract":   n_termos_cientificos_abstract,
        "detalhes_hibrido":               det_hibr,
    }


print("agregar_indicadores (BERTScore + Jaccard + NER unificado) carregado.")


# ═══════════════════════════════════════════════════════════════════════════════
# Avaliacao de Conformidade de Citacoes  ·  NBR 10520 (fase inicial)
#
# Escopo desta fase:
#  ✅ Citacoes diretas (com aspas + chamada autor-data)
#  ✅ Citacoes indiretas (parafrase com chamada autor-data)
#
# Fora do escopo nesta fase:
#  ✗ Apud  ✗ Validacao citacao-referencia  ✗ NER nas citacoes
#  ✗ Analise semantica  ✗ Verificacao da secao de referencias
# ═══════════════════════════════════════════════════════════════════════════════

_RE_PAG_SIMPLES = re.compile(
    r"(?:pp?\.\s*\d+(?:\s*[-–]\s*\d+)?|pág(?:ina)?s?\.\s*\d+(?:\s*[-–]\s*\d+)?)",
    re.IGNORECASE,
)

_S_AUTOR  = r"[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÇÜ][A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÇÜa-záéíóúâêîôûãõàçü\-]+"
_S_ANO    = r"(?:19|20)\d{2}"
_S_PAG    = r"(?:pp?\.\s*\d+(?:\s*[-–]\s*\d+)?|pág(?:ina)?s?\.\s*\d+(?:\s*[-–]\s*\d+)?)"
_S_AUTORS = (
    rf"{_S_AUTOR}"
    rf"(?:\s*[;,]\s*{_S_AUTOR})*"
    rf"(?:\s+et\s+al\.?)?"
)
_S_CHAVE  = rf"{_S_AUTORS}\s*,\s*{_S_ANO}(?:,\s*{_S_PAG})?"

_QO = r'["\u201c\u00ab]'
_QC = r'["\u201d\u00bb]'

_RE_DIR_PAREN = re.compile(
    rf"{_QO}.{{5,600}}?{_QC}\s*\({_S_CHAVE}\)",
    re.DOTALL | re.IGNORECASE,
)

_RE_DIR_INTRO = re.compile(
    rf"(?:Segundo|De\s+acordo\s+com|Conforme|Para|Nas\s+palavras\s+de)\s+"
    rf"{_S_AUTORS}\s*\(\s*{_S_ANO}(?:\s*,\s*{_S_PAG})?\)\s*,?\s*"
    rf"{_QO}.{{5,600}}?{_QC}",
    re.DOTALL | re.IGNORECASE,
)

_RE_IND_PAREN = re.compile(
    rf"\({_S_CHAVE}\)",
    re.IGNORECASE,
)

_RE_IND_TEXTUAL = re.compile(
    rf"{_S_AUTORS}\s*\(\s*{_S_ANO}(?:\s*,\s*{_S_PAG})?\)",
    re.IGNORECASE,
)

_RE_IND_INTRO = re.compile(
    rf"(?:Segundo|De\s+acordo\s+com|Conforme|Para|Na\s+visão\s+de|"
    rf"Nas\s+palavras\s+de|Corroborando)\s+"
    rf"{_S_AUTORS}(?:\s*\(\s*{_S_ANO}\s*\))?",
    re.IGNORECASE,
)


def _tem_pagina_simples(trecho: str) -> bool:
    """Verifica se o trecho contem indicacao de pagina."""
    return bool(_RE_PAG_SIMPLES.search(trecho))


def avaliar_conformidade_citacoes_nbr10520(texto: str) -> dict:
    """Avalia conformidade basica de citacoes diretas e indiretas segundo a NBR 10520."""
    txt = str(texto)

    dir_paren = _RE_DIR_PAREN.findall(txt)
    dir_intro = _RE_DIR_INTRO.findall(txt)
    diretas_trechos = dir_paren + dir_intro

    dir_com_pag = sum(1 for t in diretas_trechos if _tem_pagina_simples(t))
    dir_sem_pag = len(diretas_trechos) - dir_com_pag
    n_dir = len(diretas_trechos)

    spans_dir = set()
    for m in _RE_DIR_PAREN.finditer(txt):
        spans_dir.add(m.span())
    for m in _RE_DIR_INTRO.finditer(txt):
        spans_dir.add(m.span())

    def _sobrepos(st: int, en: int) -> bool:
        return any(s0 <= st <= s1 or s0 <= en <= s1 for s0, s1 in spans_dir)

    ind_todos: list[str] = []
    vistos: set[str] = set()
    for padrao in (_RE_IND_PAREN, _RE_IND_TEXTUAL, _RE_IND_INTRO):
        for m in padrao.finditer(txt):
            if _sobrepos(m.start(), m.end()):
                continue
            chave = m.group(0).lower().strip()
            if chave not in vistos:
                vistos.add(chave)
                ind_todos.append(m.group(0))

    n_ind = len(ind_todos)
    total = n_dir + n_ind

    alertas: list[str] = []
    if dir_sem_pag > 0:
        alertas.append(
            f"Há {dir_sem_pag} citação(ões) direta(s) possivelmente sem indicação de página. "
            "A NBR 10520 exige indicação de página para citações diretas."
        )
    if total == 0:
        alertas.append("Nenhuma citação identificada no texto.")

    if total == 0:
        status = "não identificado"
        score_citacoes = 0.0
    elif dir_sem_pag > 0:
        status = "requer revisão"
        score_citacoes = 0.6
    else:
        status = "adequado"
        score_citacoes = 1.0

    return {
        "total": total,
        "status": status,
        "score_citacoes": score_citacoes,
        "diretas": {
            "count": n_dir,
            "com_pagina": dir_com_pag,
            "sem_pagina": dir_sem_pag,
        },
        "indiretas": {
            "count": n_ind,
        },
        "alertas": alertas,
    }


avaliar_citacoes_simples = avaliar_conformidade_citacoes_nbr10520

print("avaliar_conformidade_citacoes_nbr10520 carregada.")
print("  Alias avaliar_citacoes_simples = avaliar_conformidade_citacoes_nbr10520")


def _resultado_analise_vazio(texto_original: str, motivo: str) -> dict:
    """Retorna uma estrutura mínima e consistente quando o artigo não pode ser analisado."""
    texto_original = "" if texto_original is None else str(texto_original)
    diag_idioma = (
        diagnosticar_idioma_por_percentual(texto_original)
        if texto_original.strip()
        else {
            "idioma_predominante": "indefinido",
            "manter_dataset_pt": False,
        }
    )
    citacoes = avaliar_citacoes_simples(texto_original) if texto_original.strip() else {
        "total": 0,
        "status": "não identificado",
        "score_citacoes": 0.0,
        "diretas": {"count": 0, "com_pagina": 0, "sem_pagina": 0},
        "indiretas": {"count": 0},
        "alertas": ["Texto insuficiente para análise de citações."],
    }
    validacao = {
        "secoes": {},
        "_resumo": {
            "score_conformidade": 0,
            "conforme_nbr6022": False,
            "criticos": ["texto_insuficiente"],
            "avisos": [],
            "observacoes": [],
            "detalhes_hibrido": {},
        },
    }
    indicadores = agregar_indicadores(
        validacao,
        {"erro": motivo, "labels": [], "matriz": [], "media_por_secao": {}, "secoes_problemas": []},
        {"artigo_completo": {}, "por_secao": {}, "similaridade_lexical_media": np.nan, "jaccard_por_secao": {}},
        {"entidades_identificadas": {}, "autores_detectados": [], "organizacoes_detectadas": [], "validacao_com_metadados": {}},
        None,
    )
    indicadores["erro_pipeline"] = motivo
    return {
        "idioma": diag_idioma.get("idioma_predominante", "indefinido"),
        "status_idioma": "aceito" if diag_idioma.get("manter_dataset_pt") else "rejeitado",
        "secoes": {},
        "metricas": {
            "cosine_similarity": {},
            "bertscore": {},
            "jaccard_similarity": {},
            "score_abnt": indicadores.get("score_abnt_heuristico"),
            "score_semantico_hibrido_medio": indicadores.get("score_semantico_hibrido_medio"),
            "bertscore_f1_medio": indicadores.get("bertscore_f1_medio"),
            "jaccard_medio": indicadores.get("jaccard_medio"),
            "similaridade_lexical_media": indicadores.get("similaridade_lexical_media"),
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
            "erro_pipeline": motivo,
        },
        "ner": {
            "entidades_identificadas": {},
            "autores_detectados": [],
            "organizacoes_detectadas": [],
            "validacao_com_metadados": {},
            "entidades_cientificas_abstract": {},
        },
        "citacoes": citacoes,
        "_texto_estruturado": "",
        "_secoes_raw": {},
        "_validacao": validacao,
        "_lexico": {
            "artigo_completo": {},
            "por_secao": {},
            "similaridade_lexical_media": np.nan,
            "jaccard_por_secao": {},
            "jaccard_semantico_por_secao": {},
        },
        "_coerencia": {
            "erro": motivo,
            "labels": [],
            "matriz": [],
            "media_por_secao": {},
            "secoes_problemas": [],
        },
        "_indicadores": indicadores,
        "_erro": motivo,
    }


def analisar_artigo(
    texto_original: str,
    nlp=None,
    tokenizer=None,
    model=None,
    stopwords_pt: Optional[set] = None,
    threshold_semantica: float = 0.50,
    metadados: Optional[dict] = None,
    peso_heuristica: float = PESO_HEURISTICA,
    peso_bert_param: float = PESO_BERT,
    ngram_range: tuple = NGRAM_RANGE_PADRAO,
    ft_model=None,
    kw_model=None,
    ner_pipeline_lenerbr=None,
    ner_pipeline_scierc=None,
    zs_pipeline=None,
    progress_callback=None,
) -> dict:
    """Analisa um artigo completo e retorna saída estruturada.

    Parâmetros:
        metadados            : dict com "autores" e/ou "instituicoes" para NER.
        peso_heuristica      : peso lexical no híbrido (padrão 0.2).
        peso_bert_param      : peso semântico no híbrido (padrão 0.8).
        ngram_range          : (1,2) unigramas+bigramas | (2,2) bigramas puros.
        ft_model             : KeyedVectors FastText PT (Jaccard semântico). Opcional.
        kw_model              : instância KeyBERT (termos-chave por similaridade). Opcional.
        ner_pipeline_lenerbr  : pipeline HF de NER fine-tuned no LeNER-BR. Opcional.
        ner_pipeline_scierc   : pipeline HF de NER fine-tuned no SciERC (aplicado só ao abstract). Opcional.
        zs_pipeline           : pipeline de zero-shot classification (camada 6 da detecção). Opcional.

    Nota: `model` pode ser um SentenceTransformer (BERTimbau-STS) com
    `tokenizer=None` — convenção usada em toda a validação semântica híbrida
    (ver `_chunk_encode`).

    `progress_callback(etapa: str, fracao: float)` é chamado no início de
    cada etapa do pipeline (fracao em [0, 1], estimativa grosseira — as
    etapas têm durações muito diferentes). Exceções do callback são
    engolidas para nunca derrubar a análise.
    """
    def _progresso(etapa, fracao):
        if progress_callback is None:
            return
        try:
            progress_callback(etapa, fracao)
        except Exception:
            pass
    texto_original = "" if texto_original is None else str(texto_original)
    if len(texto_original.strip()) < 20:
        return _resultado_analise_vazio(
            texto_original,
            "Texto insuficiente para análise completa (mínimo de 20 caracteres úteis).",
        )

    texto_estrut = preparar_texto_para_estrutura(texto_original)
    if len(texto_estrut.split()) < 5:
        return _resultado_analise_vazio(
            texto_original,
            "Texto insuficiente após pré-processamento estrutural.",
        )

    _progresso("Detectando seções (camadas 1-6)", 0.05)
    try:
        secoes = detectar_secoes(texto_estrut, nlp=nlp, tokenizer=tokenizer, model=model, zs_pipeline=zs_pipeline)
    except Exception as e:
        return _resultado_analise_vazio(
            texto_original,
            f"Falha na detecção de seções: {e}"
        )

    _progresso("Validando estrutura NBR 6022 (semântica híbrida)", 0.45)
    try:
        validacao = validar_estrutura_abnt(
            secoes, texto_estrut, tokenizer, model,
            peso_heuristica, peso_bert_param
        )
    except Exception as e:
        validacao = {
            "secoes": {},
            "_resumo": {
                "score_conformidade": 0,
                "conforme_nbr6022": False,
                "criticos": ["falha_validacao_estrutura"],
                "avisos": [],
                "observacoes": [str(e)],
                "detalhes_hibrido": {},
            },
        }

    _progresso("Análise léxica (BoW / TF-IDF / Jaccard)", 0.65)
    try:
        lexico = analisar_lexico(
            texto_estrut, secoes,
            stopwords_pt=stopwords_pt,
            top_n=20,
            ngram_range=ngram_range,
            ft_model=ft_model,
            kw_model=kw_model,
        )
    except Exception as e:
        lexico = {
            "artigo_completo": {},
            "por_secao": {},
            "similaridade_lexical_media": np.nan,
            "jaccard_por_secao": {},
            "jaccard_semantico_por_secao": {},
            "_erro": f"analisar_lexico: {e}"
        }

    _progresso("Coerência semântica entre seções (BERTimbau)", 0.75)
    try:
        if model is not None:
            coerencia = analisar_coerencia_semantica(
                texto_estrut, secoes, tokenizer, model,
                threshold=threshold_semantica
            )
        else:
            coerencia = {"erro": "BERTimbau não carregado", "labels": [],
                         "matriz": [], "media_por_secao": {}, "secoes_problemas": []}
    except Exception as e:
        coerencia = {"erro": str(e), "labels": [], "matriz": [],
                     "media_por_secao": {}, "secoes_problemas": []}

    _progresso("Reconhecendo entidades (NER)", 0.85)
    try:
        entidades_bruto = extrair_entidades(
            preparar_texto_para_ner(texto_original), nlp,
            ner_pipeline_lenerbr=ner_pipeline_lenerbr,
        )
    except Exception as e:
        entidades_bruto = {"erro": str(e)}
    try:
        ner_resultado = validar_ner_com_metadados(entidades_bruto, metadados)
    except Exception as e:
        ner_resultado = {
            "entidades_identificadas": entidades_bruto if isinstance(entidades_bruto, dict) else {},
            "autores_detectados": [],
            "organizacoes_detectadas": [],
            "validacao_com_metadados": {},
            "erro": str(e),
        }

    _progresso("BERTScore por seção", 0.90)
    try:
        bertscore = (calcular_bertscore_secoes(texto_estrut, secoes)
                     if BERTSCORE_DISPONIVEL else None)
    except Exception as e:
        bertscore = {"_erro": str(e)}

    try:
        entidades_cientificas_abstract = extrair_entidades_cientificas_abstract(
            texto_estrut, secoes, ner_pipeline_scierc
        )
    except Exception as e:
        entidades_cientificas_abstract = {"_erro": str(e)}

    _progresso("Avaliando citações (NBR 10520)", 0.94)
    try:
        citacoes = avaliar_citacoes_simples(texto_original)
    except Exception as e:
        citacoes = {
            "total": 0,
            "status": "erro",
            "score_citacoes": 0.0,
            "diretas": {"count": 0, "com_pagina": 0, "sem_pagina": 0},
            "indiretas": {"count": 0},
            "alertas": [str(e)],
        }

    _progresso("Agregando indicadores", 0.97)
    try:
        indicadores = agregar_indicadores(
            validacao, coerencia, lexico, ner_resultado, bertscore,
            entidades_cientificas_abstract,
        )
    except Exception as e:
        indicadores = {
            "score_abnt_heuristico": 0,
            "conforme_nbr6022": False,
            "n_criticos": 1,
            "n_avisos": 0,
            "n_observacoes": 0,
            "score_semantico_hibrido_medio": np.nan,
            "n_secoes_problemas_semantica": np.nan,
            "media_similaridade_semantica": np.nan,
            "similaridade_lexical_media": np.nan,
            "jaccard_medio": np.nan,
            "bertscore_f1_medio": np.nan,
            "n_entidades_per": 0,
            "n_entidades_org": 0,
            "detalhes_hibrido": {},
            "erro_pipeline": str(e),
        }

    try:
        diag_idioma = diagnosticar_idioma_por_percentual(texto_original)
    except Exception:
        diag_idioma = {"idioma_predominante": "indefinido", "manter_dataset_pt": False}

    secoes_saida = validacao.get("secoes", {})
    for label, sec_data in secoes_saida.items():
        lexico_secao = lexico.get("por_secao", {}).get(label, {})
        jac_secao = lexico.get("jaccard_por_secao", {}).get(label, {})
        sec_data["bow_top_terms"] = lexico_secao.get("bow_top_terms", [])
        sec_data["tfidf_top_terms"] = lexico_secao.get("tfidf_top_terms", [])
        sec_data["jaccard_score"] = jac_secao.get("jaccard", None)
        sec_data["indicador_lexical"] = jac_secao.get("indicador_lexical", None)

    return {
        "idioma": diag_idioma.get("idioma_predominante", "indefinido"),
        "status_idioma": "aceito" if diag_idioma.get("manter_dataset_pt") else "rejeitado",
        "secoes": secoes_saida,
        "metricas": {
            "cosine_similarity": coerencia.get("media_por_secao", {}),
            "bertscore": bertscore or {},
            "jaccard_similarity": lexico.get("jaccard_por_secao", {}),
            "score_abnt": indicadores.get("score_abnt_heuristico"),
            "score_semantico_hibrido_medio": indicadores.get("score_semantico_hibrido_medio"),
            "bertscore_f1_medio": indicadores.get("bertscore_f1_medio"),
            "jaccard_medio": indicadores.get("jaccard_medio"),
            "similaridade_lexical_media": indicadores.get("similaridade_lexical_media"),
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
        },
        "ner": {
            "entidades_identificadas": ner_resultado.get("entidades_identificadas", {}),
            "autores_detectados": ner_resultado.get("autores_detectados", []),
            "organizacoes_detectadas": ner_resultado.get("organizacoes_detectadas", []),
            "validacao_com_metadados": ner_resultado.get("validacao_com_metadados", {}),
            "entidades_cientificas_abstract": entidades_cientificas_abstract,
        },
        "citacoes": citacoes,
        "_texto_estruturado": texto_estrut,
        "_secoes_raw": secoes,
        "_validacao": validacao,
        "_lexico": lexico,
        "_coerencia": coerencia,
        "_indicadores": indicadores,
        "_erro": indicadores.get("erro_pipeline", lexico.get("_erro", "")),
    }


def executar_pipeline_dataset(
    df_pt: pd.DataFrame,
    coluna_texto: str,
    nlp=None,
    tokenizer=None,
    model=None,
    stopwords_pt: Optional[set] = None,
    n_amostras: Optional[int] = 100,
    random_state: int = 42,
    threshold_semantica: float = 0.50,
    peso_heuristica: float = PESO_HEURISTICA,
    peso_bert_param: float = PESO_BERT,
    ngram_range: tuple = NGRAM_RANGE_PADRAO,
    coluna_metadados: Optional[str] = None,
    ft_model=None,
    kw_model=None,
    ner_pipeline_lenerbr=None,
    ner_pipeline_scierc=None,
    zs_pipeline=None,
) -> pd.DataFrame:
    """Executa analisar_artigo para cada linha do DataFrame e retorna métricas."""
    import json

    base = df_pt.dropna(subset=[coluna_texto]).copy()
    if n_amostras is not None:
        base = base.sample(n=min(n_amostras, len(base)), random_state=random_state)
    if base.empty:
        print("Nenhum artigo elegível encontrado após remover textos nulos.")
        return pd.DataFrame()

    linhas = []
    falhas = 0

    for idx, row in base.iterrows():
        item = {"indice_df": idx}
        try:
            metadados = row[coluna_metadados] if coluna_metadados else None
            if isinstance(metadados, str):
                try:
                    metadados = json.loads(metadados)
                except Exception:
                    metadados = None

            out = analisar_artigo(
                str(row[coluna_texto]),
                nlp=nlp, tokenizer=tokenizer, model=model,
                stopwords_pt=stopwords_pt,
                threshold_semantica=threshold_semantica,
                metadados=metadados,
                peso_heuristica=peso_heuristica,
                peso_bert_param=peso_bert_param,
                ngram_range=ngram_range,
                ft_model=ft_model,
                kw_model=kw_model,
                ner_pipeline_lenerbr=ner_pipeline_lenerbr,
                ner_pipeline_scierc=ner_pipeline_scierc,
                zs_pipeline=zs_pipeline,
            )

            ind = out["_indicadores"]
            cit = out.get("citacoes", {})
            item.update({
                "idioma": out["idioma"],
                "status_idioma": out["status_idioma"],
            })
            item.update(ind)
            resumo = out["_validacao"].get("_resumo", {})
            item["secoes_detectadas"] = ", ".join(
                [k for k in out["_secoes_raw"] if not k.startswith("_")]
            )
            item["n_secoes_detectadas"] = len(
                [k for k in out["_secoes_raw"] if not k.startswith("_")]
            )
            item["criticos_lista"] = str(resumo.get("criticos", []))
            item["avisos_lista"] = str(resumo.get("avisos", []))
            item["observacoes_lista"] = str(resumo.get("observacoes", []))
            item["citacoes_diretas"] = cit.get("diretas", {}).get("count", 0)
            item["citacoes_indiretas"] = cit.get("indiretas", {}).get("count", 0)
            item["citacoes_apud"] = 0
            item["score_citacoes"] = cit.get("score_citacoes", None)
            item["status_citacoes"] = cit.get("status", "")
            item["erro_pipeline"] = out.get("_erro", "")
        except Exception as e:
            falhas += 1
            item.update({
                "idioma": "indefinido",
                "status_idioma": "erro",
                "score_abnt_heuristico": np.nan,
                "conforme_nbr6022": False,
                "n_criticos": np.nan,
                "n_avisos": np.nan,
                "n_observacoes": np.nan,
                "score_semantico_hibrido_medio": np.nan,
                "n_secoes_problemas_semantica": np.nan,
                "media_similaridade_semantica": np.nan,
                "similaridade_lexical_media": np.nan,
                "jaccard_medio": np.nan,
                "bertscore_f1_medio": np.nan,
                "n_entidades_per": np.nan,
                "n_entidades_org": np.nan,
                "detalhes_hibrido": {},
                "secoes_detectadas": "",
                "n_secoes_detectadas": 0,
                "criticos_lista": "[]",
                "avisos_lista": "[]",
                "observacoes_lista": "[]",
                "citacoes_diretas": 0,
                "citacoes_indiretas": 0,
                "citacoes_apud": 0,
                "score_citacoes": np.nan,
                "status_citacoes": "erro",
                "erro_pipeline": str(e),
            })
        linhas.append(item)

    if falhas:
        print(f"Pipeline concluído com {falhas} falha(s) isolada(s). Consulte a coluna 'erro_pipeline'.")

    return pd.DataFrame(linhas)


print("analisar_artigo + executar_pipeline_dataset carregados.")
print("  Citações: avaliar_citacoes_simples (diretas e indiretas).")

def gerar_relatorio_interpretativo(indicadores, palavras_chave=None):
    """Gera relatório interpretativo textual com base nos indicadores do pipeline.

    Observação: as contagens de seções obrigatórias são calculadas apenas a partir
    de `detalhes_hibrido`, para não misturar seções com critérios estruturais gerais
    da ABNT, como título, resumo e palavras-chave.
    """
    if not isinstance(indicadores, dict) or not indicadores:
        return "## Diagnóstico geral\nNão foi possível gerar relatório interpretativo com os dados disponíveis.\n"

    EMOJI_STATUS = {
        STATUS_CONTEM:         "✅",
        STATUS_CONTEM_OBS:     "🔵",
        STATUS_REQUER_REVISAO: "⚠️",
        STATUS_NAO_CONTEM:     "🔴",
        "ok":         "✅",
        "observacao": "🔵",
        "aviso":      "⚠️",
        "critico":    "🔴",
    }
    SECOES_OBR = {"introducao", "metodologia", "resultados", "conclusao", "referencias"}

    def _as_num(v, default=0):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return int(v)
        except Exception:
            return default

    def _fmt(v):
        if v is None:
            return "n/d"
        try:
            if isinstance(v, float) and np.isnan(v):
                return "n/d"
        except Exception:
            pass
        try:
            return f"{float(v):.3f}"
        except Exception:
            return "n/d"

    partes = ["## Diagnóstico geral\n"]
    score = indicadores.get("score_abnt_heuristico", "-")
    conforme = indicadores.get("conforme_nbr6022", False)
    partes.append(f"Score ABNT heurístico: **{score}/100**. ")
    partes.append(f"Conformidade NBR 6022: **{'Sim' if conforme else 'Não'}**.\n")

    det = indicadores.get("detalhes_hibrido", {}) or {}
    det_obr = {k: v for k, v in det.items() if k in SECOES_OBR and isinstance(v, dict)}

    if det_obr:
        partes.append("\n## Validação semântica das seções obrigatórias\n")
        for label in ["introducao", "metodologia", "resultados", "conclusao", "referencias"]:
            info = det_obr.get(label)
            if not info:
                continue
            status = info.get("status", info.get("nivel", "?"))
            emoji = EMOJI_STATUS.get(status, "❓")
            s_sem = info.get("score_semantico")
            s_str = f"{s_sem:.3f}" if isinstance(s_sem, (int, float)) and not pd.isna(s_sem) else "n/a"
            metodo_estrut = info.get("metodo_deteccao_estrutural")
            metodo_sem = ", ".join(info.get("metodo_validacao_semantica", []) or [])
            metodo_desc = []
            if metodo_estrut:
                metodo_desc.append(f"detecção estrutural: {metodo_estrut}")
            if metodo_sem:
                metodo_desc.append(f"validação semântica: {metodo_sem}")
            partes.append(
                f"- {emoji} **{label}** — *{status}* | score={s_str}"
                f"{' | ' + ' | '.join(metodo_desc) if metodo_desc else ''}\n"
            )
            obs = info.get("mensagem", "")
            if obs:
                partes.append(f"  > {obs}\n")

    partes.append("\n## Métricas complementares\n")
    partes.append(f"- BERTScore F1 médio (seções vs. referência): **{_fmt(indicadores.get('bertscore_f1_medio'))}**\n")
    partes.append(f"- Jaccard médio (termos esperados vs encontrados): **{_fmt(indicadores.get('jaccard_medio'))}**\n")
    partes.append(f"- Coerência semântica média entre seções (BERTimbau): **{_fmt(indicadores.get('media_similaridade_semantica'))}**\n")
    partes.append(f"- Similaridade lexical média entre seções (TF-IDF): **{_fmt(indicadores.get('similaridade_lexical_media'))}**\n")
    partes.append(f"- Score semântico híbrido médio (seções obrigatórias): **{_fmt(indicadores.get('score_semantico_hibrido_medio'))}**\n")

    n_aut = _as_num(indicadores.get("n_entidades_per", 0))
    n_org = _as_num(indicadores.get("n_entidades_org", 0))
    if n_aut or n_org:
        partes.append(f"- NER: {n_aut} autor(es)/pessoa(s) e {n_org} organização(ões) detectados.\n")

    partes.append("\n## Situação das seções obrigatórias\n")
    n_nao_contem = 0
    n_requer_rev = 0
    n_contem_obs = 0
    n_presentes = 0

    for label, info in det_obr.items():
        status = info.get("status", "")
        if status == STATUS_CONTEM:
            n_presentes += 1
        elif status == STATUS_CONTEM_OBS:
            n_presentes += 1
            n_contem_obs += 1
        elif status == STATUS_REQUER_REVISAO:
            n_requer_rev += 1
        elif status == STATUS_NAO_CONTEM:
            n_nao_contem += 1

    total_obr = len(det_obr) if det_obr else 5
    partes.append(f"- **{n_presentes}/{total_obr}** seções obrigatórias identificadas.\n")
    if n_nao_contem > 0:
        partes.append(
            f"- **{n_nao_contem}** seção(ões) classificadas como *Não contém seção* — "
            "adicione ou desenvolva as seções ausentes.\n"
        )
    if n_requer_rev > 0:
        partes.append(
            f"- **{n_requer_rev}** seção(ões) classificadas como *Requer revisão* — "
            "o heading existe, mas o conteúdo está semanticamente fraco; desenvolva melhor.\n"
        )
    if n_contem_obs > 0:
        partes.append(
            f"- **{n_contem_obs}** seção(ões) classificadas como *Contém seção com observação* — "
            "o conteúdo foi validado, mas adicione o heading explícito para conformidade NBR 6022.\n"
        )
    if total_obr and n_presentes == total_obr:
        partes.append("- ✅ Todas as seções obrigatórias foram identificadas.\n")

    # Problemas estruturais gerais: remove apenas as ocorrências atribuídas às seções obrigatórias.
    n_crit_geral = _as_num(indicadores.get("n_criticos", 0))
    n_avi_geral = _as_num(indicadores.get("n_avisos", 0))
    n_obs_geral = _as_num(indicadores.get("n_observacoes", 0))
    ordem_correta = indicadores.get("ordem_secoes_correta")
    n_avi_ordem = 1 if ordem_correta is False else 0
    n_crit_estrut = max(0, n_crit_geral - n_nao_contem)
    n_avi_estrut = max(0, n_avi_geral - n_requer_rev - n_avi_ordem)
    n_obs_estrut = max(0, n_obs_geral - n_contem_obs)

    if n_crit_estrut > 0 or n_avi_estrut > 0 or n_obs_estrut > 0:
        partes.append("\n## Problemas estruturais ABNT\n")
        if n_crit_estrut > 0:
            partes.append(
                f"- **{n_crit_estrut}** elemento(s) estrutural(is) crítico(s) ausente(s) "
                "ou não detectado(s), como título, resumo ou palavras-chave.\n"
            )
        if n_avi_estrut > 0:
            partes.append(
                f"- **{n_avi_estrut}** elemento(s) estrutural(is) em aviso, como abstract, autores ou informações incompletas.\n"
            )
        if n_obs_estrut > 0:
            partes.append(
                f"- **{n_obs_estrut}** elemento(s) estrutural(is) com observação, como DOI, data de submissão ou outros metadados.\n"
            )

    pares_fora_de_ordem = indicadores.get("ordem_secoes_pares_fora_de_ordem") or []
    if pares_fora_de_ordem:
        partes.append("\n## Ordem das seções\n")
        for secao_esperada_antes, secao_que_veio_antes in pares_fora_de_ordem:
            partes.append(
                f"- A seção **{secao_que_veio_antes}** aparece antes de "
                f"**{secao_esperada_antes}** no texto, fora da ordem esperada pela NBR 6022.\n"
            )

    partes.append("\n## Sugestões de melhoria\n")
    if not conforme:
        partes.append("- O artigo **não está em conformidade** com a NBR 6022 segundo o score heurístico adotado.\n")
    if palavras_chave:
        partes.append(f"- Palavras-chave relevantes (TF-IDF): {', '.join(palavras_chave)}.\n")
    if conforme and total_obr and n_presentes == total_obr:
        partes.append("- O artigo está bem estruturado. Revise o texto para polimento final.\n")

    return "".join(partes)


gerar_feedback_interpretativo = gerar_relatorio_interpretativo

print("gerar_relatorio_interpretativo carregado.")
print("  Alias gerar_feedback_interpretativo = gerar_relatorio_interpretativo")


def gerar_saida_analise(resultado: dict) -> dict:
    """Formata a saída de analisar_artigo para análise, banco de dados e futuro front-end.

    Essa função é a saída padrão do pipeline. Ela serve para:
      - análise exploratória no notebook de visualização
      - salvamento em banco de dados (SQLite)
      - exportação e auditoria
      - futura integração com Streamlit (use gerar_saida_frontend como alias)

    Parâmetros
    ----------
    resultado : retorno de analisar_artigo().

    Saída
    -----
    score_geral        : int  (0–100)
    status_geral       : str
    idioma / status_idioma
    resumo_resultados  : {aprovados, observacoes, reprovados}
    secoes             : seções obrigatórias com indicadores lexicais e semânticos
    citacoes           : resumo de conformidade NBR 10520 (diretas e indiretas)
    metricas           : scores numéricos do pipeline
    ner                : entidades identificadas
    """
    resumo     = resultado.get("_validacao", {}).get("_resumo", {})
    secoes     = resultado.get("secoes", {})
    lexico     = resultado.get("_lexico", {})
    secoes_raw = resultado.get("_secoes_raw", {})
    cit_raw    = resultado.get("citacoes", {})

    score_raw = resumo.get("score_conformidade", 0)
    try:
        score = 0 if pd.isna(score_raw) else int(score_raw)
    except Exception:
        score = 0

    if score >= 85:
        status_geral = "Em conformidade"
    elif score >= 70:
        status_geral = "Necessita revisão"
    elif score >= 50:
        status_geral = "Necessita melhorias"
    else:
        status_geral = "Fora de conformidade"

    n_criticos    = len(resumo.get("criticos", []))
    n_avisos      = len(resumo.get("avisos", []))
    n_observacoes = len(resumo.get("observacoes", []))
    total_itens   = len(ORDEM_NBR6022)
    aprovados     = max(0, total_itens - n_criticos - n_avisos - n_observacoes)

    # ── Estrutura ABNT (itens pré-textuais) ───────────────────────────────
    _NIVEL_STATUS = {"critico": "não identificado", "aviso": "incompleto", "observacao": "verificar"}
    estrutura_abnt: dict = {}
    for s in ORDEM_NBR6022:
        if s in SECOES_OBRIGATORIAS_NBR:
            continue
        presente   = s in secoes_raw and not str(s).startswith("_")
        titulo_det = secoes_raw.get(f"_titulo_detectado_{s}", "")
        metodo     = secoes_raw.get(f"_metodo_{s}", "")
        if presente:
            estrutura_abnt[s] = {
                "status":           "adequado",
                "titulo_detectado": titulo_det,
                "metodo_deteccao":  metodo,
                "mensagem":         f"'{s}' identificado{' via ' + metodo if metodo else ''}.",
            }
        elif s in SECOES_OPCIONAIS_NBR:
            estrutura_abnt[s] = {
                "status":           "opcional ausente",
                "titulo_detectado": "",
                "metodo_deteccao":  "",
                "mensagem":         f"'{s}' é elemento opcional (NBR 6022); ausência sem impacto.",
            }
        else:
            nivel = SEVERIDADE_SECAO_BASE.get(s, "observacao")
            estrutura_abnt[s] = {
                "status":           _NIVEL_STATUS.get(nivel, "verificar"),
                "titulo_detectado": "",
                "metodo_deteccao":  "",
                "mensagem":         f"'{s}' ausente ou não detectado.",
            }

    # ── Seções obrigatórias com indicadores lexicais ───────────────────────
    secoes_saida: dict = {}
    for label in SECOES_OBRIGATORIAS_NBR:
        sec        = secoes.get(label, {})
        lexico_sec = lexico.get("por_secao", {}).get(label, {})
        jac_sec    = lexico.get("jaccard_por_secao", {}).get(label, {})
        titulo_det = secoes_raw.get(f"_titulo_detectado_{label}", "")
        metodo     = secoes_raw.get(f"_metodo_{label}", "")
        score_f    = sec.get("_score_fuzzy", secoes_raw.get(f"_score_fuzzy_{label}"))
        _status_label = sec.get("status", STATUS_NAO_CONTEM)
        secoes_saida[label] = {
            "presente":          _status_label in (STATUS_CONTEM, STATUS_CONTEM_OBS),
            "status":            _status_label,
            "tem_lexico":        sec.get("tem_lexico", False),
            "tem_semantico":     sec.get("tem_semantico", None),
            "score_lexical":     sec.get("score_lexical", 0.0),
            "score_semantico":   sec.get("score_semantico", None),
            "score_final":       sec.get("score_final", 0.0),
            "titulo_detectado":  titulo_det,
            "metodo_deteccao":   sec.get("metodo_deteccao", [metodo] if metodo else []),
            "score_fuzzy":       score_f,
            "bow_top_terms":     lexico_sec.get("bow_top_terms", []),
            "tfidf_top_terms":   lexico_sec.get("tfidf_top_terms", []),
            "jaccard_score":     jac_sec.get("jaccard", None),
            "indicador_lexical": jac_sec.get("indicador_lexical", None),
            "observacao":        sec.get("mensagem", ""),
        }

    # ── Citações NBR 10520 (avaliar_citacoes_simples) ─────────────────────
    citacoes_saida: dict = {
        "total":          cit_raw.get("total", 0),
        "status":         cit_raw.get("status", "não identificado"),
        "score_citacoes": cit_raw.get("score_citacoes", 0.0),
        "alertas":        cit_raw.get("alertas", []),
        "diretas": {
            "count":      cit_raw.get("diretas", {}).get("count", 0),
            "com_pagina": cit_raw.get("diretas", {}).get("com_pagina", 0),
            "sem_pagina": cit_raw.get("diretas", {}).get("sem_pagina", 0),
        },
        "indiretas": {
            "count": cit_raw.get("indiretas", {}).get("count", 0),
        },
    }

    # ── Relatório textual via feedback interpretativo ─────────────────────
    _indicadores = resultado.get("_indicadores", {})
    try:
        relatorio_textual = gerar_feedback_interpretativo(_indicadores)
    except Exception as _e:
        relatorio_textual = f"Erro ao gerar relatório: {_e}"

    _qtd_presentes = sum(
        1 for s in secoes_saida.values()
        if s.get("presente", False)
    )

    return {
        "score_geral":    score,
        "status_geral":   status_geral,
        "idioma":         resultado.get("idioma", ""),
        "status_idioma":  resultado.get("status_idioma", ""),

        "resumo_resultados": {
            "aprovados":   aprovados,
            "observacoes": n_observacoes,
            "reprovados":  n_criticos + n_avisos,
        },

        "ordem_secoes":       resumo.get("ordem_secoes", {}),
        "estrutura_abnt":     estrutura_abnt,
        "secoes":             secoes_saida,          # alias para compatibilidade
        "secoes_obrigatorias": secoes_saida,         # campo explícito
        "qtd_secoes_presentes": _qtd_presentes,
        "citacoes":           citacoes_saida,
        "metricas": {
            "score_abnt":                    resultado.get("metricas", {}).get("score_abnt"),
            "score_semantico_hibrido_medio":  resultado.get("metricas", {}).get("score_semantico_hibrido_medio"),
            "bertscore_f1_medio":            resultado.get("metricas", {}).get("bertscore_f1_medio"),
            "jaccard_medio":                 resultado.get("metricas", {}).get("jaccard_medio"),
            "similaridade_lexical_media":    resultado.get("metricas", {}).get("similaridade_lexical_media"),
            "cosine_similarity":             resultado.get("metricas", {}).get("cosine_similarity", {}),
            "bertscore":                     resultado.get("metricas", {}).get("bertscore", {}),
            "jaccard_similarity":            resultado.get("metricas", {}).get("jaccard_similarity", {}),
        },
        "ner":                resultado.get("ner", {}),
        "relatorio_textual":  relatorio_textual,
        "llm": resultado.get("llm", {
            "usada": False,
            "modelo": None,
            "status": "desativada",
            "resposta": None,
        }),
    }


# Alias para futura integração com Streamlit
gerar_saida_frontend = gerar_saida_analise

print("gerar_saida_analise carregada.")
print("  Alias gerar_saida_frontend = gerar_saida_analise (para futura integração Streamlit).")
print()
print("Fluxo previsto para o Streamlit (futuro):")
print("  arquivo = upload_do_usuario")
print("  texto   = extrair_texto(arquivo)")
print("  resultado = analisar_artigo(texto)")
print("  saida   = gerar_saida_analise(resultado)")
print("  exibir_saida_no_frontend(saida)")
