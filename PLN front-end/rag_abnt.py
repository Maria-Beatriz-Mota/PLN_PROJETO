"""
RAG sobre manuais públicos de normalização ABNT (NBR 6022 / NBR 10520).

Base de conhecimento: PDFs de manuais de bibliotecas universitárias (UFC,
PUC Minas, UFABC) em PLN_BACKEND\\normas_rag\\. Não são o texto oficial da
ABNT, mas resumos fiéis publicados abertamente — decisão do projeto para a
versão inicial do RAG; para trocar a base, basta substituir os PDFs e
reindexar (rodar `construir_indice` de novo).

Fluxo:
    construir_indice(pasta, modelo)   # 1x por base: extrai, fatia e embeda
    indice = carregar_indice(pasta)
    trechos = recuperar_trechos(consultas, modelo, indice, top_k=3)
    contexto = montar_contexto_rag(trechos)   # bloco pronto para o prompt

O `modelo` é o mesmo SentenceTransformer (BERTimbau-STS) já carregado pelo
app via carregar_modelos.py — nenhum modelo extra é necessário.
"""
import json
import re
from pathlib import Path

import numpy as np

PASTA_NORMAS_PADRAO = (
    Path(__file__).parent.parent / "PLN_BACKEND" / "normas_rag"
)
ARQ_EMBEDDINGS = "indice_normas_embeddings.npy"
ARQ_CHUNKS = "indice_normas_chunks.json"

TAMANHO_CHUNK = 700     # caracteres por trecho
SOBREPOSICAO = 150      # caracteres repetidos entre trechos vizinhos


def _extrair_texto_pdf(caminho: Path) -> str:
    import fitz

    with fitz.open(str(caminho)) as pdf:
        paginas = [pag.get_text("text") or "" for pag in pdf]
    texto = "\n".join(paginas)
    # normaliza espaços, preservando quebras de parágrafo
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def fatiar_em_chunks(texto: str, tamanho: int = TAMANHO_CHUNK,
                     sobreposicao: int = SOBREPOSICAO) -> list:
    """Janela deslizante em caracteres, cortando preferencialmente em fim de
    frase/parágrafo para não partir regras no meio."""
    chunks = []
    inicio = 0
    n = len(texto)
    while inicio < n:
        fim = min(inicio + tamanho, n)
        if fim < n:
            # tenta recuar até um fim de frase/parágrafo próximo
            janela = texto[inicio:fim]
            corte = max(janela.rfind(". "), janela.rfind(".\n"), janela.rfind("\n\n"))
            if corte > tamanho // 2:
                fim = inicio + corte + 1
        trecho = texto[inicio:fim].strip()
        if len(trecho.split()) >= 20:  # descarta fragmentos sem conteúdo útil
            chunks.append(trecho)
        if fim >= n:
            break
        inicio = fim - sobreposicao
    return chunks


def construir_indice(pasta=PASTA_NORMAS_PADRAO, modelo=None) -> dict:
    """Extrai, fatia e embeda todos os PDFs da pasta; salva o índice nela."""
    if modelo is None:
        raise ValueError("passe o SentenceTransformer (BERTimbau-STS) em `modelo`")
    pasta = Path(pasta)
    registros = []
    for pdf in sorted(pasta.glob("*.pdf")):
        texto = _extrair_texto_pdf(pdf)
        for c in fatiar_em_chunks(texto):
            registros.append({"fonte": pdf.stem, "trecho": c})
    if not registros:
        raise FileNotFoundError(f"nenhum PDF com conteúdo útil em {pasta}")

    embeddings = modelo.encode(
        [r["trecho"] for r in registros],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(pasta / ARQ_EMBEDDINGS, embeddings.astype(np.float32))
    with open(pasta / ARQ_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False)
    return {"registros": registros, "embeddings": embeddings}


def carregar_indice(pasta=PASTA_NORMAS_PADRAO):
    """Carrega o índice salvo; devolve None se ainda não foi construído."""
    pasta = Path(pasta)
    caminho_emb = pasta / ARQ_EMBEDDINGS
    caminho_chunks = pasta / ARQ_CHUNKS
    if not (caminho_emb.is_file() and caminho_chunks.is_file()):
        return None
    embeddings = np.load(caminho_emb)
    with open(caminho_chunks, encoding="utf-8") as f:
        registros = json.load(f)
    if len(registros) != len(embeddings):
        return None  # índice inconsistente: reconstruir
    return {"registros": registros, "embeddings": embeddings}


def recuperar_trechos(consultas: list, modelo, indice: dict,
                      top_k: int = 3, score_minimo: float = 0.30) -> list:
    """Top-k trechos por consulta (cosseno), com deduplicação entre consultas."""
    if not consultas or indice is None:
        return []
    emb_consultas = modelo.encode(
        list(consultas), show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    embeddings = indice["embeddings"]
    registros = indice["registros"]
    vistos, saida = set(), []
    for i, consulta in enumerate(consultas):
        scores = embeddings @ emb_consultas[i]
        ordem = np.argsort(-scores)[: top_k * 2]  # folga para a dedup
        aceitos = 0
        for j in ordem:
            if aceitos >= top_k:
                break
            if scores[j] < score_minimo:
                break
            if int(j) in vistos:
                continue
            vistos.add(int(j))
            saida.append({
                "consulta": consulta,
                "fonte": registros[j]["fonte"],
                "trecho": registros[j]["trecho"],
                "score": round(float(scores[j]), 4),
            })
            aceitos += 1
    return saida


def montar_consultas(resultado: dict) -> list:
    """Deriva as consultas de recuperação a partir dos problemas apontados
    pelo pipeline (retorno bruto de `analisar_artigo`, ANTES de
    gerar_saida_analise): seções obrigatórias com problema, alertas de
    citação e ordem das seções."""
    consultas = []
    for label, sec in (resultado.get("secoes") or {}).items():
        if not sec.get("presente", False) or sec.get("status") == "Requer revisão":
            consultas.append(
                f"seção {label} de artigo científico: obrigatoriedade e conteúdo esperado NBR 6022"
            )
    cit = resultado.get("citacoes", {})
    if cit.get("diretas", {}).get("sem_pagina", 0) > 0:
        consultas.append("citação direta indicação de página NBR 10520")
    if cit.get("total", 0) == 0:
        consultas.append("citações no texto autor data NBR 10520")
    ordem = (
        resultado.get("_validacao", {}).get("_resumo", {}).get("ordem_secoes", {})
    )
    if ordem and not ordem.get("ordem_correta", True):
        consultas.append("ordem das seções estrutura do artigo científico NBR 6022")

    # Artigo sem problemas detectados: ainda assim fundamenta o feedback com
    # as regras gerais — sem isso, a análise "com RAG" de um artigo bom seria
    # idêntica à baseline (contexto vazio) e o par de comparação seria falso.
    if not consultas:
        consultas = [
            "estrutura e seções obrigatórias do artigo científico NBR 6022",
            "regras de citação no texto sistema autor-data NBR 10520",
        ]
    return consultas[:5]  # limita o tamanho do contexto


def montar_contexto_rag(trechos: list, max_chars_por_trecho: int = 600) -> str:
    """Formata os trechos recuperados como bloco de contexto para o prompt."""
    if not trechos:
        return ""
    linhas = ["Trechos de manuais de normalização ABNT (use-os para fundamentar as recomendações, citando a norma correspondente):"]
    for t in trechos:
        linhas.append(f"\n[{t['fonte']}] {t['trecho'][:max_chars_por_trecho]}")
    return "\n".join(linhas)
