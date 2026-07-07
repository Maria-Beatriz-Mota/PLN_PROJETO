"""
Interface do analisador ABNT.

O front-end chama `analisar_abnt(texto)` e recebe um dicionário estruturado
seguindo o contrato `ResultadoAnalise` definido abaixo.

Adaptador para o pipeline real (BERTimbau-STS + detecção estrutural em 6
camadas + validação semântica híbrida + NER + citações NBR 10520), definido
em `pipeline_abnt.py` (extraído de pipeline_abnt_funcoes_oficial.ipynb).
`pages/resultado.py` não precisa de nenhuma mudança: o contrato de retorno
é o mesmo do mock anterior.
"""
import os
from pathlib import Path
from typing import Literal, TypedDict

import pipeline_abnt
from carregar_modelos import carregar_pipeline_modelos

# HF_TOKEN para o feedback via LLM (chamar_llm_analise_abnt). O .env do
# projeto vive em PLN_BACKEND\PLN_PROJETO_FINAL e usa a chave
# HUGGINGFACE_API_KEY; a função espera HF_TOKEN — aceitamos os dois nomes.
try:
    from dotenv import load_dotenv
    load_dotenv(
        Path(__file__).parent.parent / "PLN_BACKEND" / "PLN_PROJETO_FINAL" / ".env"
    )
except ImportError:
    pass

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")


# =========================
# CONTRATO DE RETORNO
# =========================
StatusSecao = Literal["ok", "aviso", "erro"]


class Secao(TypedDict):
    nome: str
    status: StatusSecao
    mensagem: str
    sugestao: str  # string vazia quando status == "ok"


class ErroDetalhado(TypedDict):
    tipo: str        # "Citação" | "Estrutura" | "Formatação" | ...
    trecho: str
    descricao: str
    sugestao: str


class ResultadoAnalise(TypedDict):
    score: float                          # 0-100
    classificacao: str                    # "Aprovado" | "Necessita Revisão" | "Reprovado"
    aprovados: int
    avisos: int
    reprovados: int
    secoes: list[Secao]                   # visão estrutural NBR 6022
    erros_detalhados: list[ErroDetalhado]
    feedback_llm: dict                    # {usada, modelo, origem, status, resposta}


# =========================
# Nomes de exibição das 20 seções de ORDEM_NBR6022
# =========================
_NOME_EXIBICAO: dict[str, str] = {
    "titulo": "Título",
    "autores": "Autores",
    "resumo": "Resumo",
    "palavras_chave": "Palavras-chave",
    "abstract": "Abstract",
    "titulo_outro": "Título em outro idioma",
    "data_submissao": "Data de submissão",
    "doi_disponib": "DOI",
    "introducao": "Introdução",
    "referencial_teorico": "Referencial teórico",
    "metodologia": "Metodologia",
    "resultados": "Resultados",
    "discussao": "Discussão",
    "implicacoes": "Implicações",
    "conclusao": "Conclusão",
    "referencias": "Referências",
    "glossario": "Glossário",
    "apendice": "Apêndice",
    "anexo": "Anexo",
    "agradecimentos": "Agradecimentos",
}

# Seções obrigatórias (secoes_obrigatorias): status vem das constantes
# STATUS_CONTEM/STATUS_CONTEM_OBS/STATUS_REQUER_REVISAO/STATUS_NAO_CONTEM de
# pipeline_abnt (ver validar_secao_obrigatoria_semantica).
_STATUS_OBRIGATORIA_PARA_FRONT: dict[str, StatusSecao] = {
    pipeline_abnt.STATUS_CONTEM: "ok",
    pipeline_abnt.STATUS_CONTEM_OBS: "aviso",
    pipeline_abnt.STATUS_REQUER_REVISAO: "aviso",
    pipeline_abnt.STATUS_NAO_CONTEM: "erro",
}

# Demais seções (estrutura_abnt): status vem de gerar_saida_analise
# ("adequado" | "verificar" | "incompleto" | "não identificado").
_STATUS_ESTRUTURA_PARA_FRONT: dict[str, StatusSecao] = {
    "adequado": "ok",
    "verificar": "aviso",
    "incompleto": "aviso",
    "não identificado": "erro",
}

_STATUS_GERAL_PARA_CLASSIFICACAO: dict[str, str] = {
    "Em conformidade": "Aprovado",
    "Necessita revisão": "Necessita Revisão",
    "Necessita melhorias": "Necessita Revisão",
    "Fora de conformidade": "Reprovado",
}


def _sugestao_para_status(nome: str, status: StatusSecao, obrigatoria: bool) -> str:
    if status == "ok":
        return ""
    if obrigatoria:
        if status == "erro":
            return (
                f"Inclua a seção '{nome}' no artigo, com conteúdo compatível "
                "com o esperado pela NBR 6022."
            )
        return (
            f"Revise a seção '{nome}': o conteúdo não está claramente "
            "alinhado com o que o título/heading sugere."
        )
    if status == "erro":
        return f"Inclua o item '{nome}', exigido pela NBR 6022 para este tipo de artigo."
    return f"Verifique o item '{nome}' — não foi possível confirmar sua presença com segurança."


def _montar_secoes(saida: dict) -> list[Secao]:
    """Monta a lista de Secao a partir das labels de ORDEM_NBR6022: as 5
    obrigatórias vêm de saida['secoes_obrigatorias'], as demais de
    saida['estrutura_abnt']. Elementos opcionais (glossário, apêndice,
    anexo, agradecimentos, título em outro idioma) ausentes são omitidos."""
    secoes_obrigatorias = saida.get("secoes_obrigatorias", {})
    estrutura_abnt = saida.get("estrutura_abnt", {})

    secoes: list[Secao] = []
    for label in pipeline_abnt.ORDEM_NBR6022:
        nome = _NOME_EXIBICAO.get(label, label.replace("_", " ").capitalize())
        obrigatoria = label in pipeline_abnt.SECOES_OBRIGATORIAS_NBR

        if obrigatoria:
            info = secoes_obrigatorias.get(label, {})
            status_bruto = info.get("status", pipeline_abnt.STATUS_NAO_CONTEM)
            status = _STATUS_OBRIGATORIA_PARA_FRONT.get(status_bruto, "erro")
            mensagem = info.get("observacao") or f"'{nome}' não avaliado(a)."
        else:
            info = estrutura_abnt.get(label, {})
            status_bruto = info.get("status", "não identificado")
            # Elementos opcionais da NBR 6022 ausentes não entram na lista
            # nem nas contagens — ausência deles não é cobrada.
            if status_bruto == "opcional ausente":
                continue
            status = _STATUS_ESTRUTURA_PARA_FRONT.get(status_bruto, "erro")
            mensagem = info.get("mensagem") or f"'{nome}' não avaliado(a)."

        sugestao = _sugestao_para_status(nome, status, obrigatoria)

        secoes.append({
            "nome": nome,
            "status": status,
            "mensagem": mensagem,
            "sugestao": sugestao,
        })
    return secoes


def _montar_erros_detalhados(saida: dict) -> list[ErroDetalhado]:
    """Junta alertas de citação (NBR 10520) e pares de seções fora de ordem
    (NBR 6022) num único formato de ocorrência pontual."""
    erros: list[ErroDetalhado] = []

    citacoes = saida.get("citacoes", {})
    for alerta in citacoes.get("alertas", []):
        erros.append({
            "tipo": "Citação",
            "trecho": "",
            "descricao": alerta,
            "sugestao": "Revise as citações do artigo conforme a NBR 10520.",
        })

    ordem_secoes = saida.get("ordem_secoes", {})
    for label_i, label_j in ordem_secoes.get("pares_fora_de_ordem", []):
        nome_i = _NOME_EXIBICAO.get(label_i, label_i)
        nome_j = _NOME_EXIBICAO.get(label_j, label_j)
        erros.append({
            "tipo": "Estrutura",
            "trecho": f"{nome_i} / {nome_j}",
            "descricao": (
                f"A seção '{nome_i}' aparece depois de '{nome_j}' no texto, "
                "fora da ordem esperada pela NBR 6022."
            ),
            "sugestao": f"Reordene o artigo para que '{nome_i}' venha antes de '{nome_j}'.",
        })

    return erros


def _adaptar_saida_para_resultado(saida: dict) -> ResultadoAnalise:
    """Converte a saída de `pipeline_abnt.gerar_saida_analise` para o
    contrato `ResultadoAnalise`. Função pura: não depende de modelos
    carregados, só do dict `saida` já pronto -- por isso é testável
    isoladamente com um `saida` sintético.
    """
    secoes = _montar_secoes(saida)
    erros_detalhados = _montar_erros_detalhados(saida)

    # Contagens recalculadas a partir da MESMA lista `secoes` (não de
    # saida["resumo_resultados"], que agrupa crítico+aviso num só bucket e
    # não bateria com os 3 status desta lista).
    aprovados = sum(1 for s in secoes if s["status"] == "ok")
    avisos_secao = sum(1 for s in secoes if s["status"] == "aviso")
    reprovados = sum(1 for s in secoes if s["status"] == "erro")
    avisos = avisos_secao + len(erros_detalhados)

    status_geral = saida.get("status_geral", "Fora de conformidade")
    classificacao = _STATUS_GERAL_PARA_CLASSIFICACAO.get(status_geral, "Reprovado")
    score = saida.get("score_geral", 0)

    return {
        "score": score,
        "classificacao": classificacao,
        "aprovados": aprovados,
        "avisos": avisos,
        "reprovados": reprovados,
        "secoes": secoes,
        "erros_detalhados": erros_detalhados,
        "feedback_llm": saida.get("llm", {}),
    }


# =========================
# ENTRY POINT
# =========================
def analisar_abnt(texto: str, progress_callback=None, usar_rag: bool = False) -> ResultadoAnalise:
    """
    Analisa um artigo contra as normas ABNT (NBR 6022 + NBR 10520) usando o
    pipeline real: BERTimbau-STS + detecção estrutural em 6 camadas +
    validação semântica híbrida + NER (LeNER-BR/SciERC) + zero-shot NLI +
    citações NBR 10520 (`pipeline_abnt.analisar_artigo` +
    `pipeline_abnt.gerar_saida_analise`).

    Args:
        texto: conteúdo integral do artigo (já extraído do PDF/DOCX/TXT).
        progress_callback: opcional, chamado como (etapa: str, fracao: float)
            no início de cada etapa. As frações são estimativas grosseiras
            de andamento, não tempo real — a etapa de modelos domina a
            primeira execução e depois fica instantânea (cache).

    Returns:
        ResultadoAnalise: dicionário com score, contagens, seções e erros.
    """
    def _prog(etapa: str, fracao: float) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(etapa, fracao)
        except Exception:
            pass

    _prog("Carregando modelos (a primeira análise demora mais)", 0.01)
    modelos = carregar_pipeline_modelos()
    resultado = pipeline_abnt.analisar_artigo(
        texto_original=texto, progress_callback=progress_callback, **modelos
    )

    # Feedback textual via LLM. Com usar_rag=True, recupera trechos dos
    # manuais de normalização ABNT (rag_abnt.py) relevantes aos problemas
    # detectados e injeta no prompt (rag_ativo=1 na persistência); sem RAG,
    # é a baseline (rag_ativo=0). Se o índice não existir ou a recuperação
    # falhar, degrada silenciosamente para a baseline.
    rag_contexto = None
    if usar_rag:
        _prog("Recuperando trechos das normas (RAG)", 0.98)
        try:
            import rag_abnt
            indice = rag_abnt.carregar_indice()
            consultas = rag_abnt.montar_consultas(resultado)
            trechos = rag_abnt.recuperar_trechos(
                consultas, modelos["model"], indice, top_k=2
            )
            rag_contexto = rag_abnt.montar_contexto_rag(trechos) or None
        except Exception:
            rag_contexto = None

    _prog("Gerando feedback via LLM (Hugging Face / Ollama)", 0.985)
    resultado["llm"] = pipeline_abnt.chamar_llm_analise_abnt(
        texto, resultado, hf_token=_HF_TOKEN, rag_contexto=rag_contexto
    )
    resultado["llm"]["rag_usado"] = bool(rag_contexto)

    _prog("Formatando resultado", 0.99)
    saida = pipeline_abnt.gerar_saida_analise(resultado)
    return _adaptar_saida_para_resultado(saida)
