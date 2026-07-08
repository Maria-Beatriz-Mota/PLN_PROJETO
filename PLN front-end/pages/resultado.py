import json

import streamlit as st
from db import conectar, criar_tabelas
from utils.styles import load_css
from analisador import analisar_abnt

st.set_page_config(layout="wide")
st.markdown(load_css(), unsafe_allow_html=True)

criar_tabelas()


# =========================
# CARREGAMENTO DO CONTEXTO
# =========================
# Dois cenários de entrada nesta página:
#   A) Análise nova: `up_artigo.py` colocou texto em session_state["texto"]
#   B) Artigo antigo: `artigos.py` colocou o ID em session_state["artigo_id"]

def _carregar_por_id(artigo_id: int) -> tuple[str, str, "str | None"]:
    conn = conectar()
    row = conn.execute(
        "SELECT nome, texto, resultado_json FROM artigos WHERE id = ?", (artigo_id,)
    ).fetchone()
    if not row:
        st.error("Artigo não encontrado no banco.")
        st.stop()
    return row[0], row[1], row[2]


artigo_id_carregado = None
nome_artigo = None
texto = None

if "artigo_id" in st.session_state:
    # Cenário B — carregamento de artigo salvo. Se a análise já foi feita
    # antes, o resultado guardado no banco é reutilizado (instantâneo);
    # artigos salvos antes dessa coluna existir são reanalisados uma única
    # vez abaixo e passam a ter o resultado guardado também.
    artigo_id_carregado = st.session_state["artigo_id"]
    nome_artigo, texto, _resultado_salvo = _carregar_por_id(artigo_id_carregado)
    if "resultado" not in st.session_state and _resultado_salvo:
        try:
            st.session_state["resultado"] = json.loads(_resultado_salvo)
        except json.JSONDecodeError:
            pass  # JSON corrompido: cai na reanálise normal abaixo
elif st.session_state.get("analisado"):
    # Cenário A — análise nova
    texto = st.session_state.get("texto", "")
    nome_artigo = st.session_state.get("nome_arquivo", "Texto colado")
else:
    st.warning(
        "⚠ Nenhum artigo analisado ainda. "
        "Use a página **Up Artigo** para começar."
    )
    st.stop()


# =========================
# ANÁLISE (cacheada em session_state)
# =========================
# Streamlit reexecuta o script inteiro a cada interação. Guardamos o
# resultado em session_state para não repetir a análise a cada rerun.
if "resultado" not in st.session_state:
    # Barra de progresso alimentada pelo callback de etapas do pipeline.
    # As porcentagens são estimativas por etapa (não tempo real): a etapa
    # "Carregando modelos" domina a primeira análise e some nas seguintes.
    barra_progresso = st.progress(0, text="Preparando análise...")

    def _atualizar_progresso(etapa: str, fracao: float) -> None:
        fracao = min(max(float(fracao), 0.0), 1.0)
        barra_progresso.progress(fracao, text=f"{etapa}... ({int(fracao * 100)}%)")

    st.session_state["resultado"] = analisar_abnt(
        texto,
        progress_callback=_atualizar_progresso,
        usar_rag=True,  # usuário sempre recebe o feedback com RAG
    )
    barra_progresso.progress(1.0, text="Análise concluída (100%)")
    barra_progresso.empty()

    resultado_json = json.dumps(
        st.session_state["resultado"], ensure_ascii=False
    )

    if (
        artigo_id_carregado is None
        and "artigo_id_salvo" not in st.session_state
    ):
        # Análise nova (não veio da lista): insere artigo + resultado + erros.
        conn = conectar()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO artigos (nome, texto, resultado_json) VALUES (?, ?, ?)",
            (nome_artigo, texto, resultado_json),
        )
        novo_id = cur.lastrowid
        for e in st.session_state["resultado"]["erros_detalhados"]:
            cur.execute(
                """INSERT INTO erros_abnt
                       (artigo_id, tipo, trecho, descricao, sugestao)
                   VALUES (?, ?, ?, ?, ?)""",
                (novo_id, e["tipo"], e["trecho"], e["descricao"], e["sugestao"]),
            )
        # Feedback via LLM: rag_ativo distingue a baseline (0, sem RAG) das
        # análises com RAG (1) — o mesmo artigo pode ter as duas versões na
        # tabela para comparação lado a lado.
        fb = st.session_state["resultado"].get("feedback_llm", {})
        if fb.get("usada"):
            cur.execute(
                """INSERT INTO feedback_llm
                       (artigo_id, rag_ativo, modelo, origem, status, resposta)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (novo_id, int(fb.get("rag_usado", False)), fb.get("modelo"),
                 fb.get("origem"), fb.get("status"), fb.get("resposta")),
            )
        conn.commit()
        st.session_state["artigo_id_salvo"] = novo_id
    elif artigo_id_carregado is not None:
        # Artigo antigo sem resultado guardado: acabou de ser reanalisado,
        # grava o resultado para as próximas aberturas serem instantâneas.
        conn = conectar()
        conn.execute(
            "UPDATE artigos SET resultado_json = ? WHERE id = ?",
            (resultado_json, artigo_id_carregado),
        )
        fb = st.session_state["resultado"].get("feedback_llm", {})
        if fb.get("usada"):
            conn.execute(
                """INSERT INTO feedback_llm
                       (artigo_id, rag_ativo, modelo, origem, status, resposta)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (artigo_id_carregado, int(fb.get("rag_usado", False)),
                 fb.get("modelo"), fb.get("origem"),
                 fb.get("status"), fb.get("resposta")),
            )
        conn.commit()

resultado = st.session_state["resultado"]


# =========================
# RESULTADO GERAL
# =========================
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown(f"### ✅ Análise concluída — {nome_artigo}")
    st.write(
        f"Conformidade com normas ABNT: **{resultado['classificacao']}**"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(
        f'<div class="main-card score-box">{resultado["score"]:.0f}%<br>'
        f'<span style="font-size:14px;">{resultado["classificacao"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =========================
# INDICADORES
# =========================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f'<div class="small-box">✔ {resultado["aprovados"]}<br>Aprovados</div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div class="small-box">⚠ {resultado["avisos"]}<br>Avisos</div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div class="small-box">❌ {resultado["reprovados"]}<br>Reprovados</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# =========================
# ESTRUTURA POR SEÇÃO (NBR 6022)
# =========================
st.markdown(
    '<div class="header-gradient">📘 Estrutura do Artigo — NBR 6022</div>',
    unsafe_allow_html=True,
)

for secao in resultado["secoes"]:
    if secao["status"] == "ok":
        st.markdown(
            f'<div class="success">✔ {secao["mensagem"]}</div>',
            unsafe_allow_html=True,
        )
    elif secao["status"] == "aviso":
        st.markdown(
            f"""<div class="warning">
                ⚠ {secao['mensagem']}<br>
                💡 Sugestão: {secao['sugestao']}
            </div>""",
            unsafe_allow_html=True,
        )
    else:  # erro
        st.markdown(
            f"""<div class="error">
                ❌ {secao['mensagem']}<br>
                💡 Sugestão: {secao['sugestao']}
            </div>""",
            unsafe_allow_html=True,
        )


# =========================
# OCORRÊNCIAS PONTUAIS (citações etc.)
# =========================
if resultado["erros_detalhados"]:
    st.markdown("")
    st.markdown(
        '<div class="header-gradient">🔍 Ocorrências específicas</div>',
        unsafe_allow_html=True,
    )
    for e in resultado["erros_detalhados"]:
        st.markdown(
            f"""<div class="warning">
                ⚠ <b>{e['tipo']}</b> — <code>{e['trecho']}</code><br>
                {e['descricao']}<br>
                💡 {e['sugestao']}
            </div>""",
            unsafe_allow_html=True,
        )


# =========================
# FEEDBACK VIA LLM (baseline sem RAG)
# =========================
feedback_llm = resultado.get("feedback_llm", {})
if feedback_llm.get("usada") and feedback_llm.get("resposta"):
    _origem_legivel = {
        "huggingface": "Hugging Face",
        "ollama_local": "Ollama (local)",
    }.get(feedback_llm.get("origem"), feedback_llm.get("origem") or "?")
    _sufixo_rag = (
        " · fundamentado em manuais de normalização ABNT (RAG)"
        if feedback_llm.get("rag_usado") else ""
    )
    with st.expander("🤖 Feedback complementar via LLM", expanded=True):
        st.write(feedback_llm["resposta"])
        st.caption(
            f"Gerado por {feedback_llm.get('modelo', '?')} via {_origem_legivel}{_sufixo_rag}"
        )
else:
    st.caption("Feedback via LLM não disponível nesta análise.")


# =========================
# BOTÃO VOLTAR
# =========================
st.markdown("---")
if st.button("🔙 Analisar outro artigo"):
    for k in (
        "texto", "nome_arquivo", "analisado",
        "resultado", "artigo_id", "artigo_id_salvo",
    ):
        st.session_state.pop(k, None)
    st.switch_page("pages/up_artigo.py")
