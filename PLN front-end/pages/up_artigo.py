import streamlit as st
from utils.styles import load_css

st.set_page_config(layout="wide")
st.markdown(load_css(), unsafe_allow_html=True)

st.markdown("""
<div style="
    background-color: #007BFF;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 25px;
">
    📄 Assistente ABNT
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# =========================
# EXTRAÇÃO DE TEXTO
# =========================
def extrair_texto_do_upload(uploaded_file) -> str:
    """
    Extrai texto do arquivo enviado.

    Suporta:
        - PDF  → PyMuPDF (fitz)
        - DOCX → python-docx
        - TXT  → decode UTF-8 (fallback latin-1)

    Retorna string vazia se não conseguir extrair nada útil
    (ex.: PDF escaneado sem OCR).
    """
    if uploaded_file is None:
        return ""

    nome = uploaded_file.name.lower()
    dados = uploaded_file.read()

    if nome.endswith(".pdf"):
        import fitz  # PyMuPDF
        # Abre a partir dos bytes em memória (não precisa salvar em disco)
        with fitz.open(stream=dados, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)

    if nome.endswith(".docx"):
        import io
        import docx  # python-docx
        doc = docx.Document(io.BytesIO(dados))
        return "\n".join(p.text for p in doc.paragraphs)

    if nome.endswith(".txt"):
        try:
            return dados.decode("utf-8")
        except UnicodeDecodeError:
            return dados.decode("latin-1", errors="ignore")

    raise ValueError(f"Formato não suportado: {nome}")


# =========================
# UPLOAD
# =========================
st.markdown("### Avalie seu artigo científico")
uploaded_file = st.file_uploader(
    "Clique para selecionar um arquivo",
    type=["pdf", "docx", "txt"],
)

st.markdown("---")

# Texto (fallback / alternativa a colar direto)
st.markdown("### 📝 Ou cole o texto do artigo")
texto_colado = st.text_area(
    "Alternativa: cole aqui o conteúdo completo...",
    height=200,
)

st.caption(f"{len(texto_colado)} caracteres colados")


# =========================
# BOTÃO
# =========================
# O usuário sempre recebe o feedback fundamentado nas normas (RAG). A versão
# sem RAG é gerada apenas nos scripts de avaliação (comparação pareada), não
# no fluxo do site.
if st.button("✨ Analisar Artigo", use_container_width=True):

    if uploaded_file is None and texto_colado.strip() == "":
        st.warning("Por favor, envie um arquivo ou cole o texto.")
    else:
        # Arquivo enviado tem prioridade sobre texto colado
        if uploaded_file is not None:
            try:
                with st.spinner(f"Extraindo texto de {uploaded_file.name}..."):
                    texto_final = extrair_texto_do_upload(uploaded_file)
                nome_artigo = uploaded_file.name
            except Exception as e:
                st.error(f"Falha ao extrair texto do arquivo: {e}")
                st.stop()

            if not texto_final.strip():
                st.warning(
                    "Não foi possível extrair texto do arquivo. "
                    "Tente enviar em outro formato ou cole o texto abaixo."
                )
                st.stop()
        else:
            texto_final = texto_colado
            nome_artigo = "Texto colado"

        # Salva no session_state e navega
        st.session_state["texto"] = texto_final
        st.session_state["nome_arquivo"] = nome_artigo
        st.session_state["analisado"] = True
        # Invalida qualquer análise/persistência anterior desta sessão
        for k in ("resultado", "artigo_id", "artigo_id_salvo"):
            st.session_state.pop(k, None)

        st.success(
            f"Artigo carregado — {len(texto_final)} caracteres. Analisando..."
        )
        st.switch_page("pages/resultado.py")