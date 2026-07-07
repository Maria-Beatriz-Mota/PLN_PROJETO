import streamlit as st
from utils.styles import load_css
from db import criar_tabelas


# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="Assistente ABNT", layout="wide")

# Garante que as tabelas do banco existam ao subir o app.
# Idempotente (usa CREATE TABLE IF NOT EXISTS), então roda sem problema
# em toda execução.
criar_tabelas()

st.markdown(load_css(), unsafe_allow_html=True)


# =========================
# CABEÇALHO
# =========================
st.markdown("""
<div style="
    background-color: #007BFF;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 25px;
">
    📄 Assistente ABNT<br>
    <span style="font-size:16px;">
        Sistema inteligente de avaliação de artigos científicos
    </span>
</div>
""", unsafe_allow_html=True)


# =========================
# ORIENTAÇÃO INICIAL
# =========================
st.write("")  # respiro
st.info("👈 Use o menu lateral para navegar entre as páginas.")
