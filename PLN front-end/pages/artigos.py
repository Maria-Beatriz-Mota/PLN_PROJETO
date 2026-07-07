import json

import streamlit as st
from db import conectar, criar_tabelas
from utils.styles import load_css

st.set_page_config(layout="wide")
st.markdown(load_css(), unsafe_allow_html=True)

# Idempotente — necessário caso o usuário abra esta página direto,
# sem passar pela main.
criar_tabelas()

st.title("📚 Artigos analisados")

conn = conectar()
cursor = conn.cursor()

cursor.execute("""
    SELECT id, nome, data_upload, resultado_json
    FROM artigos
    ORDER BY data_upload DESC
""")
dados = cursor.fetchall()

if not dados:
    st.info(
        "Nenhum artigo analisado ainda. "
        "Use a página **Up Artigo** no menu lateral para começar."
    )
else:
    for artigo_id, nome, data_upload, resultado_json in dados:
        rotulo = f"{nome} ({data_upload})"
        # Artigos com resultado guardado mostram score/classificação na
        # lista e abrem instantaneamente; os demais serão reanalisados
        # (uma única vez) ao serem abertos.
        if resultado_json:
            try:
                r = json.loads(resultado_json)
                rotulo += f" — {r['score']:.0f}% · {r['classificacao']}"
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        # key único evita colisão quando dois artigos têm o mesmo nome
        if st.button(rotulo, key=f"artigo_{artigo_id}"):
            # Limpa qualquer resultado/estado de uma análise anterior antes de
            # carregar este artigo — sem isso, resultado.py reaproveitava o
            # `resultado` cacheado da última análise e mostrava o nome do
            # artigo certo com o score/seções de outro artigo.
            for k in ("resultado", "artigo_id_salvo", "texto", "nome_arquivo", "analisado"):
                st.session_state.pop(k, None)
            st.session_state["artigo_id"] = artigo_id
            st.switch_page("pages/resultado.py")