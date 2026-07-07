import streamlit as st
import sqlite3
from pathlib import Path

# Caminho absoluto baseado neste arquivo — evita quebrar quando
# o `streamlit run` é executado de outro diretório.
DB_PATH = str(Path(__file__).parent / "base_abnt.db")


@st.cache_resource
def conectar():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def criar_tabelas():
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS artigos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT,
        texto TEXT,
        data_upload TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS erros_abnt (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        artigo_id INTEGER,
        tipo TEXT,
        trecho TEXT,
        descricao TEXT,
        sugestao TEXT,
        FOREIGN KEY (artigo_id) REFERENCES artigos(id)
    )
    """)

    # Feedback textual gerado por LLM para cada análise. rag_ativo distingue
    # a baseline atual (0, sem RAG) das análises futuras com RAG (1) — assim
    # o MESMO artigo pode ter as duas versões na tabela e ser comparado
    # lado a lado (WHERE artigo_id = ? ORDER BY data_geracao).
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback_llm (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        artigo_id INTEGER,
        rag_ativo INTEGER NOT NULL DEFAULT 0,
        modelo TEXT,
        origem TEXT,
        status TEXT,
        resposta TEXT,
        data_geracao TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (artigo_id) REFERENCES artigos(id)
    )
    """)

    # Migração idempotente: bancos criados antes da coluna resultado_json
    # ganham a coluna aqui (SQLite não tem IF NOT EXISTS para colunas).
    # Nela fica o ResultadoAnalise completo em JSON, para reabrir artigos
    # já analisados sem reexecutar o pipeline.
    colunas = [c[1] for c in cursor.execute("PRAGMA table_info(artigos)").fetchall()]
    if "resultado_json" not in colunas:
        cursor.execute("ALTER TABLE artigos ADD COLUMN resultado_json TEXT")

    conn.commit()
    # NÃO fechar aqui: a conexão é cacheada por @st.cache_resource
    # e precisa continuar viva para os próximos usos.
