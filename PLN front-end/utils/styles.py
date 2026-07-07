"""CSS compartilhado das páginas do Assistente ABNT.

`load_css()` devolve um bloco <style> pronto para st.markdown(..., unsafe_allow_html=True).
Classes usadas pelas páginas: main-card, score-box, small-box, header-gradient,
success, warning, error.
"""


def load_css() -> str:
    return """
<style>
/* Regra geral: todo elemento com background claro próprio também define
   color explícita (com !important, porque o tema escuro do Streamlit injeta
   seletores mais específicos) — senão o texto herda o branco do tema escuro
   e some no fundo branco. */
.main-card {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 10px;
    color: #1a1a1a !important;
}
.score-box {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #007BFF !important;
}
.small-box {
    background-color: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #1a1a1a !important;
}
.header-gradient {
    background: linear-gradient(90deg, #007BFF, #00C6FF);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: bold;
    margin: 10px 0;
}
.success {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    color: #155724 !important;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 6px 0;
}
.warning {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    color: #856404 !important;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 6px 0;
}
.warning code {
    color: #1a1a1a !important;
}
.error {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    color: #721c24 !important;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 6px 0;
}
</style>
"""
