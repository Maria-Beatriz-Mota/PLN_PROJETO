"""
Carregamento cacheado dos modelos do pipeline ABNT para o front-end Streamlit.

Sem @st.cache_resource, o Streamlit reexecuta o script inteiro a cada
interação do usuário e recarregaria spaCy/BERTimbau/KeyBERT/NER/zero-shot do
zero em toda rerun -- inviável em uso interativo. `carregar_pipeline_modelos`
carrega tudo uma unica vez por processo e devolve o mesmo dict cacheado nas
chamadas seguintes.
"""
import streamlit as st

import pipeline_abnt


@st.cache_resource(show_spinner="Carregando modelos do pipeline ABNT (primeira vez pode demorar)...")
def carregar_pipeline_modelos() -> dict:
    """Carrega todos os modelos usados por `pipeline_abnt.analisar_artigo` e
    devolve um dict pronto para ser passado como `**modelos` para essa função.
    """
    nlp = pipeline_abnt.carregar_modelo_spacy()
    tokenizer, model = pipeline_abnt.carregar_bertimbau_sts()
    # FastText PT desativado de proposito: o cc.pt.300.bin (7,2 GB, guardado
    # em PLN_BACKEND\PLN_PROJETO_FINAL\modelos_offline\) leva horas para
    # carregar nesta maquina (16 GB de RAM -> swap). Sem ele so o Jaccard
    # semantico fica de fora; o resto do pipeline segue normal (ft_model=None).
    # Para reativar um dia: converter para .kv com mmap e apontar o caminho.
    ft_model = pipeline_abnt.carregar_fasttext_pt("cc.pt.300.bin")
    kw_model = pipeline_abnt.carregar_keybert(sts_model=model)
    ner_pipeline_lenerbr = pipeline_abnt.carregar_ner_lenerbr()
    ner_pipeline_scierc = pipeline_abnt.carregar_ner_scierc()
    zs_pipeline = pipeline_abnt.carregar_zero_shot(usar=pipeline_abnt.USAR_ZERO_SHOT)

    return {
        "nlp": nlp,
        "tokenizer": tokenizer,
        "model": model,
        "ft_model": ft_model,
        "kw_model": kw_model,
        "ner_pipeline_lenerbr": ner_pipeline_lenerbr,
        "ner_pipeline_scierc": ner_pipeline_scierc,
        "zs_pipeline": zs_pipeline,
    }
