# Relatorio de Resultados - Pipeline ABNT (Entrega Parcial)

Relatorio gerado automaticamente a partir do notebook analise_pipeline_abnt_apresentacao_1.ipynb.

- Total de artigos no dataset: 2963
- Artigos mantidos no filtro de idioma (PT): 420
- Artigos fora do criterio PT: 2543

## Graficos

### Distribuicao apos filtro de idioma
![Distribuicao apos filtro de idioma](./PLN_SCIELO/resultados_pipeline/relatorio_entrega_parcial/grafico_01_filtro_idioma.png)

### Distribuicao do numero de secoes detectadas
![Distribuicao do numero de secoes detectadas](./PLN_SCIELO/resultados_pipeline/relatorio_entrega_parcial/grafico_02_secoes_detectadas.png)

### Top termos BoW e TF-IDF
![Top termos BoW e TF-IDF](./PLN_SCIELO/resultados_pipeline/relatorio_entrega_parcial/grafico_03_bow_tfidf.png)

## Saidas tabulares

- [saida_df_sec_head20.csv](./PLN_SCIELO/resultados_pipeline/relatorio_entrega_parcial/saida_df_sec_head20.csv)
- [saida_top_bow.csv](./PLN_SCIELO/resultados_pipeline/relatorio_entrega_parcial/saida_top_bow.csv)
- [saida_top_tfidf.csv](./PLN_SCIELO/resultados_pipeline/relatorio_entrega_parcial/saida_top_tfidf.csv)
- [saida_df_resultados_pipeline.csv](./PLN_SCIELO/resultados_pipeline/relatorio_entrega_parcial/saida_df_resultados_pipeline.csv)
- [saida_resumo_faltas.csv](./PLN_SCIELO/resultados_pipeline/relatorio_entrega_parcial/saida_resumo_faltas.csv)

## Saida de feedback interpretativo (exemplo)

## Diagnóstico geral
Score ABNT heurístico: **83/100**. Conformidade NBR 6022: **Sim**. Seções detectadas (13): abstract, autores, conclusao, data_submissao, discussao, doi_disponib, introducao, metodologia, palavras_chave, referencias, resultados, resumo, titulo

## Problemas encontrados
Não foram encontrados problemas críticos de estrutura. Foram encontrados **1** avisos. Foram feitas **6** observações. 
Palavras-chave relevantes (TF-IDF): geriatria, geriatria gerontologia, gerontologia, concepção, bruno luciano. 

## Sugestões de melhoria
- Considere os avisos para aprimorar o texto.
- Certifique-se de que as palavras-chave estejam alinhadas ao conteúdo do artigo.
