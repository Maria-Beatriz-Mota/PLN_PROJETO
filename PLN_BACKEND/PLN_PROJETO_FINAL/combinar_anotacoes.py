# -*- coding: utf-8 -*-
"""Combina a anotação HUMANA independente com a anotação ASSISTIDA (você+IA)
e produz:
  1. concordância entre anotadores (% e Cohen's kappa), global e por seção;
  2. lista de divergências para adjudicação final;
  3. gabarito_secoes_FINAL.csv — o gold consolidado (por padrão, a anotação
     humana prevalece nas divergências; ajuste em ADJUDICACAO se quiser).

Rode DEPOIS de preencher anotacao_humana_EM_BRANCO.csv (salve como
anotacao_humana.csv). Uso:  python combinar_anotacoes.py
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
HUMANA = BASE / "anotacao_humana.csv"                 # você preenche e salva assim
ASSISTIDA = BASE / "gabarito_secoes.csv"              # a versão você+IA
FINAL = BASE / "gabarito_secoes_FINAL.csv"
DIVERG = BASE / "divergencias_anotadores.csv"

# Adjudicação: nas divergências, quem prevalece no gold final.
# "humana" (padrão) | "assistida". Você pode editar caso a caso depois,
# direto no gabarito_secoes_FINAL.csv.
ADJUDICACAO = "humana"


def cohen_kappa(y1, y2):
    """Cohen's kappa para dois anotadores binários, sem dependências externas."""
    n = len(y1)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y1, y2)) / n
    p1 = sum(y1) / n
    p2 = sum(y2) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)


def main():
    if not HUMANA.is_file():
        sys.exit(f"Falta {HUMANA.name}. Preencha anotacao_humana_EM_BRANCO.csv "
                 f"(0/1 em cada célula) e salve como anotacao_humana.csv.")

    h = pd.read_csv(HUMANA, encoding="utf-8-sig").set_index("indice_df").sort_index()
    a = pd.read_csv(ASSISTIDA, encoding="utf-8-sig").set_index("indice_df").sort_index()
    secoes = [c for c in a.columns if c in h.columns and c != "revista"]

    comuns = h.index.intersection(a.index)
    h, a = h.loc[comuns], a.loc[comuns]

    # valida preenchimento
    faltando = []
    for s in secoes:
        col = pd.to_numeric(h[s], errors="coerce")
        if col.isna().any():
            faltando += [(idx, s) for idx in h.index[col.isna()]]
    if faltando:
        print(f"⚠️  {len(faltando)} células vazias/inválidas na anotação humana. "
              f"Exemplos: {faltando[:8]}")
        print("Preencha 0/1 em todas antes de consolidar.\n")

    linhas, divs = [], []
    yh_all, ya_all = [], []
    for s in secoes:
        yh = pd.to_numeric(h[s], errors="coerce").fillna(-1).astype(int).tolist()
        ya = pd.to_numeric(a[s], errors="coerce").fillna(-1).astype(int).tolist()
        pares = [(x, y) for x, y in zip(yh, ya) if x in (0, 1) and y in (0, 1)]
        if not pares:
            continue
        yh_v, ya_v = zip(*pares)
        concord = sum(x == y for x, y in pares) / len(pares)
        linhas.append({"secao": s, "n": len(pares),
                       "concordancia": round(concord, 3),
                       "kappa": cohen_kappa(list(yh_v), list(ya_v))})
        yh_all += list(yh_v); ya_all += list(ya_v)
        for idx in h.index:
            xh = pd.to_numeric(pd.Series([h.loc[idx, s]]), errors="coerce").iloc[0]
            xa = pd.to_numeric(pd.Series([a.loc[idx, s]]), errors="coerce").iloc[0]
            if pd.notna(xh) and pd.notna(xa) and int(xh) != int(xa):
                divs.append({"indice_df": idx, "secao": s,
                             "humana": int(xh), "assistida": int(xa)})

    met = pd.DataFrame(linhas).set_index("secao")
    print("== Concordância entre anotadores (humana vs assistida) ==")
    print(met.to_string())
    print(f"\nGLOBAL: concordância={sum(x==y for x,y in zip(yh_all,ya_all))/len(yh_all):.3f} "
          f"| kappa={cohen_kappa(yh_all, ya_all)} | {len(yh_all)} células")
    print(f"Divergências: {len(divs)}")

    pd.DataFrame(divs).to_csv(DIVERG, index=False, encoding="utf-8-sig")

    # Gold consolidado
    final = a.copy()
    for s in secoes:
        hs = pd.to_numeric(h[s], errors="coerce")
        for idx in final.index:
            if pd.notna(hs.loc[idx]):
                if ADJUDICACAO == "humana":
                    final.loc[idx, s] = int(hs.loc[idx])
                # se "assistida", mantém a de a (não faz nada)
    final.reset_index().to_csv(FINAL, index=False, encoding="utf-8-sig")
    print(f"\nGerados: {DIVERG.name} (divergências) e {FINAL.name} (gold final, "
          f"adjudicação={ADJUDICACAO}).")
    print("Revise as divergências e ajuste o FINAL se quiser antes de rodar as métricas.")


if __name__ == "__main__":
    main()
