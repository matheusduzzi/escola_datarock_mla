"""Primeira olhada nas entregas da Loja Exemplo.

Rode de dentro desta pasta:

    python3 primeira_olhada.py

São três contas: o atraso por transportadora e peso do pacote, as 20 rotas que mais
atrasaram de julho a fevereiro medidas de novo em março e abril, e a regra
"toda entrega chega no prazo" nas quatro situações de entrega.
Maio e junho ficam guardados: nenhuma conta abaixo usa esses dois meses.
"""
import pandas as pd


def pct(parte, todo):
    return f"{100 * parte / todo:.1f}".replace(".", ",") + "%"


entregas = pd.read_csv("../dados/entregas/entregas.csv", parse_dates=["data_despacho"])
entregas["atrasou"] = entregas["situacao"] != "no_prazo"
ate_abril = entregas[entregas["data_despacho"] < "2026-05-01"]

print("1. Atraso por transportadora e peso, de julho a abril")
print(f"   {len(ate_abril)} entregas, {pct(ate_abril['atrasou'].sum(), len(ate_abril))} atrasaram ou se perderam")
nomes = {"Economica": "Econômica", "Rapida": "Rápida", "Propria": "Própria"}
for codigo, nome in nomes.items():
    t = ate_abril[ate_abril["transportadora"] == codigo]
    leve = t[t["peso_kg"] <= 10]["atrasou"]
    pesado = t[t["peso_kg"] > 10]["atrasou"]
    print(f"   {nome:<10} até 10 kg {pct(leve.sum(), len(leve)):>6} ({len(leve)})"
          f"   acima de 10 kg {pct(pesado.sum(), len(pesado)):>6} ({len(pesado)})")

print("2. As 20 rotas que mais atrasaram de julho a fevereiro")
antes = ate_abril[ate_abril["data_despacho"] < "2026-03-01"]
depois = ate_abril[ate_abril["data_despacho"] >= "2026-03-01"]
por_rota = antes.groupby("cod_rota")["atrasou"].mean()
piores = por_rota.sort_values(ascending=False, kind="mergesort").head(20).index
a = antes[antes["cod_rota"].isin(piores)]["atrasou"]
d = depois[depois["cod_rota"].isin(piores)]["atrasou"]
print(f"   {por_rota.size} rotas; de julho a fevereiro, todas juntas: {pct(antes['atrasou'].sum(), len(antes))}")
print(f"   as 20 piores, de julho a fevereiro: {pct(a.sum(), len(a))} ({len(a)} entregas)")
print(f"   as mesmas 20, em março e abril: {pct(d.sum(), len(d))} ({len(d)} entregas)")
print(f"   todas as rotas, em março e abril: {pct(depois['atrasou'].sum(), len(depois))}")

print("3. Quatro situações de entrega, de julho a abril")
contagem = ate_abril["situacao"].value_counts()
for s in ["no_prazo", "atraso_curto", "atraso_longo", "extravio"]:
    print(f"   {s:<13} {contagem[s]:>5} ({pct(contagem[s], len(ate_abril))})")
print(f"   toda entrega chega no prazo: acerta {contagem['no_prazo']} de {len(ate_abril)}"
      f" ({pct(contagem['no_prazo'], len(ate_abril))}), pega 0 de {contagem['extravio']} extravios")
