"""As contas da aula «PDP e ICE», nas entregas da Loja Exemplo.

    python3 pdp_ice.py                            # as seis partes da aula
    python3 pdp_ice.py --coluna transportadora    # a PDP e o ICE de outra coluna
    python3 pdp_ice.py --graficos                 # também desenha os gráficos (precisa do matplotlib)

Colunas aceitas em --coluna: distancia_km, peso_kg, prazo_prometido_dias e transportadora.
Tudo sai igual toda vez. Maio e junho de 2026 são o teste do curso: o programa nem lê essas entregas.
"""
import argparse
from pathlib import Path

import numpy as np

import dependencia as dp
import floresta as fl

AQUI = Path(__file__).resolve().parent
DADOS = AQUI.parent / "dados" / "entregas" / "entregas.csv"
SORTEIOS = [0, 1, 2, 3, 4]
EXEMPLO = [8234, 8218, 8277]
VALORES = {
    "distancia_km": [100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700,
                     2900, 3100, 3300, 3500],
    "peso_kg": [0.5, 1, 2, 3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 40],
    "prazo_prometido_dias": list(range(1, 15)),
    "transportadora": ["Economica", "Propria", "Rapida"],
}
GRUPO = {"distancia_km": "regiao", "peso_kg": "transportadora",
         "prazo_prometido_dias": "transportadora", "transportadora": "regiao"}
DE_ATE = {"distancia_km": (1300, 1700), "peso_kg": (1, 15), "prazo_prometido_dias": (3, 10),
          "transportadora": ("Economica", "Propria")}
UNIDADE = {"distancia_km": " km", "peso_kg": " kg", "prazo_prometido_dias": " dias", "transportadora": ""}
ORDEM_REGIOES = ["Sudeste", "Sul", "Centro-Oeste", "Nordeste", "Norte"]
ORDEM_TRANSP = ["Economica", "Rapida", "Propria"]


def ordem_grupo(grupo):
    return ORDEM_REGIOES if grupo == "regiao" else ORDEM_TRANSP


def rotulo(coluna, v):
    if coluna in fl.CATEGORIAS:
        return str(v)
    casas = 1 if (isinstance(v, float) and v != int(v)) else 0
    return fl.num(v, casas) + UNIDADE[coluna]


def pc(x):
    return fl.num(100 * x, 1)


def celula(valor, poucas):
    t = pc(valor)
    return f"({t})" if poucas else t


def tabela_dependencia(coluna, valores, notas, viz, cont_grupo, grupo, linhas, saida):
    """A PDP, a parte das entregas que existe e a média de cada grupo, valor por valor."""
    g = linhas[grupo].to_numpy()
    cats = ordem_grupo(grupo)
    existe = viz >= dp.MINIMO
    cab = f"{'valor':>11} {'PDP':>6} {'existe':>7} {'só onde existe':>15}  " + " ".join(f"{c:>13}" for c in cats)
    saida.append(cab)
    for k, v in enumerate(valores):
        so = notas[existe[:, k], k]
        so_txt = pc(so.mean()) if len(so) else "-"
        grupos = " ".join(f"{celula(notas[g == c, k].mean(), cont_grupo[c][k] < dp.MINIMO):>13}" for c in cats)
        saida.append(f"{rotulo(coluna, v):>11} {pc(notas[:, k].mean()):>6} {fl.num(100 * existe[:, k].mean(), 0) + '%':>7} "
                     f"{so_txt:>15}  {grupos}")
    partes = [f"mesma {coluna}" if coluna in fl.CATEGORIAS else f"mesma faixa de {coluna}"]
    for c in dp.COMPANHEIRAS[coluna]:
        partes.append(f"mesma {c}" if c in fl.CATEGORIAS else f"{c} até {fl.num(dp.TOLERANCIA[c], 0)} de diferença")
    saida.append(f"   PDP: a nota média, em %, das {len(linhas)} entregas com {coluna} trocada pelo valor da linha")
    saida.append(f"   existe: parte das entregas trocadas com pelo menos {dp.MINIMO} entregas parecidas no treino "
                 f"({'; '.join(partes)})")
    saida.append("   só onde existe: a média só dessas entregas")
    saida.append(f"   colunas de {grupo}: a média do grupo; entre parênteses, o treino tem menos de {dp.MINIMO} "
                 f"entregas do grupo nessa faixa")


def tres_entregas(valores, notas, viz, linhas, saida):
    """A conta da PDP à mão: três entregas de março com a distância trocada."""
    pontos_km = [100, 1500, 3100]
    cols = [valores.index(v) for v in pontos_km]
    saida.append("três entregas de março, com a distância trocada (nota em %; entre parênteses, não existe no treino)")
    saida.append(f"{'entrega':>8} {'região':>9} {'transportadora':>15} {'km de verdade':>14} "
                 + " ".join(f"{fl.num(v, 0) + ' km':>9}" for v in pontos_km))
    for i in EXEMPLO:
        k = int(np.flatnonzero(linhas["id_entrega"].to_numpy() == i)[0])
        lin = linhas.iloc[k]
        celulas = " ".join(f"{celula(notas[k, c], viz[k, c] < dp.MINIMO):>9}" for c in cols)
        saida.append(f"{i:>8} {lin['regiao']:>9} {lin['transportadora']:>15} {fl.num(lin['distancia_km']):>14} {celulas}")
    saida.append(f"{'PDP: a média das 1888 entregas':>48} " + " ".join(f"{pc(notas[:, c].mean()):>9}" for c in cols))
    saida.append("")


def tabela_de_ate(coluna, valores, notas, viz, grupo, linhas, saida):
    """Quanto a nota de cada entrega muda entre dois valores: o ICE lido entregas por entrega."""
    de, ate = DE_ATE[coluna]
    i, j = valores.index(de), valores.index(ate)
    d = notas[:, j] - notas[:, i]
    existe = (viz[:, i] >= dp.MINIMO) & (viz[:, j] >= dp.MINIMO)
    g = linhas[grupo].to_numpy()
    saida.append(f"de {rotulo(coluna, de)} para {rotulo(coluna, ate)}, entrega por entrega (pontos de nota)")
    saida.append(f"{'grupo':>13} {'entregas':>9} {'10%':>6} {'metade':>7} {'90%':>6} {'sobem 10+':>10} {'existem':>8}")
    for c in ["todas"] + ordem_grupo(grupo):
        m = np.ones(len(d), dtype=bool) if c == "todas" else g == c
        if not m.any():
            continue
        q = np.percentile(d[m], [10, 50, 90])
        saida.append(f"{c:>13} {int(m.sum()):>9} {fl.pontos(q[0]):>6} {fl.pontos(q[1]):>7} {fl.pontos(q[2]):>6} "
                     f"{fl.num(100 * (d[m] >= 0.10).mean(), 0) + '%':>10} {fl.num(100 * existe[m].mean(), 0) + '%':>8}")
    saida.append("   10%, metade, 90%: a mudança da nota em 10%, 50% e 90% das entregas do grupo, do menor para o maior")
    saida.append("   sobem 10+: parte das entregas cuja nota sobe 10 pontos ou mais")
    saida.append("   existem: parte das entregas com entregas parecidas no treino nos dois valores")
    return d


def treinar(treino, sorteio):
    xt, _ = fl.matriz(treino)
    return fl.Floresta(xt, treino["atrasou"].to_numpy(), arvores=100, colunas=4, min_folha=20, sorteio=sorteio)


def parte_floresta(treino, validacao, floresta, saida):
    xv, _ = fl.matriz(validacao)
    nv = floresta.nota(xv)
    yt, yv = treino["atrasou"].to_numpy(), validacao["atrasou"].to_numpy()
    saida.append("1. A floresta e as entregas")
    saida.append(f"treino: {len(treino)} entregas de julho de 2025 a fevereiro de 2026, {int(yt.sum())} atrasaram ({fl.pct(yt.mean())})")
    saida.append(f"março e abril de 2026: {len(validacao)} entregas, {int(yv.sum())} atrasaram ({fl.pct(yv.mean())}); "
                 "os gráficos usam essas entregas")
    saida.append("maio e junho, o teste, ficam fechados: o programa nem lê essas entregas")
    saida.append("   8 colunas: " + ", ".join(fl.COLUNAS))
    saida.append("   floresta: 100 árvores, 4 colunas sorteadas por nó, folha mínima 20, sorteio 0")
    saida.append(f"   acerto do par em março e abril: {fl.pct(fl.acerto_do_par(nv, yv))}; nota média {fl.pct(nv.mean())}")
    _, grupos = fl.matriz(treino.head(1))
    ultimo = {c: floresta.cortes[grupos[fl.COLUNAS.index(c)][0]][-1] for c in ("distancia_km", "peso_kg")}
    saida.append(f"   último corte que as árvores podem usar: distancia_km {fl.num(ultimo['distancia_km'])}, "
                 f"peso_kg {fl.num(ultimo['peso_kg'])}")
    return nv


def parte_onde_existem(treino, saida):
    saida.append("3. Onde as entregas existem: o treino por região")
    saida.append(f"{'região':>13} {'entregas':>9} {'menor km':>9} {'maior km':>9} {'atrasaram':>10} "
                 f"{'Economica':>10} {'Rapida':>7} {'Propria':>8}")
    for r in ORDEM_REGIOES:
        t = treino[treino["regiao"] == r]
        cont = t["transportadora"].value_counts()
        saida.append(f"{r:>13} {len(t):>9} {fl.num(t['distancia_km'].min()):>9} {fl.num(t['distancia_km'].max()):>9} "
                     f"{fl.pct(t['atrasou'].mean()):>10} {int(cont.get('Economica', 0)):>10} {int(cont.get('Rapida', 0)):>7} "
                     f"{int(cont.get('Propria', 0)):>8}")
    saida.append("   atrasaram: a parte das entregas do treino que atrasou ou se perdeu, a taxa medida, sem floresta")
    saida.append("   Economica, Rapida, Propria: entregas do treino por transportadora; Propria é a frota da loja")


def parte_combinacoes_peso(treino, validacao, saida):
    saida.append("5. O peso e os itens andam juntos")
    corr = treino["peso_kg"].corr(treino["itens"])
    saida.append(f"   correlação entre peso_kg e itens no treino: {fl.num(corr, 2)}")
    saida.append(f"{'peso':>7} {'itens no treino':>16} {'entregas trocadas com itens fora disso':>40}")
    for v in [1, 5, 15, 30]:
        baixo, alto = dp.faixas_dos_valores(VALORES["peso_kg"])
        k = VALORES["peso_kg"].index(v)
        na = treino[(treino["peso_kg"] > baixo[k]) & (treino["peso_kg"] <= alto[k])]
        lo, hi = int(na["itens"].min()), int(na["itens"].max())
        fora = ((validacao["itens"] < lo) | (validacao["itens"] > hi)).mean()
        saida.append(f"{rotulo('peso_kg', v):>7} {f'{lo} a {hi}':>16} {fl.num(100 * fora, 0) + '%':>40}")
    saida.append(f"   itens no treino: o menor e o maior número de itens das entregas do treino na faixa do peso")


def parte_ale(floresta, validacao, saida):
    cortes = [15, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3600]
    curva, passos, quantas = dp.ale(floresta, validacao, "distancia_km", cortes)
    pdp = dp.ice(floresta, validacao, "distancia_km", cortes).mean(axis=0)
    saida.append("6. Por dentro: a mesma distância pelo ALE, só com entregas reais")
    saida.append(f"{'faixa':>15} {'entregas':>9} {'passo ALE':>10} {'ALE':>6} {'PDP':>6}")
    saida.append(f"{'até ' + fl.num(cortes[0], 0) + ' km':>15} {'':>9} {'':>10} {fl.pontos(0.0):>6} {fl.pontos(0.0):>6}")
    for k in range(len(passos)):
        faixa = f"{fl.num(cortes[k], 0)} a {fl.num(cortes[k + 1], 0)}"
        saida.append(f"{faixa:>15} {quantas[k]:>9} {fl.pontos(passos[k]):>10} {fl.pontos(curva[k + 1]):>6} "
                     f"{fl.pontos(pdp[k + 1] - pdp[0]):>6}")
    saida.append("   passo ALE: a média de nota(fim da faixa) - nota(início), só nas entregas que já estão na faixa")
    saida.append(f"   ALE e PDP: quanto a nota subiu desde {fl.num(cortes[0], 0)} km, em pontos")
    return curva, cortes


def parte_sorteios(treino, validacao, saida):
    saida.append("7. Os 5 sorteios da floresta")
    dist = [100, 1500, 1700, 3100]
    lin = {v: [] for v in dist}
    salto = {c: [] for c in ORDEM_TRANSP}
    t = validacao["transportadora"].to_numpy()
    for s in SORTEIOS:
        f = treinar(treino, s)
        p = dp.ice(f, validacao, "distancia_km", dist).mean(axis=0)
        for v, x in zip(dist, p):
            lin[v].append(x)
        d = dp.ice(f, validacao, "peso_kg", [15]) - dp.ice(f, validacao, "peso_kg", [1])
        for c in ORDEM_TRANSP:
            salto[c].append(float(np.median(d[t == c, 0])))
    saida.append(f"{'conta':>42} {'menor':>6} {'maior':>6}")
    for v in dist:
        saida.append(f"{'PDP da distância em ' + fl.num(v, 0) + ' km (%)':>42} {pc(min(lin[v])):>6} {pc(max(lin[v])):>6}")
    for c in ORDEM_TRANSP:
        saida.append(f"{'peso de 1 para 15 kg, metade, ' + c:>42} {fl.pontos(min(salto[c])):>6} {fl.pontos(max(salto[c])):>6}")
    saida.append("   menor e maior: entre as florestas dos sorteios 0 a 4")


def parte_sorteios_coluna(coluna, numero, treino, validacao, saida):
    """A PDP e a mudança entre os dois valores de DE_ATE, nas florestas dos 5 sorteios."""
    valores = VALORES[coluna]
    grupo = GRUPO[coluna]
    de, ate = DE_ATE[coluna]
    g = validacao[grupo].to_numpy()
    pdp = {v: [] for v in valores}
    metade = {c: [] for c in ordem_grupo(grupo)}
    for s in SORTEIOS:
        notas = dp.ice(treinar(treino, s), validacao, coluna, valores)
        for k, v in enumerate(valores):
            pdp[v].append(notas[:, k].mean())
        d = notas[:, valores.index(ate)] - notas[:, valores.index(de)]
        for c in metade:
            metade[c].append(float(np.median(d[g == c])))
    saida.append(f"{numero}. Os 5 sorteios da floresta")
    saida.append(f"{'conta':>48} {'menor':>6} {'maior':>6}")
    for v in valores:
        saida.append(f"{'PDP em ' + rotulo(coluna, v) + ' (%)':>48} {pc(min(pdp[v])):>6} {pc(max(pdp[v])):>6}")
    for c in metade:
        txt = f"de {rotulo(coluna, de)} para {rotulo(coluna, ate)}, metade, {c}"
        saida.append(f"{txt:>48} {fl.pontos(min(metade[c])):>6} {fl.pontos(max(metade[c])):>6}")
    saida.append("   menor e maior: entre as florestas dos sorteios 0 a 4")


def graficos(coluna, valores, notas, viz, cont_grupo, grupo, linhas, pasta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        raise SystemExit("para os gráficos, instale o matplotlib: python3 -m pip install matplotlib")
    pasta.mkdir(exist_ok=True)
    cores = {"Sudeste": "#52645C", "Sul": "#C9822B", "Centro-Oeste": "#7B6BA8", "Nordeste": "#2F7FB0",
             "Norte": "#B5473A", "Economica": "#B5473A", "Rapida": "#2F7FB0", "Propria": "#52645C"}
    virgula = FuncFormatter(lambda x, _: fl.num(x, 0))
    g = linhas[grupo].to_numpy()
    existe = viz >= dp.MINIMO
    cats = ordem_grupo(grupo)
    arquivo = pasta / f"pdp_ice_{coluna}.png"
    if coluna in fl.CATEGORIAS:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        largura = 0.8 / len(valores)
        for k, v in enumerate(valores):
            for i, c in enumerate(cats):
                vazio = cont_grupo[c][k] < dp.MINIMO
                ax.bar(i + (k - 1) * largura, 100 * notas[g == c, k].mean(), largura * 0.95,
                       color="white" if vazio else cores[v], edgecolor=cores[v], hatch="//" if vazio else None,
                       label=v if i == 0 else None)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats)
        ax.set_ylabel("nota média (%)")
        ax.yaxis.set_major_formatter(virgula)
        ax.legend(title=coluna, frameon=False)
        ax.set_title(f"PDP de {coluna} por {grupo}, em março e abril de 2026", loc="left")
        fig.text(0.01, 0.01, f"Barra listrada: o treino tem menos de {dp.MINIMO} entregas dessa {grupo} com essa {coluna}.",
                 fontsize=9)
    else:
        x = np.asarray(valores, dtype=float)
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 3, 1.2]})
        a1.plot(x, 100 * notas.mean(axis=0), color="black", lw=2.5, label="PDP (todas)")
        so = [100 * notas[existe[:, k], k].mean() if existe[:, k].any() else np.nan for k in range(len(x))]
        a1.plot(x, so, color="black", lw=1.5, ls="--", label="só onde existe")
        for c in cats:
            y = np.array([100 * notas[g == c, k].mean() for k in range(len(x))])
            vazio = np.array([cont_grupo[c][k] < dp.MINIMO for k in range(len(x))])
            a1.plot(x, np.where(vazio, np.nan, y), color=cores[c], lw=1.5, label=c)
            a1.plot(x, np.where(vazio, y, np.nan), color=cores[c], lw=1, ls=":")
            a1.scatter(x[vazio], y[vazio], facecolors="white", edgecolors=cores[c], s=18, zorder=3)
        a1.set_ylabel("nota média (%)")
        a1.yaxis.set_major_formatter(virgula)
        a1.legend(frameon=False, fontsize=8, ncol=2)
        a1.set_title(f"PDP e média por {grupo}: {coluna}", loc="left")
        ref = 1 if coluna == "peso_kg" else 0
        rng = np.random.default_rng(0)
        amostra = rng.choice(len(linhas), size=min(150, len(linhas)), replace=False)
        for i in amostra:
            a2.plot(x, 100 * (notas[i] - notas[i, ref]), color=cores[g[i]], lw=0.6, alpha=0.45)
        a2.axhline(0, color="gray", lw=0.8)
        a2.set_ylabel(f"ICE centrado em {rotulo(coluna, valores[ref])} (pontos)")
        a2.yaxis.set_major_formatter(virgula)
        a2.set_title(f"ICE: 150 entregas sorteadas, cor por {grupo}", loc="left")
        a3.bar(x, 100 * existe.mean(axis=0), width=(x[1] - x[0]) * 0.8 if coluna != "peso_kg" else 0.8, color="#9AA59F")
        a3.set_ylim(0, 100)
        a3.set_ylabel("existe (%)")
        a3.set_xlabel(coluna)
        a3.xaxis.set_major_formatter(virgula)
        fig.text(0.01, 0.005, f"Pontilhado e círculo vazio: menos de {dp.MINIMO} entregas do grupo nessa faixa no treino. "
                 "Notas da floresta, não taxas de atraso medidas.", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(arquivo, dpi=120)
    plt.close(fig)
    return arquivo


def main():
    ap = argparse.ArgumentParser(description="PDP e ICE nas entregas de março e abril.")
    ap.add_argument("--dados", default=str(DADOS))
    ap.add_argument("--coluna", choices=sorted(VALORES), help="mostra a PDP e o ICE dessa coluna")
    ap.add_argument("--graficos", action="store_true", help="desenha os gráficos na pasta graficos/")
    ap.add_argument("--sem-sorteios", action="store_true", help="pula a parte 7, que treina mais 4 florestas")
    args = ap.parse_args()

    base = fl.ler(args.dados)
    treino, validacao = fl.separar(base)
    floresta = treinar(treino, 0)
    saida = []
    parte_floresta(treino, validacao, floresta, saida)
    pasta = AQUI / "graficos"
    feitos = []

    colunas = [args.coluna] if args.coluna else ["distancia_km", "peso_kg"]
    numero = 2
    for coluna in colunas:
        valores = VALORES[coluna]
        grupo = GRUPO[coluna]
        notas = dp.ice(floresta, validacao, coluna, valores)
        viz = dp.vizinhas(treino, validacao, coluna, valores)
        cont = dp.no_grupo(treino, coluna, valores, grupo, ordem_grupo(grupo))
        saida.append("")
        if args.coluna:
            saida.append(f"{numero}. PDP e ICE de {coluna}")
        elif coluna == "distancia_km":
            saida.append("2. A PDP da distância")
        else:
            numero = 4
            saida.append("4. O ICE do peso")
        if not args.coluna and coluna == "distancia_km":
            tres_entregas(valores, notas, viz, validacao, saida)
        tabela_dependencia(coluna, valores, notas, viz, cont, grupo, validacao, saida)
        saida.append("")
        tabela_de_ate(coluna, valores, notas, viz, grupo, validacao, saida)
        if not args.coluna and coluna == "distancia_km":
            saida.append("")
            parte_onde_existem(treino, saida)
        if args.coluna and coluna == "transportadora":
            saida.append("")
            saida.append(f"{numero + 1}. O prazo prometido por transportadora, no treino")
            saida.append(f"{'transportadora':>15} {'entregas':>9} {'menor prazo':>12} {'maior prazo':>12} {'média':>6}")
            for c in ORDEM_TRANSP:
                t = treino[treino["transportadora"] == c]["prazo_prometido_dias"]
                saida.append(f"{c:>15} {len(t):>9} {int(t.min()):>12} {int(t.max()):>12} {fl.num(t.mean()):>6}")
        if args.graficos:
            feitos.append((coluna, valores, notas, viz, cont, grupo))
        if args.coluna and not args.sem_sorteios:
            saida.append("")
            parte_sorteios_coluna(coluna, numero + (2 if coluna == "transportadora" else 1), treino, validacao, saida)
    if not args.coluna:
        saida.append("")
        parte_combinacoes_peso(treino, validacao, saida)
        saida.append("")
        parte_ale(floresta, validacao, saida)
        if not args.sem_sorteios:
            saida.append("")
            parte_sorteios(treino, validacao, saida)
    print("\n".join(saida))
    for coluna, valores, notas, viz, cont, grupo in feitos:
        arq = graficos(coluna, valores, notas, viz, cont, grupo, validacao, pasta)
        print(f"gráfico: graficos/{arq.name}")


if __name__ == "__main__":
    main()
