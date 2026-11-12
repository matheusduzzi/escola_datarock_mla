"""As contas da aula «SHAP e LIME», nas entregas da Loja Exemplo.

    python3 shap_e_lime.py                          # as quatro partes da aula, menos de um minuto
    python3 shap_e_lime.py --entrega 9751           # as mesmas contas para outra entrega
    python3 shap_e_lime.py --entrega 9751 --entrega 9531

Com --entrega, o programa mostra a floresta e as entregas pedidas no lugar das quatro da aula.
A entrega precisa ser de março ou abril de 2026 (ids 8216 a 10103). Tudo sai igual toda vez.
"""
import argparse
from pathlib import Path

import numpy as np

import explicar as ex
import floresta as fl


DADOS = Path(__file__).resolve().parent.parent / "dados" / "entregas" / "entregas.csv"
DA_AULA = [8914, 9110, 9661, 8336]
SORTEIOS = [0, 1, 2, 3, 4]
FUNDO = 100
FUNDO_GRANDE = 1000
LARGURAS = [0.5, 5.0]
NOMES = {"distancia_km": "distância", "valor_frete": "frete", "prazo_prometido_dias": "prazo",
         "peso_kg": "peso", "itens": "itens", "transportadora": "transportadora",
         "vespera_feriado": "véspera"}


def p(x):
    return fl.pontos(x)


def faixa_txt(a, b):
    return f"{p(a)} a {p(b)}"


def descrever(linha):
    km = fl.num(linha["distancia_km"])
    kg = fl.num(linha["peso_kg"])
    fr = fl.num(linha["valor_frete"], 2)
    vesp = "sim" if linha["vespera_feriado"] == 1 else "não"
    return (f"{linha['transportadora']}, {linha['regiao']}, {km} km, {kg} kg, {linha['itens']} {'item' if linha['itens'] == 1 else 'itens'}, "
            f"frete R$ {fr}, prazo {linha['prazo_prometido_dias']} dias, véspera de feriado: {vesp}")


def top2(v):
    ordem = np.argsort(-np.abs(v), kind="stable")[:2]
    return [int(j) for j in ordem]


def nomes_top(v):
    return ", ".join(f"{NOMES[fl.COLUNAS[j]]} {p(v[j])}" for j in top2(v))


def main():
    ap = argparse.ArgumentParser(description="SHAP e LIME nas entregas de março e abril.")
    ap.add_argument("--dados", default=str(DADOS))
    ap.add_argument("--entrega", action="append", type=int, default=[],
                    help="id de uma entrega de março ou abril de 2026; pode repetir")
    args = ap.parse_args()

    base = fl.ler(args.dados)
    treino, validacao = fl.separar(base)
    ids = args.entrega or DA_AULA
    for i in ids:
        if i not in set(validacao["id_entrega"]):
            if i in set(treino["id_entrega"]):
                raise SystemExit(f"a entrega {i} é do treino (julho a fevereiro). Use uma de março ou abril: "
                                 f"ids {validacao['id_entrega'].min()} a {validacao['id_entrega'].max()}")
            raise SystemExit(f"a entrega {i} não está em março nem em abril de 2026. Use um id de "
                             f"{validacao['id_entrega'].min()} a {validacao['id_entrega'].max()}; "
                             "maio e junho são o teste do curso e ficam fechados")

    xt, grupos = fl.matriz(treino)
    xv, _ = fl.matriz(validacao)
    yt = treino["atrasou"].to_numpy()
    yv = validacao["atrasou"].to_numpy()
    floresta = fl.Floresta(xt, yt, arvores=100, colunas=3, min_folha=20, sorteio=0)
    nv = floresta.nota(xv)

    print("1. A floresta e as entregas")
    print(f"treino: {len(yt)} entregas de julho de 2025 a fevereiro de 2026, {yt.sum()} atrasaram ({fl.pct(yt.mean())})")
    print(f"março e abril: {len(yv)} entregas, {yv.sum()} atrasaram ({fl.pct(yv.mean())}); "
          f"maio e junho, o teste, ficam fechados")
    print(f"   7 colunas: {', '.join(fl.COLUNAS)}")
    print("   floresta: 100 árvores, 3 colunas sorteadas por nó, folha mínima 20, sorteio 0")
    print(f"   acerto do par em março e abril: {fl.pct(fl.acerto_do_par(nv, yv))}; "
          f"nota média em março e abril: {fl.pct(nv.mean())}")
    corr = treino[["peso_kg", "itens", "distancia_km", "valor_frete", "prazo_prometido_dias"]].corr()
    print(f"   no treino: véspera de feriado em {fl.pct(treino['vespera_feriado'].mean())} das entregas; correlação "
          f"peso e itens {fl.num(corr.loc['peso_kg', 'itens'], 2)}, distância e frete "
          f"{fl.num(corr.loc['distancia_km', 'valor_frete'], 2)}, distância e prazo "
          f"{fl.num(corr.loc['distancia_km', 'prazo_prometido_dias'], 2)}")
    linhas = {}
    for i in ids:
        k = int(np.flatnonzero(validacao["id_entrega"].to_numpy() == i)[0])
        linhas[i] = k
        r = validacao.iloc[k]
        print(f"entrega {i}, {r['data_despacho'].strftime('%d/%m/%Y')}: {descrever(r)}")
        print(f"   nota {fl.pct(nv[k])}; atrasou: {'sim' if yv[k] == 1 else 'não'} ({r['situacao']})")

    fundos = [ex.sortear_fundo(xt, FUNDO, s) for s in SORTEIOS]
    fundo_grande = ex.sortear_fundo(xt, FUNDO_GRANDE, 0)
    res = {}

    print()
    print("2. SHAP: a nota de cada entrega contra um fundo sorteado do treino")
    print(f"   fundo: {FUNDO} entregas do treino sorteadas sem repetir; 5 sorteios (0 a 4) e um fundo de "
          f"{FUNDO_GRANDE} (sorteio 0)")
    print("   parcelas em pontos de nota; as 128 combinações das 7 colunas, sem estimativa")
    print(f"   nota média da floresta nas {len(yt)} entregas do treino: {fl.pct(floresta.nota(xt).mean())}")
    for i in ids:
        k = linhas[i]
        contas = [ex.shap_exato(floresta, xv[k], f, grupos) for f in fundos]
        b_g, phi_g, nota_g = ex.shap_exato(floresta, xv[k], fundo_grande, grupos)
        phis = np.array([c[1] for c in contas])
        bases = np.array([c[0] for c in contas])
        res[i] = {"shap": phis[0], "shap_todos": phis, "shap_grande": phi_g}
        print(f"entrega {i}, nota {fl.pct(contas[0][2])}")
        print(f"{'coluna':<24}{'sorteio 0':>10}{'nos 5 sorteios':>18}{'fundo 1000':>12}")
        for j, c in enumerate(fl.COLUNAS):
            print(f"{c:<24}{p(phis[0, j]):>10}{faixa_txt(phis[:, j].min(), phis[:, j].max()):>18}{p(phi_g[j]):>12}")
        print(f"{'valor base':<24}{fl.pct(bases[0]):>10}"
              f"{(fl.pct(bases.min()) + ' a ' + fl.pct(bases.max())):>18}{fl.pct(b_g):>12}")
        print(f"   valor base + soma das parcelas = nota, nos 6 fundos (sorteio 0: {fl.pct(bases[0])} "
              f"{p(phis[0].sum())} = {fl.pct(bases[0] + phis[0].sum())})")
        ordens = {tuple(top2(v)) for v in list(phis) + [phi_g]}
        print(f"   as duas maiores: {nomes_top(phis[0])}; "
              + ("a mesma dupla, na mesma ordem, nos 6 fundos" if len(ordens) == 1
                 else f"{len(ordens)} duplas diferentes nos 6 fundos"))

    print()
    print("3. LIME: uma reta ajustada em vizinhas sorteadas de cada entrega")
    print(f"   5000 vizinhas por sorteio, faixas nos quartis do treino, largura padrão "
          f"{fl.num(ex.largura_padrao(), 2)}; 5 sorteios (0 a 4) e duas larguras (sorteio 0)")
    print("   coeficiente: quanto a reta soma à nota quando a coluna cai na faixa da entrega, em pontos")
    for c, u in (("distancia_km", "km"), ("peso_kg", "kg")):
        q = np.unique(np.percentile(treino[c].to_numpy(dtype=float), [25, 50, 75]))
        print(f"   quartis de {c} no treino: {', '.join(fl.num(v, 2) for v in q[:-1])} e {fl.num(q[-1], 2)} {u}")
    for w in [ex.largura_padrao()] + LARGURAS:
        pesos = [np.sqrt(np.exp(-k / w ** 2)) for k in (1, 2, 7)]
        print(f"   largura {fl.num(w, 2) if w == ex.largura_padrao() else fl.num(w, 1).replace(',0', '')}: uma vizinha com 1, 2 e 7 colunas fora da faixa conta "
              f"{', '.join(fl.num(v, 2) for v in pesos)}")
    for i in ids:
        k = linhas[i]
        r = validacao.iloc[k]
        viz = [ex.sortear_vizinhanca(floresta, treino, r, 5000, s) for s in SORTEIOS]
        lims = [ex.lime(floresta, treino, r, vizinhanca=v) for v in viz]
        larg = [ex.lime(floresta, treino, r, largura=w, vizinhanca=viz[0]) for w in LARGURAS]
        coef = np.array([d["coef"] for d in lims])
        res[i].update({"lime": coef[0], "lime_todos": coef, "lime_larg": [d["coef"] for d in larg]})
        print(f"entrega {i}, nota {fl.pct(lims[0]['nota'])}")
        print(f"{'faixa da entrega':<40}{'sorteio 0':>10}{'nos 5 sorteios':>18}"
              f"{'largura 0,5':>13}{'largura 5':>11}")
        for j, c in enumerate(fl.COLUNAS):
            print(f"{ex.faixa_de(treino, c, r[c]):<40}{p(coef[0, j]):>10}"
                  f"{faixa_txt(coef[:, j].min(), coef[:, j].max()):>18}"
                  f"{p(larg[0]['coef'][j]):>13}{p(larg[1]['coef'][j]):>11}")
        r2 = [d["r2"] for d in lims]
        fora = (1 - viz[0][1]).sum(axis=1)
        print(f"   vizinhas do sorteio 0 com no máximo 1 coluna fora da faixa: {int((fora <= 1).sum())} de 5000")
        print(f"   R² da reta nas vizinhas: {fl.num(r2[0], 2)} (de {fl.num(min(r2), 2)} a {fl.num(max(r2), 2)} nos 5 sorteios)")
        sem = ex.lime(floresta, treino, r, vizinhanca=viz[0], alfa=0.0)
        print(f"   a reta na entrega: {fl.pct(lims[0]['local'])}; sem a penalidade (alfa 0): {fl.pct(sem['local'])}; "
              f"a floresta: {fl.pct(lims[0]['nota'])}")
        duplas = [tuple(top2(v)) for v in coef]
        print(f"   as duas maiores: {nomes_top(coef[0])}; "
              + ("a mesma dupla nos 5 sorteios" if len(set(duplas)) == 1
                 else f"{len(set(duplas))} duplas diferentes nos 5 sorteios")
              + f"; largura 0,5: {nomes_top(larg[0]['coef'])}; largura 5: {nomes_top(larg[1]['coef'])}")

    print()
    print("4. Lado a lado")
    print(f"{'entrega':<9}{'nota':>7}  {'SHAP, as duas maiores':<34}{'LIME, as duas maiores':<38}"
          f"{'1ª igual':<10}{'dupla igual'}")
    for i in ids:
        s, l_ = res[i]["shap"], res[i]["lime"]
        ts, tl = top2(s), top2(l_)
        print(f"{i:<9}{fl.pct(nv[linhas[i]]):>7}  {nomes_top(s):<34}{nomes_top(l_):<38}"
              f"{('sim' if ts[0] == tl[0] else 'não'):<10}{('sim' if set(ts) == set(tl) else 'não')}")
    print("   SHAP: fundo de 100, sorteio 0; LIME: largura padrão, sorteio 0")
    print(f"{'entrega':<9}{'coluna':<16}{'SHAP':>8}{'LIME':>8}   sinal")
    for i in ids:
        s, l_ = res[i]["shap"], res[i]["lime"]
        for j in sorted(set(top2(s)) | set(top2(l_))):
            mesmo = "igual" if np.sign(round(100 * s[j], 1)) == np.sign(round(100 * l_[j], 1)) else "trocado"
            print(f"{i:<9}{NOMES[fl.COLUNAS[j]]:<16}{p(s[j]):>8}{p(l_[j]):>8}   {mesmo}")

    print()
    print("5. A primeira coluna em cada conta")
    print(f"{'entrega':<9}{'SHAP, nos 6 fundos':<28}{'LIME, nos 5 sorteios':<28}{'LIME, largura 0,5 e 5'}")
    for i in ids:
        sh = [first(v) for v in list(res[i]["shap_todos"]) + [res[i]["shap_grande"]]]
        li = [first(v) for v in res[i]["lime_todos"]]
        la = [first(v) for v in res[i]["lime_larg"]]
        print(f"{i:<9}{contar(sh, 6):<28}{contar(li, 5):<28}{la[0]}, {la[1]}")
    print("   primeira: a coluna de maior valor, sem olhar o sinal")


def first(v):
    j = int(np.argmax(np.abs(v)))
    return NOMES[fl.COLUNAS[j]] + (" (−)" if v[j] < 0 else "")


def contar(nomes, total):
    vistos = {}
    for n in nomes:
        vistos[n] = vistos.get(n, 0) + 1
    if len(vistos) == 1:
        return f"{nomes[0]} nos {total}"
    return "; ".join(f"{n} em {k}" for n, k in vistos.items())


if __name__ == "__main__":
    main()
