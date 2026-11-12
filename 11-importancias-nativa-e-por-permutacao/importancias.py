"""As contas da aula «Importâncias nativa e por permutação», nas entregas da Loja Exemplo.

    python3 importancias.py                          # as seis partes da aula, menos de 2 minutos
    python3 importancias.py --grupo peso_kg,itens    # acrescenta um grupo embaralhado junto
    python3 importancias.py --tirar itens            # acrescenta a floresta treinada sem a coluna
    python3 importancias.py --so-extras --grupo peso_kg,itens --tirar itens

--grupo e --tirar podem aparecer mais de uma vez. Com --so-extras, o programa pula as partes
da aula e mostra só a floresta de referência e os pedidos extras. Tudo sai igual toda vez.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import medidas as m

DADOS = Path(__file__).resolve().parent.parent / "dados" / "entregas" / "entregas.csv"
SORTEIOS = [0, 1, 2, 3, 4]
REPETICOES = 10
ARVORES = 100
COLUNAS_POR_NO = 4
FOLHA = 20
SORTEADA = "numero_sorteado"
PARES = [("distancia_km", "valor_frete"), ("peso_kg", "itens"), ("distancia_km", "prazo_prometido_dias")]


def preparar(base, tirar=()):
    """Matrizes em faixas, com a coluna sorteada no fim (ela fica igual em todas as florestas)."""
    nomes, d = m.separar(base, tirar)
    xt, yt = d["treino"]
    xv, yv = d["validacao"]
    rng = np.random.default_rng(400)
    sorteada = rng.integers(1, 401, len(base)).astype(float)
    corte = len(yt)
    xt = np.c_[xt, sorteada[:corte]]
    xv = np.c_[xv, sorteada[corte:]]
    nomes = nomes + [SORTEADA]
    cortes = m.faixas(xt)
    return nomes, m.em_faixas(xt, cortes), yt, m.em_faixas(xv, cortes), yv, xt


def florestas(ft, yt, folha=FOLHA):
    return [m.treinar_floresta(ft, yt, ARVORES, COLUNAS_POR_NO, folha, s) for s in SORTEIOS]


def posicoes(valores):
    """Posição de cada coluna, 1 = a maior."""
    ordem = np.argsort(-np.asarray(valores), kind="stable")
    pos = np.empty(len(ordem), dtype=int)
    pos[ordem] = np.arange(1, len(ordem) + 1)
    return pos


def pontos(x, casas=2):
    """Queda do acerto do par em pontos percentuais, com sinal."""
    v = 100 * x
    return ("-" if v < -0.5 * 10 ** -casas else " ") + m.num(abs(v), casas)


def permutar(fl, f, y, grupos, sorteio):
    return m.importancia_permutacao(fl, f, y, grupos, REPETICOES, sorteio)


def grupo_por_nome(nomes, texto):
    cols = [c.strip() for c in texto.split(",") if c.strip()]
    faltam = [c for c in cols if c not in nomes]
    if faltam or not cols:
        raise SystemExit(f"--grupo: coluna que não existe: {', '.join(faltam) or texto}. Colunas: {', '.join(nomes)}")
    return cols, [nomes.index(c) for c in cols]


def main():
    p = argparse.ArgumentParser(description="Importância nativa e por permutação nas entregas.")
    p.add_argument("--dados", default=str(DADOS))
    p.add_argument("--grupo", action="append", default=[], help="colunas separadas por vírgula, embaralhadas juntas")
    p.add_argument("--tirar", action="append", default=[], help="coluna que sai do treino")
    p.add_argument("--so-extras", action="store_true", help="pula as partes da aula")
    args = p.parse_args()

    base = m.ler(args.dados)
    nomes, ft, yt, fv, yv, xt = preparar(base)
    ncol = len(nomes)
    for t in args.tirar:
        if t not in nomes or t == SORTEADA:
            raise SystemExit(f"--tirar: coluna que não existe: {t}. Colunas: {', '.join(nomes[:-1])}")
    grupos_extra = [grupo_por_nome(nomes, g) for g in args.grupo]
    if args.so_extras and not (grupos_extra or args.tirar):
        raise SystemExit("--so-extras precisa de pelo menos um --grupo ou --tirar")

    print("1. As entregas e a floresta")
    print(f"treino: {len(yt)} entregas de julho de 2025 a fevereiro de 2026, {yt.sum()} atrasaram ({m.pct(yt.mean())})")
    print(f"validação: {len(yv)} entregas de março e abril de 2026, {yv.sum()} atrasaram ({m.pct(yv.mean())})")
    print("maio e junho, o teste, ficam fechados: nenhuma conta usa essas entregas")
    print(f"   {ncol} colunas: as 16 de sempre, cod_rota como número de 1 a 400 e {SORTEADA}, de 1 a 400 sorteado ao acaso")
    print(f"   floresta: {ARVORES} árvores, {COLUNAS_POR_NO} colunas sorteadas por nó, folha mínima {FOLHA}, sorteios 0 a 4")
    fls = florestas(ft, yt)
    acertos = [m.acerto_do_par(m.nota_floresta(fl, fv), yv) for fl in fls]
    print(f"   acerto do par em março e abril: {m.pct(np.mean(acertos))} na média dos 5 sorteios "
          f"(de {m.pct(min(acertos))} a {m.pct(max(acertos))})")

    nat = np.array([m.importancia_nativa(fl, ncol) for fl in fls])          # sorteio × coluna
    grupos = [[j] for j in range(ncol)]
    perm = np.array([permutar(fl, fv, yv, grupos, 100 + s)[1] for fl, s in zip(fls, SORTEIOS)])  # sorteio × coluna × rep
    perm_media = perm.mean(axis=2)                                           # sorteio × coluna
    j_sort = ncol - 1

    if not args.so_extras:
        print()
        print("2. A importância nativa: quanto o Gini baixou em cada coluna, no treino")
        print(f"{'coluna':<28}{'nativa':>8}{'posição':>9}   valores diferentes no treino")
        media_nat = nat.mean(axis=0)
        pos_nat = posicoes(media_nat)
        for j in np.argsort(-media_nat, kind="stable"):
            print(f"{nomes[j]:<28}{m.num(media_nat[j], 3):>8}{pos_nat[j]:>9}   {len(np.unique(xt[:, j]))}")
        print("   nativa: média dos 5 sorteios; cada floresta soma 1")
        fl1 = m.treinar_floresta(ft, yt, ARVORES, COLUNAS_POR_NO, 1, 0)
        nat1 = m.importancia_nativa(fl1, ncol)
        pos1 = posicoes(nat1)
        jr = nomes.index("cod_rota")
        ac1 = m.acerto_do_par(m.nota_floresta(fl1, fv), yv)
        print(f"   com folha mínima 1 (sorteio 0): acerto {m.pct(ac1)}; cod_rota {m.num(nat1[jr], 3)}, posição {pos1[jr]}; "
              f"{SORTEADA} {m.num(nat1[j_sort], 3)}, posição {pos1[j_sort]}")

        print()
        print("3. A importância por permutação: embaralhar a coluna em março e abril e medir de novo")
        print(f"{'coluna':<28}{'queda':>8}{'posição':>9}{'nativa':>9}{'no treino':>11}")
        pm = perm_media.mean(axis=0)
        pos_p = posicoes(pm)
        perm_treino = permutar(fls[0], ft, yt, grupos, 200)[1].mean(axis=1)
        for j in np.argsort(-pm, kind="stable"):
            print(f"{nomes[j]:<28}{pontos(pm[j]):>8}{pos_p[j]:>9}{pos_nat[j]:>9}{pontos(perm_treino[j]):>11}")
        print("   queda: quanto o acerto do par caiu, em pontos; média de 10 embaralhamentos em cada um dos 5 sorteios")
        print("   no treino: a mesma conta nas entregas de julho a fevereiro, que a floresta viu (sorteio 0)")
        d1 = perm[0, :, :]
        print(f"   de um embaralhamento para outro, na mesma floresta, a queda varia em média {m.num(100 * d1.std(axis=1).mean(), 2)} ponto "
              f"(desvio, sorteio 0)")

        print()
        print("4. Colunas que andam juntas")
        for a_, b_ in PARES:
            ia, ib = nomes.index(a_), nomes.index(b_)
            print(f"   correlação entre {a_} e {b_} no treino: {m.num(np.corrcoef(xt[:, ia], xt[:, ib])[0, 1], 2)}")
        print(f"{'embaralhando':<38}{'queda':>8}")
        idd, idf = nomes.index("distancia_km"), nomes.index("valor_frete")
        juntas = np.mean([permutar(fl, fv, yv, [[idd, idf]], 300 + s)[1].mean() for fl, s in zip(fls, SORTEIOS)])
        for rot, q in (("distancia_km sozinha", pm[idd]), ("valor_frete sozinha", pm[idf]),
                       ("distancia_km e valor_frete juntas", juntas)):
            print(f"{rot:<38}{pontos(q):>8}")
        print("   sozinha: a queda da parte 3; juntas: as duas embaralhadas com a mesma ordem, uma combinando com a outra")
        print()
        print(f"{'floresta treinada sem':<24}{'acerto':>8}{'perde':>8}   queda da que fica, por permutação")
        ref = np.mean(acertos)
        for tirar, fica in (("valor_frete", "distancia_km"), ("distancia_km", "valor_frete")):
            nm, f2t, y2t, f2v, y2v, _ = preparar(base, (tirar,))
            fl2 = florestas(f2t, y2t)
            ac2 = np.mean([m.acerto_do_par(m.nota_floresta(fl, f2v), y2v) for fl in fl2])
            jf = nm.index(fica)
            q2 = np.mean([permutar(fl, f2v, y2v, [[jf]], 400 + s)[1].mean() for fl, s in zip(fl2, SORTEIOS)])
            antes = pm[nomes.index(fica)]
            print(f"{tirar:<24}{m.pct(ac2):>8}{pontos(ref - ac2):>8}   {fica}: {pontos(antes).strip()} com as duas, "
                  f"{pontos(q2).strip()} sem {tirar}")
        print("   acerto e perde: média dos 5 sorteios; perde = acerto com as 18 colunas menos acerto sem a coluna, em pontos")

        print()
        print("5. A estabilidade: os 5 sorteios da floresta lado a lado")
        regua = perm_media[:, j_sort].max()
        print(f"   régua: a maior queda média de {SORTEADA} nos 5 sorteios, {pontos(regua).strip()} ponto")
        print(f"{'coluna':<28}{'nativa':>8}{'posições':>10}{'queda':>8}{'menor':>8}{'maior':>8}{'posições':>10}{'acima da régua':>16}")
        pos_nat_s = np.array([posicoes(v) for v in nat])
        pos_p_s = np.array([posicoes(v) for v in perm_media])
        for j in np.argsort(-pm, kind="stable"):
            acima = int((perm_media[:, j] > regua).sum())
            print(f"{nomes[j]:<28}{m.num(nat[:, j].mean(), 3):>8}{f'{pos_nat_s[:, j].min()} a {pos_nat_s[:, j].max()}':>10}"
                  f"{pontos(pm[j]):>8}{pontos(perm_media[:, j].min()):>8}{pontos(perm_media[:, j].max()):>8}"
                  f"{f'{pos_p_s[:, j].min()} a {pos_p_s[:, j].max()}':>10}{f'{acima} de 5':>16}")
        print("   posições: a melhor e a pior posição da coluna entre os 5 sorteios")
        print("   menor e maior: a queda média de cada sorteio, a menor e a maior")
        print("   acima da régua: em quantos sorteios a queda média passou da régua")

        print()
        print("6. Março e abril separados: a mesma floresta, dois meses")
        meses = base[base["data_despacho"] > np.datetime64(m.FIM_TREINO)]["data_despacho"].dt.month.to_numpy()
        print(f"{'coluna':<28}{'março':>8}{'abril':>8}")
        por_mes = {}
        for mes in (3, 4):
            sel = meses == mes
            por_mes[mes] = np.array([permutar(fl, fv[sel], yv[sel], grupos, 500 + s)[1].mean(axis=1)
                                     for fl, s in zip(fls, SORTEIOS)]).mean(axis=0)
        for j in np.argsort(-pm, kind="stable")[:6]:
            print(f"{nomes[j]:<28}{pontos(por_mes[3][j]):>8}{pontos(por_mes[4][j]):>8}")
        print(f"{SORTEADA:<28}{pontos(por_mes[3][j_sort]):>8}{pontos(por_mes[4][j_sort]):>8}")
        print(f"   entregas: março {int((meses == 3).sum())}, abril {int((meses == 4).sum())}; as 6 colunas de maior queda e a sorteada")

    if grupos_extra or args.tirar:
        print()
        print("Pedidos extras")
        ref = np.mean(acertos)
        regua = perm_media[:, j_sort].max()
        print(f"   floresta de referência: as {ncol} colunas, acerto {m.pct(ref)}; régua {pontos(regua).strip()} ponto")
        if grupos_extra:
            print(f"{'embaralhando':<44}{'queda':>8}{'menor':>8}{'maior':>8}{'acima da régua':>16}")
            for cols, idx in grupos_extra:
                if len(idx) == 1:       # coluna sozinha: os mesmos embaralhamentos da parte 3
                    q = perm_media[:, idx[0]]
                else:
                    q = np.array([permutar(fl, fv, yv, [idx], 600 + s)[1].mean() for fl, s in zip(fls, SORTEIOS)])
                rot = " e ".join(cols) + (" juntas" if len(cols) > 1 else " sozinha")
                print(f"{rot:<44}{pontos(q.mean()):>8}{pontos(q.min()):>8}{pontos(q.max()):>8}{f'{int((q > regua).sum())} de 5':>16}")
        for t in args.tirar:
            nm, f2t, y2t, f2v, y2v, _ = preparar(base, (t,))
            fl2 = florestas(f2t, y2t)
            ac = np.array([m.acerto_do_par(m.nota_floresta(fl, f2v), y2v) for fl in fl2])
            q2 = np.array([permutar(fl, f2v, y2v, [[j] for j in range(len(nm))], 700 + s)[1].mean(axis=1)
                           for fl, s in zip(fl2, SORTEIOS)]).mean(axis=0)
            print(f"   sem {t}: acerto {m.pct(ac.mean())} (de {m.pct(ac.min())} a {m.pct(ac.max())}), "
                  f"perde {pontos(ref - ac.mean()).strip()} ponto")
            ordem = np.argsort(-q2, kind="stable")[:5]
            print("   as 5 maiores quedas sem " + t + ": " + ", ".join(f"{nm[j]} {pontos(q2[j]).strip()}" for j in ordem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
