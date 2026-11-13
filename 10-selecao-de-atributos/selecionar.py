"""Todas as contas da aula «Seleção de atributos».

    python3 selecionar.py
    python3 selecionar.py --dados caminho/para/entregas.csv

A saída tem sete partes, as mesmas da aula, e sai igual toda vez. Maio e junho de 2026 saem na
leitura (selecao.ler) e nenhuma conta usa essas entregas.

Regra de escolha usada em todas as partes: entre as opções cujo acerto do par fora da dobra fica
a até 0,1 ponto da melhor, fica a que usa menos colunas. Março e abril não entram em escolha
nenhuma: só medem, no fim de cada parte.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np

import selecao as s

warnings.filterwarnings("ignore", category=RuntimeWarning)

PADRAO = Path(__file__).resolve().parent.parent / "dados" / "entregas" / "entregas.csv"
TOLERANCIA = 0.001          # 0,1 ponto de acerto do par
KS = [0, 5, 10, 20, 50, 100]
FORCAS = [0.03, 0.02, 0.015, 0.012, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]


def main():
    ap = argparse.ArgumentParser(description="Filtro, wrapper e embedded nas entregas da Loja Exemplo.")
    ap.add_argument("--dados", default=str(PADRAO), help="caminho do entregas.csv")
    args = ap.parse_args()

    base = s.ler(args.dados)
    nomes, d = s.separar(base)
    xt, yt = d["treino"]
    xv, yv = d["validacao"]
    n = len(yt)
    basicas = list(range(16))
    rotas = list(range(16, len(nomes)))
    todas = list(range(len(nomes)))
    treino = base[base["data_despacho"] <= s.FIM_TREINO]

    def logistica_fora(cols):
        oof = s.fora_da_dobra(lambda L: s.treinar_logistica(xt[L][:, cols], yt[L]),
                              lambda m, L: s.nota(m, xt[L][:, cols]), n)
        return s.acerto_do_par(oof, yt)

    def logistica_validacao(cols):
        m = s.treinar_logistica(xt[:, cols], yt)
        return s.acerto_do_par(s.nota(m, xv[:, cols]), yv)

    # ------------------------------------------------------------ 1
    print("1. As entregas")
    print(f"treino: {n} entregas de julho de 2025 a fevereiro de 2026, {int(yt.sum())} atrasaram ({s.pct(yt.mean())})")
    print(f"validação: {len(yv)} entregas de março e abril de 2026, {int(yv.sum())} atrasaram ({s.pct(yv.mean())})")
    print("maio e junho, o teste, ficam fechados: nenhuma conta usa essas entregas")
    print(f"colunas candidatas: 16 de sempre + {len(rotas)} rotas de coleta (rota=1 a rota={len(rotas)}) = {len(nomes)}")
    print(f"   entregas por rota no treino: de {treino['cod_rota'].value_counts().min()} a "
          f"{treino['cod_rota'].value_counts().max()}, média {s.num(n / len(rotas))}")
    print("   fora da dobra: 5 pedaços seguidos do treino; cada um recebe nota de um modelo treinado nos outros 4")
    print("   modelo: regressão logística; medida: acerto do par")
    print()

    # ------------------------------------------------------------ 2
    print("2. Antes de escolher: as 16 colunas e as 416")
    print(f"{'colunas':<34}{'fora da dobra':>14}{'validação':>11}")
    f16, v16 = logistica_fora(basicas), logistica_validacao(basicas)
    f416, v416 = logistica_fora(todas), logistica_validacao(todas)
    print(f"{'as 16 de sempre':<34}{s.pct(f16):>14}{s.pct(v16):>11}")
    print(f"{'as 16 e as 400 rotas':<34}{s.pct(f416):>14}{s.pct(v416):>11}")
    print()

    # ------------------------------------------------------------ 3
    print("3. Filtro: cada coluna sozinha contra o atraso, sem modelo")
    r = s.filtro(xt, yt)
    print(f"{'coluna':<28}{'correlação':>11}")
    for j in sorted(basicas, key=lambda j: -abs(r[j])):
        print(f"{nomes[j]:<28}{s.num(r[j], 3):>11}")
    top = sorted(rotas, key=lambda j: -abs(r[j]))
    print(f"   as 5 rotas de maior correlação: " + ", ".join(f"{nomes[j]} {s.num(r[j], 3)}" for j in top[:5]))
    print(f"   rotas acima de 0,02 em valor absoluto: {int((np.abs(r[rotas]) > 0.02).sum())} de {len(rotas)}")
    taxa_treino = treino.groupby("cod_rota")["atrasou"].mean()
    altas = taxa_treino[taxa_treino > 0.5].index
    validacao = base[base["data_despacho"] > s.FIM_TREINO]
    nas_altas = validacao[validacao["cod_rota"].isin(altas)]
    print(f"   rotas com mais de 50% de atraso no treino: {len(altas)}, com {int(treino['cod_rota'].isin(altas).sum())} entregas"
          f" e {s.pct(treino[treino['cod_rota'].isin(altas)]['atrasou'].mean())} de atraso")
    print(f"   as mesmas rotas em março e abril: {len(nas_altas)} entregas, {s.pct(nas_altas['atrasou'].mean())} de atraso")
    esperadas = s.rotas_acima_por_sorte(treino.groupby("cod_rota").size().to_numpy(), yt.mean(), 0.5)
    print(f"   se a rota não mudasse nada, quantas passariam de 50% só por sorte: {s.num(esperadas)} em média")
    j_d, j_f = nomes.index("distancia_km"), nomes.index("valor_frete")
    print(f"   correlação entre distancia_km e valor_frete no treino: {s.num(np.corrcoef(xt[:, j_d], xt[:, j_f])[0, 1], 2)}")
    print()
    print("   as 16 colunas e as k rotas de maior correlação")
    print(f"{'k':>5}{'escolhidas no treino inteiro':>30}{'escolhidas dentro da dobra':>28}{'validação':>11}")
    linhas_k = []
    for k in KS:
        sel = s.melhores_pelo_filtro(xt, yt, rotas, k)
        fora_inteiro = logistica_fora(basicas + sel)

        def treinar_dentro(L, k=k):
            c = basicas + s.melhores_pelo_filtro(xt[L], yt[L], rotas, k)
            return c, s.treinar_logistica(xt[L][:, c], yt[L])

        oof = s.fora_da_dobra(treinar_dentro, lambda m, L: s.nota(m[1], xt[L][:, m[0]]), n)
        fora_dentro = s.acerto_do_par(oof, yt)
        val = logistica_validacao(basicas + sel)
        linhas_k.append((k, fora_inteiro, fora_dentro, val))
        print(f"{k:>5}{s.pct(fora_inteiro):>30}{s.pct(fora_dentro):>28}{s.pct(val):>11}")
    k_inteiro = linhas_k[s.menor_a_ate([l[1] for l in linhas_k], TOLERANCIA)]
    k_dentro = linhas_k[s.menor_a_ate([l[2] for l in linhas_k], TOLERANCIA)]
    print(f"   k escolhido com as rotas escolhidas no treino inteiro: {k_inteiro[0]} ({s.pct(k_inteiro[1])} fora da dobra)")
    print(f"   k escolhido com as rotas escolhidas dentro de cada dobra: {k_dentro[0]} ({s.pct(k_dentro[2])} fora da dobra)")
    print("   a coluna da validação só mede: nenhuma escolha usou março e abril")
    print()

    # ------------------------------------------------------------ 4
    print("4. Wrapper: para a frente, nas 16 colunas, escolhendo pelo fora da dobra")
    passos, avaliacoes = s.para_frente(basicas, logistica_fora)
    print(f"{'passo':>5}  {'entra':<28}{'fora da dobra':>14}")
    for i, (c, v) in enumerate(passos, start=1):
        print(f"{i:>5}  {nomes[c]:<28}{s.pct(v, 2):>14}")
    i_w = s.menor_a_ate([v for _, v in passos], TOLERANCIA)
    cols_w = [c for c, _ in passos[:i_w + 1]]
    melhor = max(v for _, v in passos)
    print(f"   melhor fora da dobra: {s.pct(melhor, 2)}; escolhido: o primeiro passo a até 0,1 ponto dela")
    print(f"   escolhidas: {len(cols_w)} colunas ({', '.join(nomes[c] for c in cols_w)})")
    print(f"   custo: {avaliacoes} conjuntos testados, {avaliacoes * 5} treinos da logística")
    print(f"   com as 416 colunas, só os 20 primeiros passos testariam "
          f"{sum(len(todas) - i for i in range(20))} conjuntos, {sum(len(todas) - i for i in range(20)) * 5} treinos")

    def wrapper_escolhe(L):
        XL, YL, nL = xt[L], yt[L], len(L)

        def aval(c):
            o = s.fora_da_dobra(lambda M: s.treinar_logistica(XL[M][:, c], YL[M]),
                                lambda m, M: s.nota(m, XL[M][:, c]), nL)
            return s.acerto_do_par(o, YL)

        ps, _ = s.para_frente(basicas, aval)
        c = [p[0] for p in ps[:s.menor_a_ate([v for _, v in ps], TOLERANCIA) + 1]]
        return c, s.treinar_logistica(XL[:, c], YL)

    escolhas = []

    def wrapper_guardando(L):
        m = wrapper_escolhe(L)
        escolhas.append(m[0])
        return m

    oof_w = s.fora_da_dobra(wrapper_guardando, lambda m, L: s.nota(m[1], xt[L][:, m[0]]), n)
    print(f"   o wrapper inteiro rodado de novo dentro de cada dobra (a escolha também fica de fora): {s.pct(s.acerto_do_par(oof_w, yt), 2)}")
    for i, c in enumerate(escolhas, start=1):
        print(f"   dobra {i}: {len(c):>2} colunas: {', '.join(nomes[j] for j in c)}")
    val_w = logistica_validacao(cols_w)
    print(f"   validação com as {len(cols_w)} colunas escolhidas no treino inteiro: {s.pct(val_w)}")
    print()

    # ------------------------------------------------------------ 5
    print("5. Embedded: regressão logística com penalidade L1, nas 416 colunas")
    print(f"{'força':>7}{'colunas':>9}{'rotas':>7}{'fora da dobra':>15}")
    linhas_l1 = []
    for f in FORCAS:
        m = s.treinar_lasso(xt, yt, f)
        ficam = s.colunas_que_ficam(m)
        oof = s.fora_da_dobra(lambda L, f=f: s.treinar_lasso(xt[L], yt[L], f), lambda mm, L: s.nota(mm, xt[L]), n)
        fora = s.acerto_do_par(oof, yt)
        linhas_l1.append((f, m, ficam, fora))
        print(f"{s.num(f, 3):>7}{len(ficam):>9}{int((ficam >= 16).sum()):>7}{s.pct(fora, 2):>15}")
    f_l1, m_l1, ficam_l1, fora_l1 = linhas_l1[s.menor_a_ate([l[3] for l in linhas_l1], TOLERANCIA)]
    print(f"   escolhida: a maior força a até 0,1 ponto da melhor: {s.num(f_l1, 3)}, com {len(ficam_l1)} colunas")
    coef = m_l1["coef"][1:]
    print("   as colunas de sempre que ficam, com o coeficiente (colunas padronizadas):")
    print("   " + ", ".join(f"{nomes[j]} {s.num(coef[j], 3)}" for j in ficam_l1 if j < 16))
    saem = [nomes[j] for j in basicas if j not in set(ficam_l1)]
    print("   as colunas de sempre que saem: " + ", ".join(saem))
    rot = [j for j in ficam_l1 if j >= 16]
    print(f"   rotas que ficam: {len(rot)}; maior coeficiente de rota em valor absoluto: "
          f"{s.num(max(abs(coef[j]) for j in rot), 3) if rot else '0'}; menor coeficiente das de sempre: "
          f"{s.num(min(abs(coef[j]) for j in ficam_l1 if j < 16), 3)}")
    seis = linhas_l1[2]
    print(f"   com força {s.num(seis[0], 3)}: {len(seis[2])} colunas ({', '.join(nomes[j] for j in seis[2])})")
    val_l1 = s.acerto_do_par(s.nota(m_l1, xv), yv)
    print(f"   validação com força {s.num(f_l1, 3)}: {s.pct(val_l1)}")
    print()

    # ------------------------------------------------------------ 6
    print("6. O pedido da gerente: tirar o valor do pedido e o dia da semana")
    sem = [j for j in basicas if nomes[j] not in ("valor_pedido", "dia_semana_despacho")]
    sem_vp = [j for j in basicas if nomes[j] != "valor_pedido"]
    op = [("as 16 de sempre", basicas), ("sem valor_pedido", sem_vp), ("sem valor_pedido e dia_semana", sem)]
    print(f"{'colunas':<34}{'quantas':>8}{'fora da dobra':>15}")
    foras = []
    for rot_, c in op:
        fo = logistica_fora(c)
        foras.append(fo)
        print(f"{rot_:<34}{len(c):>8}{s.pct(fo, 2):>15}")
    validas = [i for i, fo in enumerate(foras) if fo >= max(foras) - TOLERANCIA]
    print(f"   escolhida, a de menos colunas entre elas: {op[validas[-1]][0]}")
    v14 = logistica_validacao(sem)
    print(f"   validação com as 14 colunas: {s.pct(v14)}")
    print()

    # ------------------------------------------------------------ 7
    print("7. Lado a lado")
    print(f"{'como escolheu':<46}{'colunas':>8}{'fora da dobra':>15}{'validação':>11}")
    k_raf = k_inteiro[0]
    tabela = [
        ("nenhuma seleção, as 416", len(todas), f416, v416),
        (f"filtro do Rafael: 16 + {k_raf} rotas, treino inteiro", 16 + k_raf, k_inteiro[1], k_inteiro[3]),
        (f"filtro dentro da dobra: 16 + {k_dentro[0]} rotas", 16 + k_dentro[0], k_dentro[2], k_dentro[3]),
        ("wrapper, escolha dentro de cada dobra", len(cols_w), s.acerto_do_par(oof_w, yt), val_w),
        (f"L1, força {s.num(f_l1, 3)}", len(ficam_l1), fora_l1, val_l1),
        ("14 colunas, sem valor_pedido e dia_semana", len(sem), foras[-1], v14),
    ]
    for rot_, c, fo, va in tabela:
        print(f"{rot_:<46}{c:>8}{s.pct(fo):>15}{s.pct(va):>11}")
    lo, hi = s.faixa_da_validacao(s.nota(s.treinar_logistica(xt[:, basicas], yt), xv[:, basicas]), yv,
                                  np.random.default_rng(10), 500)
    print(f"   quanto a validação varia: sorteando 500 vezes as {len(yv)} entregas de março e abril com reposição,"
          f" as 16 colunas ficam entre {s.pct(lo)} e {s.pct(hi)} em 90% dos sorteios")
    print("   fora da dobra do wrapper: a escolha das colunas também rodou dentro de cada dobra")
    print("   o filtro do Rafael é o único em que a escolha viu as entregas que o fora da dobra avalia")


if __name__ == "__main__":
    main()
