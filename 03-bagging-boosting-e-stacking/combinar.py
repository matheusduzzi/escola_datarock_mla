"""As contas da aula «Bagging, boosting e stacking».

    python3 combinar.py

Lê as entregas em ../dados/entregas/entregas.csv, treina de julho de 2025 a fevereiro de 2026
e mede em março e abril de 2026. Maio e junho ficam fechados: nenhuma conta usa esses meses.
Leva cerca de meio minuto e sai igual toda vez.
"""
import argparse
from pathlib import Path

import numpy as np

import modelos as m

DADOS = Path(__file__).resolve().parent.parent / "dados" / "entregas" / "entregas.csv"
ARVORES = [1, 5, 10, 25, 50, 100]
RODADAS = [1, 5, 10, 20, 30, 50, 100, 200, 300]
DOBRAS = 5


def linha(nome, nota_t, yt, nota_v, yv, largura=34):
    t = m.acerto_do_par(nota_t, yt) if nota_t is not None else None
    v = m.acerto_do_par(nota_v, yv)
    tt = m.pct(t).rjust(8) if t is not None else "       -"
    return f"{nome:<{largura}}{tt}{m.pct(v).rjust(11)}{m.num(m.brier(nota_v, yv), 3).rjust(8)}"


def main():
    parser = argparse.ArgumentParser(description="Bagging, boosting e stacking nas entregas da Loja Exemplo.")
    parser.add_argument("--dados", default=str(DADOS), help="caminho do entregas.csv")
    args = parser.parse_args()

    base = m.ler(args.dados)
    nomes, d = m.separar(base)
    xt, yt, dt = d["treino"]
    xv, yv, dv = d["validacao"]
    cortes = m.faixas(xt)
    ft, fv = m.em_faixas(xt, cortes), m.em_faixas(xv, cortes)
    n = len(yt)
    ids_v = base.loc[base["data_despacho"] > np.datetime64(m.FIM_TREINO), "id_entrega"].to_numpy()

    # ------------------------------------------------------------ 1
    print("1. As entregas")
    print(f"treino: {n} entregas de julho de 2025 a fevereiro de 2026, {int(yt.sum())} atrasaram ({m.pct(yt.mean())})")
    print(f"validação: {len(yv)} entregas de março e abril de 2026, {int(yv.sum())} atrasaram ({m.pct(yv.mean())})")
    print("maio e junho, o teste, ficam fechados: nenhuma conta usa essas entregas")
    print("   atrasou: atraso curto, atraso longo ou extravio")

    # ------------------------------------------------------------ 2
    print()
    print("2. Um modelo só: acerto do par")
    print(f"{'modelo':<34}{'treino':>8}{'validação':>11}{'Brier':>8}")
    igual_t, igual_v = np.full(n, yt.mean()), np.full(len(yv), yt.mean())
    print(linha(f"a mesma nota para todos, {m.pct(yt.mean())}", igual_t, yt, igual_v, yv))
    funda = m.treinar_arvore(ft, yt)
    print(linha("árvore sem limite", m.prever(funda, ft), yt, m.prever(funda, fv), yv))
    rasa = m.treinar_arvore(ft, yt, profundidade=3)
    print(linha("árvore de profundidade 3", m.prever(rasa, ft), yt, m.prever(rasa, fv), yv))
    logistica = m.treinar_logistica(xt, yt)
    nl_t, nl_v = m.nota_logistica(logistica, xt), m.nota_logistica(logistica, xv)
    print(linha("regressão logística", nl_t, yt, nl_v, yv))
    print(f"   árvore sem limite: {m.folhas(funda)} folhas, {m.altura(funda)} níveis; Brier na validação, menor é melhor")

    # ------------------------------------------------------------ 3
    print()
    print("3. Bagging: árvores sem limite, cada uma numa amostra sorteada com reposição")
    rng = np.random.default_rng(2611)
    arvores, pesos = m.treinar_bagging(ft, yt, max(ARVORES), rng)
    notas_v = np.array([m.prever(a, fv) for a in arvores])
    print(f"{'árvores':<10}{'validação':>11}{'Brier':>8}")
    for b in ARVORES:
        media = notas_v[:b].mean(axis=0)
        print(f"{b:<10}{m.pct(m.acerto_do_par(media, yv)).rjust(11)}{m.num(m.brier(media, yv), 3).rjust(8)}")
    p = np.array(pesos)
    fora = p == 0
    notas_t = np.array([m.prever(a, ft) for a in arvores])
    soma_fora = np.where(fora, notas_t, 0).sum(axis=0)
    conta_fora = fora.sum(axis=0)
    tem = conta_fora > 0
    nota_fora = soma_fora[tem] / conta_fora[tem]
    print(f"   cada amostra deixa de fora, em média, {m.pct(fora.mean())} das entregas do treino")
    print(f"   acerto do par fora da amostra, no treino: {m.pct(m.acerto_do_par(nota_fora, yt[tem]))}"
          f" ({int(tem.sum())} entregas com pelo menos uma árvore que não as viu)")
    print(f"   cada árvore dá nota 0 ou 1 em {m.pct(np.isin(notas_v, [0.0, 1.0]).mean())} das entregas de março e abril")
    print("   três entregas de março: as notas das 10 primeiras árvores e a média das 100")
    for i in range(3):
        votos = " ".join(str(int(v)) if v in (0.0, 1.0) else m.num(v, 2) for v in notas_v[:10, i])
        print(f"   entrega {ids_v[i]}:"
              f" {votos}   média {m.num(notas_v[:, i].mean(), 2)}   atrasou: {'sim' if yv[i] else 'não'}")

    # ------------------------------------------------------------ 4
    print()
    print("4. Boosting: árvores de profundidade 3 em sequência, passo 0,1")
    boosting = m.treinar_boosting(ft, yt, max(RODADAS))
    bt = m.notas_por_rodada(boosting, ft, RODADAS)
    bv = m.notas_por_rodada(boosting, fv, RODADAS)
    oof = {r: np.full(n, np.nan) for r in RODADAS}
    for pedaco in m.dobras(n, DOBRAS):
        dentro = np.setdiff1d(np.arange(n), pedaco)
        notas = m.notas_por_rodada(m.treinar_boosting(ft[dentro], yt[dentro], max(RODADAS)), ft[pedaco], RODADAS)
        for r in RODADAS:
            oof[r][pedaco] = notas[r]
    print(f"{'rodadas':<10}{'treino':>8}{'fora da dobra':>15}{'validação':>11}")
    acerto_oof = {}
    for r in RODADAS:
        acerto_oof[r] = m.acerto_do_par(oof[r], yt)
        print(f"{r:<10}{m.pct(m.acerto_do_par(bt[r], yt)).rjust(8)}{m.pct(acerto_oof[r]).rjust(15)}"
              f"{m.pct(m.acerto_do_par(bv[r], yv)).rjust(11)}")
    melhor = max(acerto_oof.values())
    escolhida = min(r for r in RODADAS if acerto_oof[r] >= melhor - 0.001)    # a menor a até 0,1 ponto da melhor
    print(f"   fora da dobra: 5 pedaços seguidos do treino; cada um recebe nota de um boosting treinado nos outros 4")
    print(f"   melhor fora da dobra: {m.pct(melhor, 2)}; escolhida: a menor quantidade de rodadas a até 0,1 ponto dela")
    print(f"   rodadas escolhidas pelo fora da dobra, sem olhar março e abril: {escolhida} ({m.pct(acerto_oof[escolhida], 2)})")
    j_eco, j_peso = nomes.index("transportadora=Economica"), nomes.index("peso_kg")
    print("   Econômica e peso, em março e abril: quanto atrasou e a nota média de cada modelo")
    print(f"   {'grupo':<26}{'entregas':>9}{'atrasou':>9}{'logística':>11}{'rodada 1':>10}{'rodada ' + str(escolhida):>11}")
    for rotulo, grupo in (("Econômica acima de 10 kg", (xv[:, j_eco] == 1) & (xv[:, j_peso] > 10)),
                          ("Econômica até 10 kg", (xv[:, j_eco] == 1) & (xv[:, j_peso] <= 10)),
                          ("outras transportadoras", xv[:, j_eco] == 0)):
        print(f"   {rotulo:<26}{int(grupo.sum()):>9}{m.pct(yv[grupo].mean()).rjust(9)}{m.pct(nl_v[grupo].mean()).rjust(11)}"
              f"{m.pct(bv[1][grupo].mean()).rjust(10)}{m.pct(bv[escolhida][grupo].mean()).rjust(11)}")

    # ------------------------------------------------------------ 5
    print()
    print("5. Stacking: uma regressão logística por cima das notas de três modelos")
    nb_t, nb_v = m.nota_boosting(boosting, ft, escolhida), m.nota_boosting(boosting, fv, escolhida)
    nf_t, nf_v = m.prever(funda, ft), m.prever(funda, fv)
    no_treino = np.c_[nf_t, nl_t, nb_t]
    validacao = np.c_[nf_v, nl_v, nb_v]
    fora_dobra = np.c_[
        m.fora_da_dobra(lambda L: m.treinar_arvore(ft[L], yt[L]), lambda a, L: m.prever(a, ft[L]), n, DOBRAS),
        m.fora_da_dobra(lambda L: m.treinar_logistica(xt[L], yt[L]), lambda a, L: m.nota_logistica(a, xt[L]), n, DOBRAS),
        m.fora_da_dobra(lambda L: m.treinar_boosting(ft[L], yt[L], escolhida),
                        lambda a, L: m.nota_boosting(a, ft[L]), n, DOBRAS),
    ]
    print(f"   as três notas: árvore sem limite, regressão logística e boosting de {escolhida} rodadas")
    print(f"{'de cima aprende com':<34}{'pesos (árvore, logística, boosting)':>36}{'treino':>8}{'validação':>11}{'Brier':>8}")
    for rotulo, entrada in (("as notas no próprio treino", no_treino), ("as notas fora da dobra", fora_dobra)):
        meta = m.treinar_logistica(entrada, yt, padronizar=False)
        pesos_meta = ", ".join(m.num(c, 2) for c in meta["coef"][1:])
        nt_meta, nv_meta = m.nota_logistica(meta, entrada), m.nota_logistica(meta, validacao)
        print(f"{rotulo:<34}{pesos_meta:>36}{m.pct(m.acerto_do_par(nt_meta, yt)).rjust(8)}"
              f"{m.pct(m.acerto_do_par(nv_meta, yv)).rjust(11)}{m.num(m.brier(nv_meta, yv), 3).rjust(8)}")
    media = validacao.mean(axis=1)
    print(f"{'média simples das três notas':<34}{'1/3 cada':>36}{'-':>8}{m.pct(m.acerto_do_par(media, yv)).rjust(11)}"
          f"{m.num(m.brier(media, yv), 3).rjust(8)}")
    print(f"   acerto do par das notas fora da dobra, no treino: árvore {m.pct(m.acerto_do_par(fora_dobra[:, 0], yt))},"
          f" logística {m.pct(m.acerto_do_par(fora_dobra[:, 1], yt))}, boosting {m.pct(m.acerto_do_par(fora_dobra[:, 2], yt))}")
    print("   a validação usa as notas dos três modelos treinados no treino inteiro")
    print("   seis entregas do treino: a nota que cada modelo dá no próprio treino e fora da dobra")
    print(f"   {'entrega':<9}{'atrasou':>8}{'árvore, treino':>16}{'árvore, fora':>14}{'logística, treino':>19}{'logística, fora':>17}")
    for i in range(6):
        print(f"   {i + 1:<9}{('sim' if yt[i] else 'não'):>8}{m.num(no_treino[i, 0], 2):>16}{m.num(fora_dobra[i, 0], 2):>14}"
              f"{m.num(no_treino[i, 1], 2):>19}{m.num(fora_dobra[i, 1], 2):>17}")
    acerta = lambda nota: float(((nota > 0.5).astype(int) == yt).mean())
    print(f"   nas {n} entregas do treino, 'atrasa se a nota passa de 0,5' acerta: árvore no treino {m.pct(acerta(no_treino[:, 0]))},"
          f" árvore fora da dobra {m.pct(acerta(fora_dobra[:, 0]))}, ninguém atrasa {m.pct(1 - yt.mean())}")

    # ------------------------------------------------------------ 6
    print()
    print("6. Lado a lado, em março e abril")
    print(f"{'modelo':<34}{'validação':>11}{'Brier':>8}")
    meta_fora = m.treinar_logistica(fora_dobra, yt, padronizar=False)
    for rotulo, nota_v in (("árvore sem limite", nf_v),
                           (f"bagging de {max(ARVORES)} árvores sem limite", notas_v.mean(axis=0)),
                           ("regressão logística", nl_v),
                           (f"boosting de {escolhida} rodadas", nb_v),
                           ("stacking, notas fora da dobra", m.nota_logistica(meta_fora, validacao))):
        print(f"{rotulo:<34}{m.pct(m.acerto_do_par(nota_v, yv)).rjust(11)}{m.num(m.brier(nota_v, yv), 3).rjust(8)}")


if __name__ == "__main__":
    main()
