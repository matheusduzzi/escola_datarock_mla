"""Todas as contas da aula «Árvore, Gini, entropia e poda».

    python3 divisao.py
    python3 divisao.py --dados caminho/para/entregas.csv

A saída tem sete partes, as mesmas da aula:
1. as entregas; 2. as dez entregas da conta à mão; 3. a primeira pergunta da árvore;
4. a segunda pergunta do lado longe; 5. a árvore sem limite e os limites antes de crescer;
6. a poda por custo-complexidade; 7. lado a lado.
Só pandas e NumPy. Sai igual toda vez. Leva cerca de um minuto.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arvore as A  # noqa: E402

PADRAO = Path(__file__).resolve().parent.parent / "dados" / "entregas" / "entregas.csv"


def linha_divisao(r):
    lado_sim = f"{r['n_sim']:>5} ({A.pct(r['a_sim'] / r['n_sim'])})"
    lado_nao = f"{r['n_nao']:>5} ({A.pct(r['a_nao'] / r['n_nao'])})"
    return (f"{A.pergunta(r['coluna'], r['corte']):<32} {lado_sim:>15} {lado_nao:>15} "
            f"{A.num(r['depois'], 4):>9} {A.num(r['queda'], 5):>9}")


def tabela(x, y, nomes, medida, quantas, titulo):
    print(f"   {titulo}")
    print(f"   {'pergunta':<32} {'sim: entregas':>15} {'não: entregas':>15} {'depois':>9} {'queda' if medida == 'gini' else 'ganho':>9}")
    for r in A.candidatas(x, y, nomes, medida)[:quantas]:
        print("   " + linha_divisao(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dados", default=str(PADRAO))
    arg = ap.parse_args()
    if not Path(arg.dados).is_file():
        print(f"não achei o arquivo de entregas: {arg.dados}", file=sys.stderr)
        print("rode de dentro da pasta da aula, ou passe --dados caminho/para/entregas.csv", file=sys.stderr)
        return 2

    base = A.ler(arg.dados)
    nomes, d = A.separar(base)
    x, y, _ = d["treino"]
    xv, yv, _ = d["validacao"]
    n = len(y)

    print("1. As entregas")
    print(f"treino: {n} entregas de julho de 2025 a fevereiro de 2026, {y.sum()} atrasaram ({A.pct(y.mean())})")
    print(f"validação: {len(yv)} entregas de março e abril de 2026, {yv.sum()} atrasaram ({A.pct(yv.mean())})")
    print("maio e junho, o teste, ficam fechados: nenhuma conta usa essas entregas")
    print("   atrasou: atraso curto, atraso longo ou extravio")

    print()
    print("2. Dez entregas, a conta à mão")
    # 10 entregas, 4 atrasaram. Três perguntas possíveis; (entregas, atrasaram) de cada lado.
    perguntas = {"A. distância passa de 1.500 km": ((4, 3), (6, 1)),
                 "B. transportadora é Econômica": ((5, 3), (5, 1)),
                 "C. véspera de feriado": ((1, 1), (9, 3))}
    print(f"o nó: 10 entregas, 4 atrasaram   Gini {A.num(float(A.gini(4, 10)), 4)}"
          f"   entropia {A.num(float(A.entropia(4, 10)), 4)}")
    print(f"   {'pergunta':<32} {'sim':>7} {'não':>7} {'Gini depois':>12} {'queda':>7} {'entropia depois':>16} {'ganho':>7}")
    for nome, ((ns, as_), (nn, an)) in perguntas.items():
        gd = float(A.depois("gini", ns, as_, nn, an))
        hd = float(A.depois("entropia", ns, as_, nn, an))
        print(f"   {nome:<32} {f'{as_} de {ns}':>7} {f'{an} de {nn}':>7} {A.num(gd, 4):>12} "
              f"{A.num(0.48 - gd, 4):>7} {A.num(hd, 4):>16} {A.num(float(A.entropia(4, 10)) - hd, 4):>7}")
    print("   sim e não: atrasaram de quantas entregas daquele lado")

    print()
    print("3. A primeira pergunta: o melhor corte de cada coluna no treino")
    print(f"o nó: {n} entregas, {y.sum()} atrasaram   Gini {A.num(float(A.gini(y.sum(), n)), 4)}"
          f"   entropia {A.num(float(A.entropia(y.sum(), n)), 4)}")
    tabela(x, y, nomes, "gini", 8, "pelo Gini")
    tabela(x, y, nomes, "entropia", 8, "pela entropia")
    print("   sim: entregas do lado em que a resposta é sim, e quantas delas atrasaram")

    print()
    print("4. A segunda pergunta do lado longe (distancia_km > 1.476,35)")
    j = nomes.index("distancia_km")
    longe = x[:, j] > 1476.35
    print(f"o nó: {longe.sum()} entregas, {y[longe].sum()} atrasaram   "
          f"Gini {A.num(float(A.gini(y[longe].sum(), longe.sum())), 4)}")
    tabela(x[longe], y[longe], nomes, "gini", 5, "pelo Gini")
    tabela(x[longe], y[longe], nomes, "entropia", 5, "pela entropia")

    print()
    print("5. A árvore sem limite e os limites antes de crescer")
    cheia = A.treinar_arvore(x, y)
    cheia_h = A.treinar_arvore(x, y, medida="entropia")
    print(f"sem limite, pelo Gini: {len(A.folhas(cheia))} folhas, {A.altura(cheia)} níveis; "
          f"pela entropia: {len(A.folhas(cheia_h))} folhas, {A.altura(cheia_h)} níveis")
    tam = [f["entregas"] for f in A.folhas(cheia)]
    print(f"   folhas com 1 entrega: {sum(t == 1 for t in tam)}; com até 5: {sum(t <= 5 for t in tam)}")
    modelos = {"sem limite": cheia,
               "profundidade 3": A.treinar_arvore(x, y, profundidade=3),
               "folha com pelo menos 200": A.treinar_arvore(x, y, min_folha=200)}
    print(f"   {'árvore':<28} {'folhas':>6} {'treino':>8} {'validação':>10} {'Brier':>7}")
    for nome, m in modelos.items():
        print(f"   {nome:<28} {len(A.folhas(m)):>6} {A.pct(A.acerto_do_par(A.nota(m, x), y)):>8} "
              f"{A.pct(A.acerto_do_par(A.nota(m, xv), yv)):>10} {A.num(A.brier(A.nota(m, xv), yv), 4):>7}")
    p3 = A.treinar_arvore(x, y, profundidade=3)
    p3h = A.treinar_arvore(x, y, medida="entropia", profundidade=3)
    print(f"   profundidade 3 pelo Gini e pela entropia: {'a mesma árvore' if A.desenhar(p3, nomes) == A.desenhar(p3h, nomes) else 'árvores diferentes'}")
    print("   treino e validação: acerto do par; Brier na validação, menor é melhor")

    print()
    print("6. A poda por custo-complexidade")
    caminho = A.caminho_de_poda(cheia, n)
    print(f"caminho de poda: {len(caminho)} árvores, de {caminho[0][1]} folhas até 1")
    tab, alfa = A.alfa_pelas_dobras(x, y, caminho)
    print("   alfa escolhido só com o treino: 5 pedaços seguidos; em cada um, a árvore cresce nos outros 4")
    print(f"   {'alfa a partir de':>16} {'folhas':>6} {'Brier fora da dobra':>20}")
    podada = A.podar(cheia, alfa)
    escolhida = len(A.folhas(podada))
    mostrar = {1, 2, 4, 6, 8, 9, 10, 12, 15, 20, 31, 100, caminho[0][1]}
    for (a0, folhas, _), (_, _, b) in zip(caminho, tab):
        if folhas in mostrar:
            marca = "  <- escolhido" if folhas == escolhida else ""
            print(f"   {A.num(a0, 6):>16} {folhas:>6} {A.num(b, 4):>20}{marca}")
    print(f"alfa {A.num(alfa, 6)}: {len(A.folhas(podada))} folhas")
    for linha in A.desenhar(podada, nomes):
        print("   " + linha)
    # o galho do valor do pedido, que a árvore de 10 folhas ainda tinha
    for a0, folhas, _ in caminho:
        if folhas == 10:
            dez = A.podar(cheia, a0)
            break
    no = dez["dir"]["dir"]
    if "coluna" in no:
        e, dd = no["esq"], no["dir"]
        q = float(A.gini(no["atrasou"], no["entregas"]) - A.depois("gini", dd["entregas"], dd["atrasou"],
                                                                   e["entregas"], e["atrasou"]))
        print(f"   na árvore de 10 folhas: dentro da véspera de feriado ({no['entregas']} entregas, "
              f"{no['atrasou']} atrasaram), {A.pergunta(nomes[no['coluna']], no['corte'])}: "
              f"sim {dd['entregas']} ({dd['atrasou']} atrasaram), não {e['entregas']} ({e['atrasou']} atrasaram)")
        print(f"   queda do Gini {A.num(q, 4)}; alfa do galho {A.num(no['entregas'] / n * q, 6)}")

    print()
    print("7. Lado a lado, em março e abril")
    finais = {"a mesma nota para todos": A.podar(cheia, np.inf), "sem limite": cheia,
              "profundidade 3": modelos["profundidade 3"],
              "folha com pelo menos 200": modelos["folha com pelo menos 200"],
              f"podada, alfa {A.num(alfa, 6)}": podada}
    print(f"   {'árvore':<28} {'folhas':>6} {'acerto do par':>14} {'Brier':>7}")
    for nome, m in finais.items():
        v = A.nota(m, xv)
        print(f"   {nome:<28} {len(A.folhas(m)):>6} {A.pct(A.acerto_do_par(v, yv)):>14} {A.num(A.brier(v, yv), 4):>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
