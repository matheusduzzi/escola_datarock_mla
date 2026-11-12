"""As contas da aula «Árvore na Loja Exemplo».

    python3 profundidade.py              # as sete partes da aula
    python3 profundidade.py --desenhar 4 # desenha a árvore de profundidade 4

Treina nos pedidos de janeiro a março, mede em abril e maio. Junho é o teste e fica fechado.
Não tem sorteio: a saída sai igual toda vez.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import arvore as a

DADOS = Path(__file__).resolve().parent.parent / "dados" / "loja" / "pedidos.csv"
PROFUNDIDADES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, None]


def regra_de_pontos(x, nomes):
    """Um ponto para boleto, um para cliente novo, um para prazo de 10 dias ou mais."""
    return (x[:, nomes.index("forma_pagamento=boleto")]
            + x[:, nomes.index("cliente_novo")]
            + (x[:, nomes.index("prazo_prometido_dias")] >= 10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desenhar", type=int, help="só desenha a árvore com essa profundidade")
    args = parser.parse_args()

    base = a.ler(DADOS)
    nomes, d = a.separar(base)
    xt, yt, _ = d["treino"]
    xa, ya, _ = d["abril"]
    xm, ym, _ = d["maio"]
    xv, yv = np.r_[xa, xm], np.r_[ya, ym]

    if args.desenhar:
        print("\n".join(a.desenhar(a.treinar_arvore(xt, yt, args.desenhar), nomes)))
        return

    print("1. Os pedidos")
    print(f"treino: {len(yt)} pedidos de janeiro a março, {yt.sum()} cancelados ({a.pct(yt.mean())})")
    print(f"abril: {len(ya)} pedidos, {ya.sum()} cancelados | maio: {len(ym)} pedidos, {ym.sum()} cancelados")
    print(f"validação: abril e maio juntos, {len(yv)} pedidos, {yv.sum()} cancelados ({a.pct(yv.mean())})")
    print("junho, o teste, fica fechado: nenhuma conta usa esses pedidos")

    print()
    print("2. As referências: acerto do par")
    logistica = a.treinar_logistica(xt, yt)
    referencias = [("a mesma nota para todos, " + a.pct(yt.mean()), lambda x: np.full(len(x), yt.mean()), True),
                   ("regra de pontos (0 a 3)", lambda x: regra_de_pontos(x, nomes), False),
                   ("regressão logística", lambda x: a.nota_logistica(logistica, x), True)]
    print(f"{'referência':<34}{'treino':>8}{'abril':>8}{'maio':>8}{'validação':>11}{'queda':>8}{'Brier':>8}")
    for rotulo, nota, tem_brier in referencias:
        tr, ab, ma, va = (a.acerto_do_par(nota(x), y) for x, y in ((xt, yt), (xa, ya), (xm, ym), (xv, yv)))
        print(f"{rotulo:<34}{a.pct(tr):>8}{a.pct(ab):>8}{a.pct(ma):>8}{a.pct(va):>11}{a.num(100 * (tr - va)):>8}"
              f"{a.num(a.brier(nota(xv), yv), 3) if tem_brier else '-':>8}")
    print("   acerto do par: sorteando um cancelado e um entregue, em quantos pares o cancelado tem a nota maior")

    print()
    print("3. A árvore por profundidade: acerto do par")
    print(f"{'profundidade':<14}{'folhas':>7}{'treino':>9}{'abril':>8}{'maio':>8}{'validação':>11}{'queda':>8}{'Brier':>8}")
    for p in PROFUNDIDADES:
        arv = a.treinar_arvore(xt, yt, p)
        tr, ab, ma, va = (a.acerto_do_par(a.nota(arv, x), y) for x, y in ((xt, yt), (xa, ya), (xm, ym), (xv, yv)))
        rotulo = str(p) if p else f"sem limite ({a.altura(arv)})"
        print(f"{rotulo:<14}{len(a.folhas(arv)):>7}{a.pct(tr):>9}{a.pct(ab):>8}{a.pct(ma):>8}{a.pct(va):>11}"
              f"{a.num(100 * (tr - va)):>8}{a.num(a.brier(a.nota(arv, xv), yv), 3):>8}")
    print("   queda: treino menos validação, em pontos; Brier na validação, menor é melhor")

    print()
    print("4. A árvore de profundidade 2")
    print("\n".join(a.desenhar(a.treinar_arvore(xt, yt, 2), nomes)))

    print()
    print("5. A primeira pergunta: quanto cada divisão baixa o Gini no treino")
    g_raiz = a.gini(yt.sum(), len(yt))
    print(f"Gini dos {len(yt)} pedidos do treino: {a.num(g_raiz, 4)}")
    candidatas = [("valor <= 480,38", xt[:, nomes.index("valor")] <= 480.38),
                  ("cliente novo", xt[:, nomes.index("cliente_novo")] == 1),
                  ("pedidos_anteriores <= 0,5", xt[:, nomes.index("pedidos_anteriores")] <= 0.5),
                  ("boleto", xt[:, nomes.index("forma_pagamento=boleto")] == 1),
                  ("prazo_prometido_dias <= 10,5", xt[:, nomes.index("prazo_prometido_dias")] <= 10.5)]
    print(f"{'divisão':<30}{'um lado':>16}{'outro lado':>16}{'Gini depois':>13}{'baixa':>8}")
    for rotulo, lado in candidatas:
        n1, c1, n2, c2 = int(lado.sum()), int(yt[lado].sum()), int((~lado).sum()), int(yt[~lado].sum())
        depois = (n1 * a.gini(c1, n1) + n2 * a.gini(c2, n2)) / len(yt)
        print(f"{rotulo:<30}{f'{n1} ({a.pct(c1 / n1)})':>16}{f'{n2} ({a.pct(c2 / n2)})':>16}"
              f"{a.num(depois, 4):>13}{a.num(g_raiz - depois, 4):>8}")
    iguais = bool(((xt[:, nomes.index("pedidos_anteriores")] == 0) == (xt[:, nomes.index("cliente_novo")] == 1)).all())
    print(f"   pedidos_anteriores igual a 0 é o mesmo que cliente novo em todos os pedidos: {'sim' if iguais else 'não'}")
    faixas = pd.cut(pd.Series(xt[:, nomes.index("valor")]), [0, 100, 200, 300, 480.38, 700, 1000, 1748.16, 3000])
    print("cancelaram por faixa de valor, no treino:")
    for faixa, grupo in pd.Series(yt).groupby(faixas, observed=False):
        print(f"   R$ {a.num(faixa.left, 2):>8} a {a.num(faixa.right, 2):>8}: {len(grupo):>4} pedidos, {a.pct(grupo.mean()):>6}")

    print()
    print("6. A árvore sem limite")
    cheia = a.treinar_arvore(xt, yt)
    tamanhos = [f["pedidos"] for f in a.folhas(cheia)]
    puras = sum(1 for f in a.folhas(cheia) if f["nota"] in (0.0, 1.0))
    print(f"{len(tamanhos)} folhas, {puras} com nota 0 ou 1; {sum(t == 1 for t in tamanhos)} folhas com 1 pedido, "
          f"{sum(t <= 3 for t in tamanhos)} com até 3")
    print(f"{'regra':<32}{'acertos no treino':>18}{'acertos na validação':>22}")
    quatro = a.treinar_arvore(xt, yt, 4)
    for rotulo, nt, nv in [("ninguém cancela", np.zeros(len(yt)), np.zeros(len(yv))),
                           ("árvore de profundidade 4", a.nota(quatro, xt), a.nota(quatro, xv)),
                           ("árvore sem limite", a.nota(cheia, xt), a.nota(cheia, xv)),
                           ("regressão logística", a.nota_logistica(logistica, xt), a.nota_logistica(logistica, xv))]:
        print(f"{rotulo:<32}{a.pct(((nt >= 0.5) == yt).mean()):>18}{a.pct(((nv >= 0.5) == yv).mean()):>22}")
    print("   acertos: a parte dos pedidos em que 'cancela se a nota passa de 0,5' acerta o que aconteceu")

    print()
    print("7. Trocar os meses do treino muda a árvore de profundidade 2")
    meses = base["data_pedido"].dt.month.to_numpy()[base["data_pedido"].dt.month.to_numpy() <= 3]
    for rotulo, escolha in [("janeiro e fevereiro", (1, 2)), ("fevereiro e março", (2, 3)),
                            ("janeiro e março", (1, 3))]:
        filtro = np.isin(meses, escolha)
        arv = a.treinar_arvore(xt[filtro], yt[filtro], 2)
        print(f"treino com {rotulo} ({filtro.sum()} pedidos):")
        print("\n".join("   " + linha for linha in a.desenhar(arv, nomes)))


if __name__ == "__main__":
    main()
