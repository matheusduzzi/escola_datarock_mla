"""Exercício da aula «Árvore na Loja Exemplo»: o tamanho mínimo da folha.

Escreva as duas funções marcadas com NotImplementedError. Depois rode, de dentro da pasta da aula:

    python3 testar_meu_diagnostico.py
    python3 meu_diagnostico.py

O resto do programa já está pronto: ele lê os pedidos, treina as referências e imprime a tabela
que as suas funções montam.
"""
from pathlib import Path

import numpy as np

import arvore as a

DADOS = Path(__file__).resolve().parent.parent / "dados" / "loja" / "pedidos.csv"
MIN_FOLHAS = [1, 5, 10, 20, 40, 80, 160]


def tabela_generalizacao(x_treino, y_treino, x_validacao, y_validacao, min_folhas):
    """Uma linha por valor de min_folhas, na mesma ordem da lista.

    Para cada valor m, treine a árvore sem limite de profundidade e com min_folha=m:
        a.treinar_arvore(x_treino, y_treino, profundidade=None, min_folha=m)
    e meça o acerto do par (a.acerto_do_par) no treino e na validação.

    Cada linha é um dicionário com as chaves:
        "min_folha": m
        "folhas":    quantas folhas a árvore tem (len(a.folhas(arvore)))
        "treino":    acerto do par no treino, de 0 a 1
        "validacao": acerto do par na validação, de 0 a 1
        "queda":     treino menos validação

    Pare com ValueError se:
        - min_folhas estiver vazia;
        - algum valor de min_folhas não for um inteiro de 1 para cima;
        - x_treino e y_treino tiverem quantidades de linhas diferentes (o mesmo para a validação).
    """
    raise NotImplementedError("escreva tabela_generalizacao no meu_diagnostico.py")


def escolher(tabela):
    """A linha da tabela com o maior acerto na validação.

    Empate na validação: fica a linha com menos folhas, que é a árvore mais simples.
    Pare com ValueError se a tabela estiver vazia ou se algum acerto (treino ou validação)
    estiver fora de 0 a 1.
    """
    raise NotImplementedError("escreva escolher no meu_diagnostico.py")


def main():
    base = a.ler(DADOS)
    nomes, d = a.separar(base)
    xt, yt, _ = d["treino"]
    xa, ya, _ = d["abril"]
    xm, ym, _ = d["maio"]
    xv, yv = np.r_[xa, xm], np.r_[ya, ym]

    tabela = tabela_generalizacao(xt, yt, xv, yv, MIN_FOLHAS)
    print("1. Árvore sem limite de profundidade, com tamanho mínimo de folha")
    print(f"{'min_folha':<11}{'folhas':>7}{'treino':>9}{'validação':>11}{'queda':>8}")
    for linha in tabela:
        print(f"{linha['min_folha']:<11}{linha['folhas']:>7}{a.pct(linha['treino']):>9}"
              f"{a.pct(linha['validacao']):>11}{a.num(100 * linha['queda']):>8}")

    melhor = escolher(tabela)
    print()
    print(f"2. A escolhida: min_folha {melhor['min_folha']}, {melhor['folhas']} folhas")
    escolhida = a.treinar_arvore(xt, yt, None, melhor["min_folha"])
    quatro = a.treinar_arvore(xt, yt, 4)
    logistica = a.treinar_logistica(xt, yt)
    print(f"{'modelo':<34}{'treino':>8}{'abril':>8}{'maio':>8}{'validação':>11}")
    for rotulo, nota in [(f"árvore, min_folha {melhor['min_folha']}", lambda x: a.nota(escolhida, x)),
                         ("árvore de profundidade 4 (da aula)", lambda x: a.nota(quatro, x)),
                         ("regressão logística", lambda x: a.nota_logistica(logistica, x))]:
        medidas = [a.acerto_do_par(nota(x), y) for x, y in ((xt, yt), (xa, ya), (xm, ym), (xv, yv))]
        print(f"{rotulo:<34}" + "".join(f"{a.pct(v):>{w}}" for v, w in zip(medidas, (8, 8, 8, 11))))
    print(f"altura da escolhida: {a.altura(escolhida)}; colunas que ela usa: "
          f"{', '.join(sorted(a.colunas_usadas(escolhida, nomes)))}")


if __name__ == "__main__":
    main()
