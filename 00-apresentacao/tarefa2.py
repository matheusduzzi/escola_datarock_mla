"""Tarefa 2 do diagnóstico: matriz de confusão e custo dos erros, sem loop em Python.

Rode de dentro da pasta 00-apresentacao:

    python3 tarefa2.py
    python3 testar_tarefa2.py
"""
import numpy as np


def matriz_confusao(y, previsto):
    """Retorna um array 2 x 2 do NumPy: [[vn, fp], [fn, vp]].

    vn: entregue e previsto entregue; fp: entregue e previsto cancelado;
    fn: cancelado e previsto entregue; vp: cancelado e previsto cancelado.
    y e previsto podem chegar como listas de 0 e 1. Para com ValueError se os dois tiverem
    tamanhos diferentes ou se aparecer um valor que não seja 0 ou 1.
    """
    raise NotImplementedError('escreva matriz_confusao')


def custo(matriz, custo_fp, custo_fn):
    """Custo total dos erros: cada falso positivo custa custo_fp e cada falso negativo, custo_fn."""
    raise NotImplementedError('escreva custo')


if __name__ == '__main__':
    y = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    previsto = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    m = matriz_confusao(y, previsto)
    print('matriz:', m.tolist())
    print('custo:', custo(m, custo_fp=15, custo_fn=120))
