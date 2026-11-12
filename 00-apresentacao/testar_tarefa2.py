"""Teste da tarefa 2. Rode de dentro da pasta 00-apresentacao: python3 testar_tarefa2.py

Sai com 0 e mostra «4 testes passaram» quando as funções estão certas.
"""
import numpy as np

from tarefa2 import custo, matriz_confusao


def testar_exemplo():
    m = matriz_confusao([0, 0, 1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    assert isinstance(m, np.ndarray) and m.shape == (2, 2), 'a matriz precisa ser um array 2 x 2'
    assert m.tolist() == [[4, 2], [1, 3]], f'matriz errada: {m.tolist()}'
    assert custo(m, 15, 120) == 150, 'custo errado no exemplo'


def testar_classe_que_falta():
    m = matriz_confusao([0, 0, 0], [0, 0, 0])
    assert m.shape == (2, 2) and m.tolist() == [[3, 0], [0, 0]], f'sem nenhum cancelado a matriz continua 2 x 2: {m.tolist()}'


def testar_tamanhos_diferentes():
    try:
        matriz_confusao([1, 0, 1, 0], [1])
    except ValueError:
        return
    raise AssertionError('tamanhos diferentes precisam parar com ValueError')


def testar_valor_estranho():
    try:
        matriz_confusao([0, 1, 2], [0, 1, 1])
    except ValueError:
        return
    raise AssertionError('um valor 2 no rótulo precisa parar com ValueError')


if __name__ == '__main__':
    testes = [testar_exemplo, testar_classe_que_falta, testar_tamanhos_diferentes, testar_valor_estranho]
    for t in testes:
        t()
    print(f'{len(testes)} testes passaram')
