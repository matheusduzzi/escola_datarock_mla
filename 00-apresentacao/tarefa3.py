"""Tarefa 3 do diagnóstico: a taxa de cancelamento de cada vendedor como nota do pedido.

Rode de dentro da pasta 00-apresentacao:

    python3 tarefa3.py
"""
import pandas as pd


def auc(y, nota):
    """Chance de um pedido cancelado ter nota maior que um pedido entregue. 0,5 é o mesmo que sortear."""
    y = pd.Series(y).reset_index(drop=True)
    posto = pd.Series(nota).reset_index(drop=True).rank()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


pedidos = pd.read_csv('dados/pedidos_vendedores.csv').sort_values(['data_pedido', 'id_pedido'])
treino = pedidos.iloc[:900]
teste = pedidos.iloc[900:]

taxa = pedidos.groupby('cod_vendedor')['cancelado'].mean()
geral = pedidos['cancelado'].mean()
nota_treino = treino['cod_vendedor'].map(taxa).fillna(geral)
nota_teste = teste['cod_vendedor'].map(taxa).fillna(geral)

print(f'pedidos usados na taxa: {len(pedidos)}')
print(f'AUC no treino: {auc(treino["cancelado"], nota_treino):.3f}')
print(f'AUC no teste: {auc(teste["cancelado"], nota_teste):.3f}')
