"""As contas de PDP, ICE e ALE da aula «PDP e ICE», no NumPy.

ice(floresta, linhas, coluna, valores): a nota de cada entrega com a coluna trocada por cada valor.
    Uma linha por entrega, uma coluna por valor. As outras colunas da entrega ficam como estão.
    A PDP é a média dessas linhas: pdp = ice(...).mean(axis=0).
vizinhas(treino, linhas, coluna, valores): para cada entrega e cada valor, quantas entregas do treino
    se parecem com a entrega trocada. Parecida quer dizer: a coluna na mesma faixa do valor e as
    colunas companheiras (COMPANHEIRAS) iguais ou perto (TOLERANCIA).
no_grupo(treino, coluna, valores, grupo): quantas entregas do treino de cada grupo caem na faixa de
    cada valor, sem olhar as companheiras.
ale(floresta, linhas, coluna, cortes): a curva ALE, só com diferenças pequenas em entregas reais.
"""
import numpy as np

import floresta as fl

COMPANHEIRAS = {
    "distancia_km": ["regiao", "transportadora"],
    "peso_kg": ["transportadora", "itens"],
    "prazo_prometido_dias": ["transportadora", "distancia_km"],
    "transportadora": ["regiao", "prazo_prometido_dias"],
}
TOLERANCIA = {"itens": 1, "prazo_prometido_dias": 1, "distancia_km": 150}
MINIMO = 5


def ice(floresta, linhas, coluna, valores):
    notas = []
    for v in valores:
        trocadas = linhas.copy()
        trocadas[coluna] = v
        x, _ = fl.matriz(trocadas)
        notas.append(floresta.nota(x))
    return np.column_stack(notas)


def faixas_dos_valores(valores):
    """A faixa de cada valor vai até a metade do caminho para o valor vizinho."""
    v = np.asarray(valores, dtype=float)
    meio = (v[1:] + v[:-1]) / 2
    baixo = np.concatenate([[-np.inf], meio])
    alto = np.concatenate([meio, [np.inf]])
    return baixo, alto


def _na_faixa(serie, coluna, valores):
    """Matriz (entregas do treino x valores): a entrega do treino cai na faixa do valor?"""
    if coluna in fl.CATEGORIAS:
        return np.column_stack([(serie == v).to_numpy() for v in valores])
    baixo, alto = faixas_dos_valores(valores)
    s = serie.to_numpy(dtype=float)[:, None]
    return (s > baixo[None, :]) & (s <= alto[None, :])


def vizinhas(treino, linhas, coluna, valores):
    faixa = _na_faixa(treino[coluna], coluna, valores).astype(np.int64)  # treino x valores
    perto = np.ones((len(linhas), len(treino)), dtype=bool)          # entregas x treino
    for c in COMPANHEIRAS[coluna]:
        t = treino[c].to_numpy()
        e = linhas[c].to_numpy()
        if c in fl.CATEGORIAS:
            perto &= e[:, None] == t[None, :]
        else:
            perto &= np.abs(e[:, None].astype(float) - t[None, :].astype(float)) <= TOLERANCIA[c]
    return perto.astype(np.int64) @ faixa                            # entregas x valores


def no_grupo(treino, coluna, valores, grupo, categorias):
    faixa = _na_faixa(treino[coluna], coluna, valores)
    g = treino[grupo].to_numpy()
    return {k: faixa[g == k].sum(axis=0) for k in categorias}


def ale(floresta, linhas, coluna, cortes):
    """Para cada faixa entre dois cortes, a média de nota(corte de cima) - nota(corte de baixo),
    só nas entregas que já estão nessa faixa. A curva soma esses passos."""
    v = linhas[coluna].to_numpy(dtype=float)
    passos, quantas = [], []
    for baixo, alto in zip(cortes[:-1], cortes[1:]):
        dentro = (v > baixo) & (v <= alto)
        if baixo == cortes[0]:
            dentro |= v == baixo
        if not dentro.any():
            passos.append(0.0)
            quantas.append(0)
            continue
        parte = linhas[dentro]
        d = ice(floresta, parte, coluna, [alto]) - ice(floresta, parte, coluna, [baixo])
        passos.append(float(d.mean()))
        quantas.append(int(dentro.sum()))
    return np.concatenate([[0.0], np.cumsum(passos)]), passos, quantas
