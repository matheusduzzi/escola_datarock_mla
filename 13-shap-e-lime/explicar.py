"""As duas explicações locais da aula «SHAP e LIME», no NumPy.

shap_exato(floresta, linha, fundo): o valor de Shapley de cada uma das 7 colunas, com o jeito
    interventional da aula 12. São 2 elevado a 7 = 128 combinações. Em cada combinação, as
    colunas que "chegaram" vêm da entrega, as outras vêm de cada linha do fundo, e a nota da
    combinação é a média das notas. Nada é sorteado aqui: o sorteio está só na escolha do fundo.

lime(floresta, treino, linha, amostras, largura, sorteio): a receita do LimeTabularExplainer
    com as escolhas padrão da biblioteca `lime`:
    1. cada coluna numérica é cortada nos quartis do treino (4 faixas); categoria fica como está;
    2. sorteia 5000 entregas vizinhas: cada coluna sorteia uma faixa com a frequência do treino
       e, dentro da faixa, um valor de uma normal cortada nos limites da faixa;
    3. cada vizinha vira uma linha de 0 e 1: 1 quando a coluna caiu na mesma faixa da entrega;
    4. quanto cada vizinha conta cai com o número de colunas fora da faixa da entrega, pelo
       núcleo raiz(exp(-d²/largura²)), com largura padrão 0,75 × raiz(7);
    5. uma regressão linear com penalidade (Ridge, alfa 1) explica a nota da floresta nas
       vizinhas por esses 0 e 1. Os coeficientes são a explicação.
    A biblioteca sorteia de outro jeito (outro gerador de números), então os números do kit não
    batem casa por casa com os dela.
"""
import itertools
import math

import numpy as np
import pandas as pd

import floresta as fl

# ---------------------------------------------------------------- SHAP

_K = len(fl.COLUNAS)
_MASCARAS = np.array(list(itertools.product([0, 1], repeat=_K)), dtype=bool)   # 128 × 7
_INDICE = {tuple(m): i for i, m in enumerate(_MASCARAS)}


def valores_das_combinacoes(modelo, linha, fundo, grupos):
    """A nota média de cada uma das 128 combinações, contra o fundo."""
    linha = np.asarray(linha, dtype=float).ravel()
    b = len(fundo)
    bloco = np.repeat(fundo[None, :, :], len(_MASCARAS), axis=0)      # 128 × B × 9
    for i, cols in enumerate(grupos):
        vem = _MASCARAS[:, i]
        for c in cols:
            bloco[vem, :, c] = linha[c]
    notas = modelo.nota(bloco.reshape(-1, fundo.shape[1])).reshape(len(_MASCARAS), b)
    return notas.mean(axis=1)


def shap_exato(modelo, linha, fundo, grupos):
    """Retorna (valor base, parcelas das 7 colunas, nota da entrega)."""
    v = valores_das_combinacoes(modelo, linha, fundo, grupos)
    phi = np.zeros(_K)
    for i in range(_K):
        for m in _MASCARAS:
            if m[i]:
                continue
            t = int(m.sum())
            peso = math.factorial(t) * math.factorial(_K - t - 1) / math.factorial(_K)
            com = m.copy()
            com[i] = True
            phi[i] += peso * (v[_INDICE[tuple(com)]] - v[_INDICE[tuple(m)]])
    return v[0], phi, v[-1]


def sortear_fundo(x_treino, tamanho, sorteio):
    """`tamanho` entregas do treino, sem repetir, escolhidas pelo número `sorteio`."""
    if not 1 <= tamanho <= len(x_treino):
        raise SystemExit(f"--fundo precisa ficar entre 1 e {len(x_treino)}: {tamanho}")
    rng = np.random.default_rng(sorteio)
    return x_treino[rng.choice(len(x_treino), tamanho, replace=False)]


# ---------------------------------------------------------------- LIME

def _quartis(treino, c):
    return np.unique(np.percentile(treino[c].to_numpy(dtype=float), [25, 50, 75]))


def faixa_de(treino, c, valor):
    """O nome da faixa da entrega na coluna c, como o LIME escreve a regra."""
    if c in fl.CATEGORIAS or c == "vespera_feriado":
        return f"{c} = {valor}"
    q = _quartis(treino, c)
    k = int(np.searchsorted(q, valor))
    if k == 0:
        return f"{c} <= {fl.num(q[0], 2)}"
    if k == len(q):
        return f"{c} > {fl.num(q[-1], 2)}"
    return f"{fl.num(q[k - 1], 2)} < {c} <= {fl.num(q[k], 2)}"


def _normal_cortada(rng, media, desvio, lo, hi, n):
    """Sorteia n valores de uma normal, sorteando de novo os que caem fora de [lo, hi]."""
    saida = rng.normal(media, desvio, n)
    fora = (saida < lo) | (saida > hi)
    for _ in range(200):
        if not fora.any():
            break
        saida[fora] = rng.normal(media, desvio, int(fora.sum()))
        fora = (saida < lo) | (saida > hi)
    saida[fora] = np.clip(saida[fora], lo, hi)
    return saida


def vizinhas(treino, linha, amostras, sorteio):
    """As vizinhas da entrega: (tabela com os valores, matriz de 0 e 1)."""
    rng = np.random.default_rng(sorteio)
    valores = {}
    binaria = np.ones((amostras, _K), dtype=float)
    for j, c in enumerate(fl.COLUNAS):
        col = treino[c]
        x0 = linha[c]
        if c in fl.CATEGORIAS or c == "vespera_feriado":
            cats, freq = np.unique(col.to_numpy(), return_counts=True)
            sorteio_c = cats[rng.choice(len(cats), amostras, p=freq / freq.sum())]
            sorteio_c[0] = x0
            binaria[:, j] = (sorteio_c == x0).astype(float)
            valores[c] = sorteio_c
            continue
        v = col.to_numpy(dtype=float)
        q = _quartis(treino, c)
        faixa_treino = np.searchsorted(q, v)
        nf = len(q) + 1
        freq = np.bincount(faixa_treino, minlength=nf).astype(float)
        bordas = np.r_[v.min(), q, v.max()]
        faixa_x = int(np.searchsorted(q, x0))
        fx = rng.choice(nf, amostras, p=freq / freq.sum())
        fx[0] = faixa_x
        binaria[:, j] = (fx == faixa_x).astype(float)
        saida = np.empty(amostras)
        for k in range(nf):
            quem = np.flatnonzero(fx == k)
            if len(quem) == 0:
                continue
            sel = v[faixa_treino == k]
            media = sel.mean() if len(sel) else 0.0
            desvio = (sel.std() if len(sel) else 0.0) + 1e-11
            saida[quem] = _normal_cortada(rng, media, desvio, bordas[k], bordas[k + 1], len(quem))
        saida[0] = x0
        valores[c] = saida
    return pd.DataFrame(valores), binaria


def ridge_pesado(z, y, w, alfa=1.0):
    """Ridge com intercepto e peso por linha, como o scikit-learn: o intercepto não é penalizado
    e o peso de cada vizinha entra como está, sem normalizar. Com vizinhas que contam pouco, a
    penalidade alfa pesa mais."""
    zm = (w[:, None] * z).sum(0) / w.sum()
    ym = (w * y).sum() / w.sum()
    zc, yc = z - zm, y - ym
    a = zc.T @ (w[:, None] * zc) + alfa * np.eye(z.shape[1])
    coef = np.linalg.solve(a, zc.T @ (w * yc))
    inter = ym - zm @ coef
    pred = inter + z @ coef
    r2 = 1 - (w * (y - pred) ** 2).sum() / (w * (y - ym) ** 2).sum()
    return coef, inter, r2


def largura_padrao():
    return 0.75 * math.sqrt(_K)


def lime(modelo, treino, linha, amostras=5000, largura=None, sorteio=0, vizinhanca=None, alfa=1.0):
    """Retorna dict com coeficientes das 7 colunas, intercepto, R² com peso, previsão local e nota.

    vizinhanca: (tabela, binaria, notas) já sorteadas; com ela, só a largura muda.
    """
    largura = largura_padrao() if largura is None else float(largura)
    if largura <= 0:
        raise SystemExit(f"--largura precisa ser maior que zero: {largura}")
    if vizinhanca is None:
        vizinhanca = sortear_vizinhanca(modelo, treino, linha, amostras, sorteio)
    _, z, notas = vizinhanca
    d2 = (1 - z).sum(axis=1)                       # quantas colunas saíram da faixa da entrega
    w = np.sqrt(np.exp(-d2 / largura ** 2))
    with np.errstate(all="ignore"):
        coef, inter, r2 = ridge_pesado(z, notas, w, alfa)
    return {"coef": coef, "intercepto": inter, "r2": r2, "local": inter + coef.sum(),
            "nota": notas[0], "largura": largura}


def sortear_vizinhanca(modelo, treino, linha, amostras, sorteio):
    tabela, z = vizinhas(treino, linha, amostras, sorteio)
    x, _ = fl.matriz(tabela)
    return tabela, z, modelo.nota(x)
