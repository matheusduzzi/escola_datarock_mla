"""A árvore, as referências e as medidas da aula «Árvore na Loja Exemplo».

ler(caminho): lê os pedidos e tira junho logo na leitura. Junho é o teste e fica fechado.
separar(base): janeiro a março é o treino; abril e maio medem. As categorias saem só do treino.
treinar_arvore(x, y, profundidade, min_folha): árvore de classificação pelo Gini. Em cada nó,
    testa todas as colunas e todos os cortes e fica com a divisão que mais baixa o Gini.
    Para quando o nó é puro, quando chega na profundidade ou quando não sobra divisão
    com pelo menos min_folha pedidos de cada lado.
nota(arvore, x): a nota de cada pedido, que é a parte que cancelou na folha dele, no treino.
treinar_logistica(x, y) e nota_logistica(modelo, x): a regressão logística do curso anterior,
    com as colunas padronizadas pelo treino e uma penalidade pequena.
acerto_do_par(nota, y): sorteando um cancelado e um entregue, em quantos pares o cancelado tem a
    nota maior. Empate vale meio.
brier(nota, y): a média de (nota - o que aconteceu)², com 1 para cancelado e 0 para entregue.
"""
import numpy as np
import pandas as pd

FIM_TREINO = "2026-03-31"
FIM_ABRIL = "2026-04-30"
FIM_VALIDACAO = "2026-05-31"
NUMERICAS = ["valor", "prazo_prometido_dias", "pedidos_anteriores", "cliente_novo"]
CATEGORIAS = ["categoria", "forma_pagamento"]


def ler(caminho):
    base = pd.read_csv(caminho, parse_dates=["data_pedido"])
    return base[base["data_pedido"] <= pd.Timestamp(FIM_VALIDACAO)]    # junho sai aqui


def separar(base):
    """Retorna os nomes das colunas e (x, y, id) do treino, de abril e de maio."""
    treino = base[base["data_pedido"] <= pd.Timestamp(FIM_TREINO)]
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
    nomes = NUMERICAS + [f"{c}={v}" for c in CATEGORIAS for v in categorias[c]]

    def matriz(p):
        x = p[NUMERICAS].astype(float).copy()
        for c in CATEGORIAS:
            for v in categorias[c]:
                x[f"{c}={v}"] = (p[c] == v).astype(float)
        return x.to_numpy(dtype=float), p["cancelado"].to_numpy(dtype=int), p["id_pedido"].to_numpy()

    abril = base[(base["data_pedido"] > pd.Timestamp(FIM_TREINO)) & (base["data_pedido"] <= pd.Timestamp(FIM_ABRIL))]
    maio = base[base["data_pedido"] > pd.Timestamp(FIM_ABRIL)]
    return nomes, {"treino": matriz(treino), "abril": matriz(abril), "maio": matriz(maio)}


# ---------------------------------------------------------------- a árvore

def gini(cancelados, total):
    p = cancelados / total
    return 2 * p * (1 - p)


def _melhor_divisao(x, y, min_folha):
    """A coluna e o corte que mais baixam o Gini, com pelo menos min_folha de cada lado.

    Empate: fica a primeira coluna na ordem de x e o menor corte. Retorna None quando nenhuma
    divisão baixa o Gini.
    """
    n = len(y)
    total_c = y.sum()
    gini_no = gini(total_c, n)
    melhor = None
    for j in range(x.shape[1]):
        ordem = np.argsort(x[:, j], kind="mergesort")
        xs, ys = x[ordem, j], y[ordem]
        esq_n = np.arange(1, n)
        esq_c = np.cumsum(ys)[:-1]
        pode = (xs[1:] != xs[:-1]) & (esq_n >= min_folha) & (n - esq_n >= min_folha)
        if not pode.any():
            continue
        dir_n, dir_c = n - esq_n, total_c - esq_c
        filhos = (esq_n * gini(esq_c, esq_n) + dir_n * gini(dir_c, dir_n)) / n
        filhos = np.where(pode, filhos, np.inf)
        k = int(np.argmax(filhos <= filhos.min() + 1e-12))    # empate: o menor corte
        ganho = gini_no - filhos[k]
        if ganho > 1e-12 and (melhor is None or ganho > melhor[2] + 1e-12):
            melhor = (j, (xs[k] + xs[k + 1]) / 2, ganho)
    return melhor


def treinar_arvore(x, y, profundidade=None, min_folha=1):
    """profundidade=None: cresce até cada folha ser pura ou não ter mais divisão."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    def crescer(idx, nivel):
        n, c = len(idx), int(y[idx].sum())
        no = {"pedidos": n, "cancelados": c, "nota": c / n}
        if c in (0, n) or (profundidade is not None and nivel >= profundidade) or n < 2 * min_folha:
            return no
        divisao = _melhor_divisao(x[idx], y[idx], min_folha)
        if divisao is None:
            return no
        j, corte, _ = divisao
        vai_esq = x[idx, j] <= corte
        no.update(coluna=j, corte=float(corte),
                  esq=crescer(idx[vai_esq], nivel + 1), dir=crescer(idx[~vai_esq], nivel + 1))
        return no

    return crescer(np.arange(len(y)), 0)


def nota(arvore, x):
    x = np.asarray(x, dtype=float)
    saida = np.empty(len(x))
    for i, linha in enumerate(x):
        no = arvore
        while "coluna" in no:
            no = no["esq"] if linha[no["coluna"]] <= no["corte"] else no["dir"]
        saida[i] = no["nota"]
    return saida


def folhas(arvore):
    if "coluna" not in arvore:
        return [arvore]
    return folhas(arvore["esq"]) + folhas(arvore["dir"])


def altura(arvore):
    if "coluna" not in arvore:
        return 0
    return 1 + max(altura(arvore["esq"]), altura(arvore["dir"]))


def colunas_usadas(arvore, nomes):
    if "coluna" not in arvore:
        return set()
    return {nomes[arvore["coluna"]]} | colunas_usadas(arvore["esq"], nomes) | colunas_usadas(arvore["dir"], nomes)


def _corte(x):
    texto = num(x, 2)
    return texto[:-3] if texto.endswith(",00") else texto.rstrip("0")


def desenhar(arvore, nomes, margem=""):
    """O texto da árvore: a pergunta de cada nó e, em cada folha, pedidos e parte que cancelou."""
    if "coluna" not in arvore:
        n = arvore["pedidos"]
        return [f"{margem}folha: {n} pedido{'s' if n != 1 else ''}, {pct(arvore['nota'])} cancelaram"]
    nome = nomes[arvore["coluna"]]
    if "=" in nome or nome == "cliente_novo":       # coluna 0 ou 1: a pergunta é direta
        pergunta = f"{nome.split('=')[0]} é {nome.split('=')[1]}" if "=" in nome else "cliente novo"
        ramos = [("não", arvore["esq"]), ("sim", arvore["dir"])]
    else:
        pergunta = f"{nome} <= {_corte(arvore['corte'])}"
        ramos = [("sim", arvore["esq"]), ("não", arvore["dir"])]
    linhas = []
    for resposta, filho in ramos:
        linhas.append(f"{margem}{pergunta}? {resposta}:")
        linhas += desenhar(filho, nomes, margem + "    ")
    return linhas


# ---------------------------------------------------------------- a regressão logística

def treinar_logistica(x, y, penalidade=1.0, passos=30):
    x = np.asarray(x, dtype=float)
    media, desvio = x.mean(axis=0), x.std(axis=0)
    desvio[desvio == 0] = 1.0
    m = np.c_[np.ones(len(x)), (x - media) / desvio]
    y = np.asarray(y, dtype=float)
    coef = np.zeros(m.shape[1])
    pen = np.eye(m.shape[1]) * penalidade
    pen[0, 0] = 0
    with np.errstate(all="ignore"):
        for _ in range(passos):
            p = 1 / (1 + np.exp(-(m @ coef)))
            grad = m.T @ (p - y) + pen @ coef
            hess = m.T @ (m * (p * (1 - p))[:, None]) + pen
            coef = coef - np.linalg.solve(hess, grad)
    return {"media": media, "desvio": desvio, "coef": coef}


def nota_logistica(modelo, x):
    m = np.c_[np.ones(len(x)), (np.asarray(x, dtype=float) - modelo["media"]) / modelo["desvio"]]
    with np.errstate(all="ignore"):
        return 1 / (1 + np.exp(-(m @ modelo["coef"])))


# ---------------------------------------------------------------- as medidas

def acerto_do_par(nota, y):
    posto = pd.Series(np.asarray(nota, dtype=float)).rank().to_numpy()
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(nota, y):
    return float(np.mean((np.asarray(nota, dtype=float) - np.asarray(y)) ** 2))


def num(x, casas=1):
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"
