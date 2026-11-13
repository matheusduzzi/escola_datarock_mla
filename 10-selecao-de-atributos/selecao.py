"""Os três jeitos de escolher colunas da aula «Seleção de atributos».

Só pandas e NumPy. Tudo sai igual toda vez que você roda.

ler(caminho): lê as entregas e tira maio e junho de 2026 logo na leitura. Esses dois meses são
    o teste e ficam fechados. O rótulo `atrasou` é 1 quando a situação não é `no_prazo`
    (atraso curto, atraso longo e extravio entram juntos).
separar(base): julho de 2025 a fevereiro de 2026 é o treino; março e abril de 2026 medem.
    As categorias saem só do treino. Além das 16 colunas de sempre, cada rota de coleta
    (`cod_rota`) vira uma coluna 0 ou 1: `rota=1`, `rota=2`, ...

filtro(x, y): a correlação de cada coluna, sozinha, com `atrasou`. Não treina modelo nenhum.
treinar_logistica(x, y, penalidade): a regressão logística do curso ML supervisionado, com a
    penalidade L2 de sempre (os coeficientes encolhem, mas nenhum vai a zero).
treinar_lasso(x, y, forca): a regressão logística com penalidade L1. Com força suficiente,
    coeficientes vão a zero, e a coluna sai do modelo durante o treino.
para_frente(x, y, colunas, avaliar): o wrapper. Começa sem coluna nenhuma e, a cada passo,
    põe a coluna que mais melhora `avaliar`.
fora_da_dobra(treinar, prever, n, dobras): a nota de cada entrega do treino dada por um modelo
    que não viu essa entrega. Tudo que `treinar` faz, inclusive escolher colunas, acontece
    só com as linhas que ele recebe.
acerto_do_par(nota, y): sorteando uma entrega que atrasou e uma que chegou no prazo, em quantos
    pares a que atrasou tem a nota maior. Empate vale meio.
"""
import numpy as np
import pandas as pd

FIM_TREINO = "2026-02-28"
FIM_VALIDACAO = "2026-04-30"
NUMERICAS = ["distancia_km", "peso_kg", "itens", "valor_frete", "prazo_prometido_dias",
             "vespera_feriado", "dia_semana_despacho", "valor_pedido"]
CATEGORIAS = ["transportadora", "regiao"]


def ler(caminho):
    base = pd.read_csv(caminho, parse_dates=["data_despacho"])
    base = base[base["data_despacho"] <= pd.Timestamp(FIM_VALIDACAO)].copy()   # maio e junho saem aqui
    base["atrasou"] = (base["situacao"] != "no_prazo").astype(int)
    return base.reset_index(drop=True)


def separar(base):
    """Retorna os nomes das colunas e, para treino e validação, (x, y)."""
    treino = base[base["data_despacho"] <= pd.Timestamp(FIM_TREINO)]
    validacao = base[base["data_despacho"] > pd.Timestamp(FIM_TREINO)]
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
    rotas = sorted(treino["cod_rota"].unique())
    nomes = (NUMERICAS + [f"{c}={v}" for c in CATEGORIAS for v in categorias[c]]
             + [f"rota={r}" for r in rotas])

    def matriz(p):
        partes = [p[NUMERICAS].to_numpy(dtype=float)]
        for c in CATEGORIAS:
            partes.append(np.column_stack([(p[c] == v).to_numpy(dtype=float) for v in categorias[c]]))
        partes.append((p["cod_rota"].to_numpy()[:, None] == np.array(rotas)[None, :]).astype(float))
        return np.hstack(partes), p["atrasou"].to_numpy(dtype=int)

    return nomes, {"treino": matriz(treino), "validacao": matriz(validacao)}


# ---------------------------------------------------------------- filtro

def filtro(x, y):
    """A correlação de cada coluna com y, calculada só nas linhas recebidas. Coluna constante dá 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    den = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(all="ignore"):
        cov = xc.T @ yc
    return np.divide(cov, den, out=np.zeros(x.shape[1]), where=den > 0)


def melhores_pelo_filtro(x, y, candidatas, k):
    """As k colunas de `candidatas` com a maior correlação, em valor absoluto. Empate: a primeira."""
    if k == 0:
        return []
    r = np.abs(filtro(x[:, candidatas], y))
    ordem = np.argsort(-r, kind="stable")
    return [candidatas[i] for i in ordem[:k]]


# ---------------------------------------------------------------- regressões logísticas

def _padronizar(x):
    media, desvio = x.mean(axis=0), x.std(axis=0)
    desvio = np.where(desvio == 0, 1.0, desvio)
    return media, desvio


def treinar_logistica(x, y, penalidade=1.0, passos=12):
    x = np.asarray(x, dtype=float)
    media, desvio = _padronizar(x)
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


def nota(modelo, x):
    m = np.c_[np.ones(len(x)), (np.asarray(x, dtype=float) - modelo["media"]) / modelo["desvio"]]
    with np.errstate(all="ignore"):
        return 1 / (1 + np.exp(-(m @ modelo["coef"])))


def treinar_lasso(x, y, forca, inicio=None, iteracoes=400):
    """Logística com penalidade L1 nas colunas padronizadas, pela descida de gradiente acelerada.

    Perda: média da log-loss + forca × soma de |coeficiente|. O intercepto não é penalizado.
    Depois de cada passo, todo coeficiente menor que o limiar vira zero exato.
    """
    x = np.asarray(x, dtype=float)
    media, desvio = _padronizar(x)
    z = (x - media) / desvio
    y = np.asarray(y, dtype=float)
    n, p = z.shape
    m = np.c_[np.ones(n), z]
    # passo seguro: 1 / (maior autovalor de m'm / 4n)
    v = np.ones(p + 1) / np.sqrt(p + 1)
    for _ in range(30):
        v = m.T @ (m @ v)
        v = v / np.linalg.norm(v)
    passo = 4 * n / float(v @ (m.T @ (m @ v)))
    coef = np.zeros(p + 1) if inicio is None else inicio.copy()
    coef[0] = np.log(y.mean() / (1 - y.mean())) if inicio is None else coef[0]
    anterior, acel, t = coef.copy(), coef.copy(), 1.0
    for _ in range(iteracoes):
        with np.errstate(all="ignore"):
            pr = 1 / (1 + np.exp(-(m @ acel)))
        grad = m.T @ (pr - y) / n
        novo = acel - passo * grad
        novo[1:] = np.sign(novo[1:]) * np.maximum(np.abs(novo[1:]) - passo * forca, 0.0)
        t_novo = (1 + np.sqrt(1 + 4 * t * t)) / 2
        acel = novo + ((t - 1) / t_novo) * (novo - anterior)
        anterior, t = novo, t_novo
    return {"media": media, "desvio": desvio, "coef": anterior}


def colunas_que_ficam(modelo):
    return np.flatnonzero(modelo["coef"][1:] != 0)


# ---------------------------------------------------------------- wrapper

def para_frente(colunas, avaliar, ate=None):
    """Seleção para a frente. avaliar(lista_de_colunas) retorna um número, maior é melhor.

    Retorna a lista de passos: (coluna que entrou, nota com as colunas até ali).
    `treinos` conta quantas vezes `avaliar` foi chamada.
    """
    escolhidas, restantes, passos, treinos = [], list(colunas), [], 0
    ate = len(colunas) if ate is None else ate
    while restantes and len(escolhidas) < ate:
        notas = []
        for c in restantes:
            notas.append(avaliar(escolhidas + [c]))
            treinos += 1
        i = int(np.argmax(notas))
        escolhidas.append(restantes.pop(i))
        passos.append((escolhidas[-1], notas[i]))
    return passos, treinos


def menor_a_ate(valores, tolerancia):
    """O primeiro índice cujo valor fica a até `tolerancia` do maior valor da lista."""
    valores = np.asarray(valores, dtype=float)
    return int(np.argmax(valores >= valores.max() - tolerancia))


# ---------------------------------------------------------------- validação

def dobras(n, k):
    """k pedaços seguidos, na ordem das linhas (as entregas estão em ordem de data)."""
    return np.array_split(np.arange(n), k)


def fora_da_dobra(treinar, prever_nota, n, k=5):
    """A nota de cada linha do treino, dada por um modelo treinado nas outras k - 1 dobras.

    treinar(linhas) treina um modelo só com essas linhas; prever_nota(modelo, linhas) dá a nota.
    """
    saida = np.full(n, np.nan)
    for fora in dobras(n, k):
        dentro = np.setdiff1d(np.arange(n), fora)
        saida[fora] = prever_nota(treinar(dentro), fora)
    return saida


def acerto_do_par(nota_, y):
    posto = pd.Series(np.asarray(nota_, dtype=float)).rank().to_numpy()
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rotas_acima_por_sorte(tamanhos, taxa, limite):
    """Se toda rota tivesse a mesma chance `taxa` de atraso, quantas passariam de `limite` em média.

    Para cada rota com n entregas, soma a chance (binomial) de mais de limite × n atrasarem.
    """
    from math import comb
    total = 0.0
    for n in tamanhos:
        n = int(n)
        total += sum(comb(n, a) * taxa ** a * (1 - taxa) ** (n - a) for a in range(n + 1) if a > limite * n)
    return total


def faixa_da_validacao(nota_, y, rng, sorteios=500):
    """Sorteia as linhas com reposição e retorna os percentis 5 e 95 do acerto do par."""
    nota_, y = np.asarray(nota_), np.asarray(y)
    valores = []
    for _ in range(sorteios):
        i = rng.integers(0, len(y), len(y))
        valores.append(acerto_do_par(nota_[i], y[i]))
    return float(np.percentile(valores, 5)), float(np.percentile(valores, 95))


def num(x, casas=1):
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"
