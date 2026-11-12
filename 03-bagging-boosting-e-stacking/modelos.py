"""Os modelos e as medidas da aula «Bagging, boosting e stacking».

Só pandas e NumPy. Tudo sai igual toda vez que você roda.

ler(caminho): lê as entregas e tira maio e junho de 2026 logo na leitura. Esses dois meses são
    o teste e ficam fechados. O rótulo `atrasou` é 1 quando a situação não é `no_prazo`
    (atraso curto, atraso longo e extravio entram juntos).
separar(base): julho de 2025 a fevereiro de 2026 é o treino; março e abril de 2026 medem.
    As categorias e as faixas de corte saem só do treino.

As árvores trabalham com faixas: cada coluna tem até 32 cortes possíveis, tirados do treino.
A pergunta de um nó é sempre "a coluna passa do corte?".

treinar_arvore(f, y, peso, profundidade, min_folha): árvore de classificação pelo Gini.
    `peso` diz quantas vezes cada entrega conta (o bagging usa isso para repetir entregas).
    Empate entre divisões: fica a primeira coluna e o menor corte.
treinar_bagging(f, y, arvores, rng): várias árvores sem limite, cada uma numa amostra sorteada
    com reposição do treino. A nota é a média das notas.
treinar_boosting(f, y, rodadas, profundidade, passo): árvores pequenas em sequência. Cada uma
    aprende o erro que a soma das anteriores ainda comete.
treinar_logistica(x, y): a regressão logística do curso ML supervisionado.
fora_da_dobra(treinar, prever, n, dobras): a nota de cada entrega do treino dada por um modelo
    que não viu essa entrega.
acerto_do_par(nota, y): sorteando uma entrega que atrasou e uma que chegou no prazo, em quantos
    pares a que atrasou tem a nota maior. Empate vale meio.
brier(nota, y): a média de (nota - o que aconteceu)², com 1 para atrasou e 0 para no prazo.
"""
import numpy as np
import pandas as pd

FIM_TREINO = "2026-02-28"
FIM_VALIDACAO = "2026-04-30"
NUMERICAS = ["distancia_km", "peso_kg", "itens", "valor_frete", "prazo_prometido_dias",
             "vespera_feriado", "dia_semana_despacho", "valor_pedido"]
CATEGORIAS = ["transportadora", "regiao"]
MAX_CORTES = 31


def ler(caminho):
    base = pd.read_csv(caminho, parse_dates=["data_despacho"])
    base = base[base["data_despacho"] <= pd.Timestamp(FIM_VALIDACAO)].copy()   # maio e junho saem aqui
    base["atrasou"] = (base["situacao"] != "no_prazo").astype(int)
    return base.reset_index(drop=True)


def separar(base):
    """Retorna os nomes das colunas e, para treino e validação, (x, y, datas)."""
    treino = base[base["data_despacho"] <= pd.Timestamp(FIM_TREINO)]
    validacao = base[base["data_despacho"] > pd.Timestamp(FIM_TREINO)]
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
    nomes = NUMERICAS + [f"{c}={v}" for c in CATEGORIAS for v in categorias[c]]

    def matriz(p):
        x = p[NUMERICAS].astype(float).copy()
        for c in CATEGORIAS:
            for v in categorias[c]:
                x[f"{c}={v}"] = (p[c] == v).astype(float)
        return x.to_numpy(dtype=float), p["atrasou"].to_numpy(dtype=int), p["data_despacho"].to_numpy()

    return nomes, {"treino": matriz(treino), "validacao": matriz(validacao)}


# ---------------------------------------------------------------- as faixas

def faixas(x_treino):
    """Os cortes possíveis de cada coluna, tirados só do treino.

    Coluna com poucos valores: o ponto do meio entre cada par de valores vizinhos.
    Coluna com muitos valores: os pontos que dividem o treino em 32 partes iguais.
    """
    cortes = []
    for j in range(x_treino.shape[1]):
        valores = np.unique(x_treino[:, j])
        if len(valores) <= MAX_CORTES + 1:
            c = (valores[1:] + valores[:-1]) / 2
        else:
            c = np.unique(np.quantile(x_treino[:, j], np.linspace(0, 1, MAX_CORTES + 2)[1:-1]))
        cortes.append(c)
    return cortes


def em_faixas(x, cortes):
    """A faixa de cada valor: a faixa k quer dizer valor <= cortes[k] e acima do corte anterior."""
    f = np.empty(x.shape, dtype=np.int64)
    for j, c in enumerate(cortes):
        f[:, j] = np.searchsorted(c, x[:, j], side="left")
    return f


# ---------------------------------------------------------------- as árvores

def _crescer(f, conta, s0, s1, profundidade, min_folha, ganho_do_no, valor_da_folha):
    """Cresce a árvore nível por nível. Serve para as duas árvores da aula.

    conta: quantas vezes cada linha conta no tamanho do nó (0 = a linha fica de fora).
    s0, s1: as duas somas por linha que medem o nó; ganho_do_no(s0, s1) dá a nota do nó
    (maior é melhor) e valor_da_folha(s0, s1) dá o valor da folha.
    Uma divisão vale quando os dois lados têm pelo menos min_folha e o ganho passa de zero.
    Empate entre divisões: fica a primeira coluna e o menor corte.
    """
    ncol = f.shape[1]
    nb = int(f.max()) + 2
    nos = {"coluna": [-1], "faixa": [-1], "esq": [-1], "dir": [-1], "valor": [0.0], "tamanho": [0.0]}
    no_de = np.where(conta > 0, 0, -1)
    ativos = [0]
    nivel = 0
    while ativos:
        linhas = np.flatnonzero(no_de >= 0)
        linhas = linhas[np.isin(no_de[linhas], ativos)]
        mapa = np.full(len(nos["coluna"]), -1)
        mapa[ativos] = np.arange(len(ativos))
        a = len(ativos)
        p_no = mapa[no_de[linhas]]
        t_c = np.bincount(p_no, weights=conta[linhas], minlength=a)
        t_0 = np.bincount(p_no, weights=s0[linhas], minlength=a)
        t_1 = np.bincount(p_no, weights=s1[linhas], minlength=a)
        for i, no in enumerate(ativos):
            nos["valor"][no] = valor_da_folha(t_0[i], t_1[i])
            nos["tamanho"][no] = t_c[i]
        if profundidade is not None and nivel >= profundidade:
            break
        chave = ((p_no[:, None] * ncol + np.arange(ncol)[None, :]) * nb + f[linhas]).ravel()
        tam = a * ncol * nb

        def hist(v):
            return np.bincount(chave, weights=np.repeat(v[linhas], ncol), minlength=tam).reshape(a, ncol, nb)

        ec, e0, e1 = (hist(v).cumsum(2)[:, :, :-1] for v in (conta, s0, s1))
        dc, d0, d1 = t_c[:, None, None] - ec, t_0[:, None, None] - e0, t_1[:, None, None] - e1
        pode = (ec >= min_folha) & (dc >= min_folha)
        with np.errstate(all="ignore"):
            ganho = ganho_do_no(e0, e1) + ganho_do_no(d0, d1) - ganho_do_no(t_0, t_1)[:, None, None]
        ganho = np.where(pode, ganho, -np.inf).reshape(a, -1)
        novos = []
        for i, no in enumerate(ativos):
            melhor = ganho[i].max()
            if not np.isfinite(melhor) or melhor <= 1e-9:
                continue
            k = int(np.argmax(ganho[i] >= melhor - 1e-9))
            j, faixa = divmod(k, nb - 1)
            esq, dir_ = len(nos["coluna"]), len(nos["coluna"]) + 1
            nos["coluna"][no], nos["faixa"][no], nos["esq"][no], nos["dir"][no] = j, faixa, esq, dir_
            for _ in range(2):
                for chave_no, v in (("coluna", -1), ("faixa", -1), ("esq", -1), ("dir", -1), ("valor", 0.0), ("tamanho", 0.0)):
                    nos[chave_no].append(v)
            daqui = linhas[p_no == i]
            vai_esq = f[daqui, j] <= faixa
            no_de[daqui[vai_esq]] = esq
            no_de[daqui[~vai_esq]] = dir_
            novos += [esq, dir_]
        ativos = novos
        nivel += 1
    return {k: np.array(v) for k, v in nos.items()}


def _gini_do_no(w, c):
    """Tamanho do nó vezes o Gini 2p(1 - p), com sinal trocado: maior é melhor."""
    w = np.asarray(w, dtype=float)
    p = np.divide(c, w, out=np.zeros_like(w), where=w > 0)
    return -w * 2 * p * (1 - p)


def treinar_arvore(f, y, peso=None, profundidade=None, min_folha=1):
    """Árvore de classificação pelo Gini. A nota de uma folha é a parte que atrasou nela."""
    peso = np.ones(len(y)) if peso is None else np.asarray(peso, dtype=float)
    y = np.asarray(y, dtype=float)
    return _crescer(f, peso, peso, peso * y, profundidade, min_folha, _gini_do_no,
                    lambda w, c: c / w if w > 0 else 0.0)


def prever(arvore, f):
    no = np.zeros(len(f), dtype=np.int64)
    while True:
        interno = arvore["coluna"][no] >= 0
        if not interno.any():
            return arvore["valor"][no]
        quem = np.flatnonzero(interno)
        atual = no[quem]
        vai_esq = f[quem, arvore["coluna"][atual]] <= arvore["faixa"][atual]
        no[quem] = np.where(vai_esq, arvore["esq"][atual], arvore["dir"][atual])


def folhas(arvore):
    return int((arvore["coluna"] < 0).sum())


def altura(arvore):
    def h(no):
        if arvore["coluna"][no] < 0:
            return 0
        return 1 + max(h(arvore["esq"][no]), h(arvore["dir"][no]))
    return h(0)


# ---------------------------------------------------------------- bagging

def treinar_bagging(f, y, arvores, rng, min_folha=1):
    """Cada árvore cresce sem limite numa amostra sorteada com reposição, do tamanho do treino.

    Retorna as árvores e, para cada uma, o peso de cada entrega (0 quando ela ficou de fora).
    """
    n = len(y)
    modelos, pesos = [], []
    for _ in range(arvores):
        peso = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        modelos.append(treinar_arvore(f, y, peso=peso, min_folha=min_folha))
        pesos.append(peso)
    return modelos, pesos


# ---------------------------------------------------------------- boosting

def _newton_do_no(h, g):
    """Quanto o nó baixa a perda: (soma dos erros)² / (soma das curvaturas + 1)."""
    return g ** 2 / (h + 1.0)


def treinar_boosting(f, y, rodadas=300, profundidade=3, passo=0.1, min_folha=20):
    """Boosting de árvores pequenas para a perda logística.

    Começa com a mesma nota para todos (a parte que atrasou no treino). Em cada rodada calcula,
    para cada entrega, o erro da nota atual (y - nota) e treina uma árvore pequena nesse erro.
    A árvore entra na soma multiplicada pelo passo.
    """
    y = np.asarray(y, dtype=float)
    base = float(np.log(y.mean() / (1 - y.mean())))
    soma = np.full(len(y), base)
    um = np.ones(len(y))
    arvores = []
    for _ in range(rodadas):
        p = 1 / (1 + np.exp(-soma))
        a = _crescer(f, um, p * (1 - p), y - p, profundidade, min_folha,
                     _newton_do_no, lambda h, g: g / (h + 1.0))
        arvores.append(a)
        soma = soma + passo * prever(a, f)
    return {"base": base, "passo": passo, "arvores": arvores}


def nota_boosting(modelo, f, rodadas=None):
    arvores = modelo["arvores"] if rodadas is None else modelo["arvores"][:rodadas]
    soma = np.full(len(f), modelo["base"])
    for a in arvores:
        soma = soma + modelo["passo"] * prever(a, f)
    return 1 / (1 + np.exp(-soma))


def notas_por_rodada(modelo, f, rodadas):
    """A nota depois de cada número de rodadas da lista, sem recalcular do zero."""
    soma = np.full(len(f), modelo["base"])
    saida, feitas = {}, 0
    for r in sorted(rodadas):
        for a in modelo["arvores"][feitas:r]:
            soma = soma + modelo["passo"] * prever(a, f)
        feitas = r
        saida[r] = 1 / (1 + np.exp(-soma))
    return saida


# ---------------------------------------------------------------- a regressão logística

def treinar_logistica(x, y, penalidade=1.0, passos=30, padronizar=True):
    x = np.asarray(x, dtype=float)
    if padronizar:
        media, desvio = x.mean(axis=0), x.std(axis=0)
        desvio[desvio == 0] = 1.0
    else:
        media, desvio = np.zeros(x.shape[1]), np.ones(x.shape[1])
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


# ---------------------------------------------------------------- stacking

def dobras(n, k):
    """k pedaços seguidos, na ordem das linhas (as entregas estão em ordem de data)."""
    return np.array_split(np.arange(n), k)


def fora_da_dobra(treinar, prever_nota, n, k=5):
    """A nota de cada linha do treino, dada por um modelo treinado nas outras k - 1 dobras.

    treinar(linhas) treina um modelo só com essas linhas; prever_nota(modelo, linhas) dá a nota.
    """
    nota = np.full(n, np.nan)
    for fora in dobras(n, k):
        dentro = np.setdiff1d(np.arange(n), fora)
        nota[fora] = prever_nota(treinar(dentro), fora)
    return nota


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
