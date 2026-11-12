"""A floresta da aula «SHAP e LIME», nas entregas da Loja Exemplo.

Só pandas e NumPy. Tudo sai igual toda vez que você roda.

ler(caminho): lê as entregas e tira maio e junho de 2026 logo na leitura. Esses dois meses são
    o teste e ficam fechados. O rótulo `atrasou` é 1 quando a situação não é `no_prazo`.
separar(base): julho de 2025 a fevereiro de 2026 é o treino; março e abril de 2026 são as
    entregas que a aula explica. São 7 colunas. A transportadora entra na floresta como três
    colunas de 0 e 1, mas as explicações tratam as três como uma coluna só.

A árvore e a floresta são as da aula 04, «Random Forest», com a folha mínima de 20 entregas da
aula 11: cada coluna tem até 32 cortes possíveis, tirados do treino, e cada nó sorteia algumas
colunas.
"""
import numpy as np
import pandas as pd

FIM_TREINO = "2026-02-28"
FIM_VALIDACAO = "2026-04-30"
COLUNAS = ["distancia_km", "valor_frete", "prazo_prometido_dias", "peso_kg", "itens",
           "transportadora", "vespera_feriado"]
CATEGORIAS = {"transportadora": ["Economica", "Propria", "Rapida"]}
MAX_CORTES = 31


def ler(caminho):
    base = pd.read_csv(caminho, parse_dates=["data_despacho"])
    faltam = [c for c in COLUNAS + ["id_entrega", "situacao"] if c not in base.columns]
    if faltam:
        raise SystemExit(f"o arquivo não tem as colunas: {', '.join(faltam)}")
    base = base[base["data_despacho"] <= pd.Timestamp(FIM_VALIDACAO)].copy()   # maio e junho saem aqui
    base["atrasou"] = (base["situacao"] != "no_prazo").astype(int)
    return base.reset_index(drop=True)


def separar(base):
    treino = base[base["data_despacho"] <= pd.Timestamp(FIM_TREINO)].reset_index(drop=True)
    validacao = base[base["data_despacho"] > pd.Timestamp(FIM_TREINO)].reset_index(drop=True)
    return treino, validacao


def matriz(linhas):
    """As 7 colunas viram 9 números: a transportadora vira três colunas de 0 e 1.

    Retorna a matriz e, para cada uma das 7 colunas, os índices das colunas da matriz.
    """
    partes, grupos, k = [], [], 0
    for c in COLUNAS:
        if c in CATEGORIAS:
            for v in CATEGORIAS[c]:
                partes.append((linhas[c] == v).to_numpy(dtype=float))
            grupos.append(list(range(k, k + len(CATEGORIAS[c]))))
            k += len(CATEGORIAS[c])
        else:
            partes.append(linhas[c].to_numpy(dtype=float))
            grupos.append([k])
            k += 1
    return np.column_stack(partes), grupos


# ---------------------------------------------------------------- as faixas

def faixas(x_treino):
    """Os cortes possíveis de cada coluna, tirados só do treino (até 32 faixas)."""
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
    f = np.empty(x.shape, dtype=np.int64)
    for j, c in enumerate(cortes):
        f[:, j] = np.searchsorted(c, x[:, j], side="left")
    return f


# ---------------------------------------------------------------- a árvore

def _gini_do_no(w, c):
    w = np.asarray(w, dtype=float)
    p = np.divide(c, w, out=np.zeros_like(w), where=w > 0)
    return -w * 2 * p * (1 - p)


def _sortear_colunas(f, linhas, p_no, a, m, rng):
    """Para cada nó, m colunas sorteadas entre as que variam dentro dele."""
    ordem = np.argsort(p_no, kind="stable")
    inicio = np.searchsorted(p_no[ordem], np.arange(a))
    bloco = f[linhas[ordem]]
    varia = np.maximum.reduceat(bloco, inicio, axis=0) > np.minimum.reduceat(bloco, inicio, axis=0)
    sorteada = np.argsort(rng.random(varia.shape), axis=1)
    varia_na_ordem = np.take_along_axis(varia, sorteada, axis=1)
    posicao = np.argsort(~varia_na_ordem, axis=1, kind="stable")
    primeiras = np.take_along_axis(sorteada, posicao, axis=1)[:, :m]
    usa = np.take_along_axis(varia_na_ordem, posicao, axis=1)[:, :m]
    primeiras = np.where(usa, primeiras, primeiras[:, :1])
    return np.sort(primeiras, axis=1)


def treinar_arvore(f, y, peso, min_folha, colunas, rng):
    """Cresce a árvore nível por nível, sem limite de profundidade (a da aula 04)."""
    ncol = f.shape[1]
    m = int(colunas)
    s1 = peso * np.asarray(y, dtype=float)
    nb = int(f.max()) + 2
    nos = {"coluna": [-1], "faixa": [-1], "esq": [-1], "dir": [-1], "valor": [0.0]}
    no_de = np.where(peso > 0, 0, -1)
    ativos = [0]
    while ativos:
        linhas = np.flatnonzero(no_de >= 0)
        linhas = linhas[np.isin(no_de[linhas], ativos)]
        mapa = np.full(len(nos["coluna"]), -1)
        mapa[ativos] = np.arange(len(ativos))
        a = len(ativos)
        p_no = mapa[no_de[linhas]]
        t_c = np.bincount(p_no, weights=peso[linhas], minlength=a)
        t_1 = np.bincount(p_no, weights=s1[linhas], minlength=a)
        for i, no in enumerate(ativos):
            nos["valor"][no] = t_1[i] / t_c[i] if t_c[i] > 0 else 0.0
        if m == ncol:
            escolhidas = np.broadcast_to(np.arange(ncol), (a, ncol))
        else:
            escolhidas = _sortear_colunas(f, linhas, p_no, a, m, rng)
        cols = escolhidas[p_no]
        chave = ((p_no[:, None] * m + np.arange(m)[None, :]) * nb + f[linhas[:, None], cols]).ravel()
        tam = a * m * nb

        def hist(v):
            return np.bincount(chave, weights=np.repeat(v[linhas], m), minlength=tam).reshape(a, m, nb)

        ec, e1 = (hist(v).cumsum(2)[:, :, :-1] for v in (peso, s1))
        dc, d1 = t_c[:, None, None] - ec, t_1[:, None, None] - e1
        pode = (ec >= min_folha) & (dc >= min_folha)
        with np.errstate(all="ignore"):
            ganho = _gini_do_no(ec, e1) + _gini_do_no(dc, d1) - _gini_do_no(t_c, t_1)[:, None, None]
        ganho = np.where(pode, ganho, -np.inf).reshape(a, -1)
        novos = []
        for i, no in enumerate(ativos):
            melhor = ganho[i].max()
            if not np.isfinite(melhor) or melhor <= 1e-9:
                continue
            k = int(np.argmax(ganho[i] >= melhor - 1e-9))
            slot, faixa = divmod(k, nb - 1)
            j = int(escolhidas[i, slot])
            esq, dir_ = len(nos["coluna"]), len(nos["coluna"]) + 1
            nos["coluna"][no], nos["faixa"][no], nos["esq"][no], nos["dir"][no] = j, faixa, esq, dir_
            for _ in range(2):
                for chave_no, v in (("coluna", -1), ("faixa", -1), ("esq", -1), ("dir", -1), ("valor", 0.0)):
                    nos[chave_no].append(v)
            daqui = linhas[p_no == i]
            vai_esq = f[daqui, j] <= faixa
            no_de[daqui[vai_esq]] = esq
            no_de[daqui[~vai_esq]] = dir_
            novos += [esq, dir_]
        ativos = novos
    return {k: np.array(v) for k, v in nos.items()}


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


# ---------------------------------------------------------------- a floresta

class Floresta:
    """A Random Forest da aula 04: cada árvore sorteia o treino com reposição."""

    def __init__(self, x, y, arvores=100, colunas=3, min_folha=20, sorteio=0):
        self.cortes = faixas(x)
        f = em_faixas(x, self.cortes)
        n = len(y)
        self.arvores = []
        for s in np.random.SeedSequence(sorteio).spawn(arvores):
            rng = np.random.default_rng(s)
            peso = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            self.arvores.append(treinar_arvore(f, y, peso, min_folha, colunas, rng))

    def nota(self, x):
        """A nota de cada linha: a média das árvores, a parte das entregas da folha que atrasou."""
        f = em_faixas(np.asarray(x, dtype=float), self.cortes)
        return np.mean([prever(a, f) for a in self.arvores], axis=0)


def acerto_do_par(nota, y):
    nota = np.asarray(nota, dtype=float)
    y = np.asarray(y)
    posto = pd.Series(nota).rank().to_numpy()
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def num(x, casas=1):
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"


def pontos(x, casas=1):
    """Diferença de nota em pontos percentuais, com sinal."""
    v = round(100 * x, casas)
    if v == 0:
        return num(0.0, casas)
    return ("+" if v > 0 else "−") + num(abs(v), casas)
