"""O modelo de cancelamento da aula «Sucesso do cliente».

Só pandas e NumPy. Tudo sai igual toda vez que você roda.

ler(caminho): lê os clientes do Sistema Exemplo. Uma linha é um cliente ativo numa foto
    trimestral; o rótulo `cancelou_90d` é 1 quando ele cancelou nos 90 dias depois da foto.
dividir(base): as fotos de julho e outubro de 2025 são o treino, a de janeiro de 2026 mede,
    a de abril de 2026 é o teste e só abre com `abrir_teste=True`.
Preparo(treino, com_contato, nps_media): as colunas do modelo, aprendidas só no treino:
    as categorias de plano e segmento, a mediana do NPS e a faixa de cada coluna.
    `com_contato` acrescenta `recebeu_contato_30d`; `nps_media` troca a coluna `nps_vazio`
    pelo NPS vazio preenchido com a média do treino.
guarda(preparo, foto): antes de dar nota, separa as linhas que o treino não cobre
    (categoria nova, valor que passa da faixa do treino em mais de 10% da largura dela) e para com erro se faltar coluna ou se a foto vier
    com cliente repetido ou com mais de uma data.
quantos_na_lista(nota, y, k): dos k clientes com a maior nota, quantos cancelaram. Empate na
    fronteira da lista conta pela média dos empatados.

A floresta é a das aulas 04, «Random Forest», e 11, «Importâncias nativa e por permutação»:
cada coluna tem até 32 cortes tirados do treino e cada nó sorteia algumas colunas. A regressão
logística é a da aula 02, «Árvore na Loja Exemplo», com as colunas padronizadas pelo treino.
"""
import numpy as np
import pandas as pd

TREINO = ["2025-07-01", "2025-10-01"]
VALIDACAO = "2026-01-01"
TESTE = "2026-04-01"
NUMERICAS = ["meses_de_casa", "usuarios_contratados", "logins_30d", "variacao_logins_pct",
             "chamados_30d", "atraso_pagamento_dias", "desconto_ativo"]
CATEGORIAS = ["plano", "segmento"]
MAX_CORTES = 31
FOLGA = 0.10   # quanto um valor pode passar da faixa do treino, em parte da faixa
ESPERADAS = (["id_cliente", "data_referencia"] + NUMERICAS + CATEGORIAS +
             ["nps_ultimo", "recebeu_contato_30d"])


class ErroDaFoto(ValueError):
    """A foto não serve para dar nota: falta coluna, cliente repetido ou datas misturadas."""


def ler(caminho):
    return pd.read_csv(caminho, dtype={"data_referencia": str})


def dividir(base, abrir_teste=False):
    """Retorna treino, validação e, se abrir_teste, o teste. Sem abrir, o teste fica None."""
    treino = base[base["data_referencia"].isin(TREINO)].reset_index(drop=True)
    validacao = base[base["data_referencia"] == VALIDACAO].reset_index(drop=True)
    teste = base[base["data_referencia"] == TESTE].reset_index(drop=True) if abrir_teste else None
    return treino, validacao, teste


def sinais_da_equipe(foto):
    """Os quatro sinais que a equipe usa para escolher para quem ligar, somados (0 a 4)."""
    return ((foto["variacao_logins_pct"] <= -30).astype(int) + (foto["atraso_pagamento_dias"] > 10).astype(int)
            + (foto["chamados_30d"] >= 3).astype(int) + (foto["nps_ultimo"] <= 6).astype(int)).to_numpy()


# ---------------------------------------------------------------- as colunas e a guarda

class Preparo:
    def __init__(self, treino, com_contato=False, nps_media=False):
        self.com_contato, self.nps_media = com_contato, nps_media
        self.categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
        self.nps_preenche = float(treino["nps_ultimo"].mean() if nps_media else treino["nps_ultimo"].median())
        self.faixa = {c: (float(treino[c].min()), float(treino[c].max())) for c in NUMERICAS}
        self.nps_vazio_treino = float(treino["nps_ultimo"].isna().mean())
        self.nomes = (NUMERICAS + ["nps"] + ([] if nps_media else ["nps_vazio"])
                      + (["recebeu_contato_30d"] if com_contato else [])
                      + [f"{c}={v}" for c in CATEGORIAS for v in self.categorias[c]])

    def matriz(self, foto):
        x = foto[NUMERICAS].astype(float).copy()
        x["nps"] = foto["nps_ultimo"].fillna(self.nps_preenche).astype(float)
        if not self.nps_media:
            x["nps_vazio"] = foto["nps_ultimo"].isna().astype(float)
        if self.com_contato:
            x["recebeu_contato_30d"] = foto["recebeu_contato_30d"].astype(float)
        for c in CATEGORIAS:
            for v in self.categorias[c]:
                x[f"{c}={v}"] = (foto[c] == v).astype(float)
        return x[self.nomes].to_numpy(dtype=float)


def guarda(preparo, foto):
    """Confere a foto antes da nota.

    Para com ErroDaFoto quando falta coluna, quando há mais de uma data de referência, quando um
    cliente aparece duas vezes ou quando uma coluna numérica vem vazia. Retorna (fora, motivos,
    avisos): `fora` marca as linhas que vão para a revisão à mão, porque o treino não tem
    nada parecido; `motivos` diz por quê, linha a linha; `avisos` traz o que muda na foto inteira.
    """
    faltam = [c for c in ESPERADAS if c not in foto.columns]
    if faltam:
        raise ErroDaFoto(f"faltam colunas na foto: {', '.join(faltam)}")
    datas = sorted(foto["data_referencia"].astype(str).unique())
    if len(datas) != 1:
        raise ErroDaFoto(f"a foto precisa ter uma data de referência só; veio com {len(datas)}: {', '.join(datas)}")
    repetidos = foto["id_cliente"][foto["id_cliente"].duplicated()].unique()
    if len(repetidos):
        raise ErroDaFoto(f"cliente repetido na foto: {', '.join(str(i) for i in repetidos[:5])}")
    vazias = [c for c in NUMERICAS if foto[c].isna().any()]
    if vazias:
        raise ErroDaFoto(f"coluna numérica com valor vazio: {', '.join(vazias)}")
    motivos = [[] for _ in range(len(foto))]
    for c in CATEGORIAS:
        for i in np.flatnonzero(~foto[c].isin(preparo.categorias[c]).to_numpy()):
            motivos[i].append(f"{c} novo: {foto[c].iloc[i]}")
    for c in NUMERICAS:
        menor, maior = preparo.faixa[c]
        folga = FOLGA * (maior - menor)
        v = foto[c].to_numpy(dtype=float)
        for i in np.flatnonzero((v < menor - folga) | (v > maior + folga)):
            motivos[i].append(f"{c} fora da faixa do treino ({v[i]:g}; treino de {menor:g} a {maior:g})")
    fora = np.array([bool(m) for m in motivos])
    avisos = []
    vazio = float(foto["nps_ultimo"].isna().mean())
    if abs(vazio - preparo.nps_vazio_treino) > 0.10:
        avisos.append(f"NPS vazio em {pct(vazio)} da foto; no treino eram {pct(preparo.nps_vazio_treino)}")
    return fora, motivos, avisos


# ---------------------------------------------------------------- as medidas

def acerto_do_par(nota, y):
    """Sorteando um cliente que cancelou e um que ficou, em quantos pares o que cancelou tem a nota
    maior. Empate vale meio."""
    posto = pd.Series(np.asarray(nota, dtype=float)).rank().to_numpy()
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def quantos_na_lista(nota, y, k):
    """Dos k clientes de maior nota, quantos cancelaram. Os empatados na fronteira entram pela média."""
    nota, y = np.asarray(nota, dtype=float), np.asarray(y, dtype=float)
    if not 1 <= k <= len(nota):
        raise ValueError(f"a lista precisa ter de 1 a {len(nota)} clientes: {k}")
    corte = np.sort(nota)[::-1][k - 1]
    acima = nota > corte
    empate = nota == corte
    return float(y[acima].sum() + (k - acima.sum()) * y[empate].mean())


def lista(nota, k, fora=None):
    """As posições dos k clientes de maior nota, deixando de fora as linhas marcadas pela guarda.
    Empate: fica quem aparece antes na foto."""
    nota = np.asarray(nota, dtype=float).copy()
    if fora is not None:
        nota[np.asarray(fora)] = -np.inf
    disponiveis = int(np.isfinite(nota).sum())
    if not 1 <= k <= disponiveis:
        raise ValueError(f"a lista precisa ter de 1 a {disponiveis} clientes: {k}")
    return np.argsort(-nota, kind="stable")[:k]


def num(x, casas=1):
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"


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


# ---------------------------------------------------------------- a árvore

def _gini_do_no(w, c):
    """Tamanho do nó vezes o Gini 2p(1 - p), com sinal trocado: maior é melhor."""
    w = np.asarray(w, dtype=float)
    p = np.divide(c, w, out=np.zeros_like(w), where=w > 0)
    return -w * 2 * p * (1 - p)


def _sortear_colunas(f, linhas, p_no, a, m, rng):
    """Para cada nó, m colunas sorteadas entre as que variam dentro dele, em ordem crescente.

    Coluna que tem um valor só no nó não conta no sorteio, como no scikit-learn: o nó segue
    sorteando até achar m colunas que variam. Nó com menos de m colunas que variam fica com
    todas elas (o resto da linha repete uma coluna parada, que não divide nada).
    """
    ordem = np.argsort(p_no, kind="stable")
    inicio = np.searchsorted(p_no[ordem], np.arange(a))
    bloco = f[linhas[ordem]]
    varia = np.maximum.reduceat(bloco, inicio, axis=0) > np.minimum.reduceat(bloco, inicio, axis=0)
    sorteada = np.argsort(rng.random(varia.shape), axis=1)
    varia_na_ordem = np.take_along_axis(varia, sorteada, axis=1)
    # as que variam primeiro, na ordem do sorteio; as paradas no fim
    posicao = np.argsort(~varia_na_ordem, axis=1, kind="stable")
    primeiras = np.take_along_axis(sorteada, posicao, axis=1)[:, :m]
    usa = np.take_along_axis(varia_na_ordem, posicao, axis=1)[:, :m]
    primeiras = np.where(usa, primeiras, primeiras[:, :1])
    return np.sort(primeiras, axis=1)


def treinar_arvore(f, y, peso=None, min_folha=1, colunas=None, rng=None):
    """Cresce a árvore nível por nível, sem limite de profundidade.

    peso: quantas vezes cada linha conta (0 = fica de fora). A floresta usa isso para repetir
    as linhas sorteadas. min_folha vale para o tamanho com as repetições.
    colunas: quantas colunas cada nó sorteia (None = todas).
    Uma divisão vale quando os dois lados têm pelo menos min_folha e o Gini baixa.
    Empate entre divisões: fica a primeira coluna e o menor corte.
    """
    ncol = f.shape[1]
    m = ncol if colunas is None else int(colunas)
    if not 1 <= m <= ncol:
        raise ValueError(f"colunas precisa ficar entre 1 e {ncol}: {colunas}")
    if m < ncol and rng is None:
        raise ValueError("para sortear colunas, passe o rng")
    peso = np.ones(len(y)) if peso is None else np.asarray(peso, dtype=float)
    s1 = peso * np.asarray(y, dtype=float)
    nb = int(f.max()) + 2
    nos = {"coluna": [-1], "faixa": [-1], "esq": [-1], "dir": [-1], "valor": [0.0], "tamanho": [0.0]}
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
            nos["tamanho"][no] = t_c[i]
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
                for chave_no, v in (("coluna", -1), ("faixa", -1), ("esq", -1), ("dir", -1), ("valor", 0.0), ("tamanho", 0.0)):
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


def folhas(arvore):
    return int((arvore["coluna"] < 0).sum())


# ---------------------------------------------------------------- a floresta

def treinar_floresta(f, y, arvores=100, colunas=4, min_folha=1, sorteio=0):
    """A Random Forest da aula 04.

    Cada árvore sorteia, com reposição, tantas linhas quanto o treino tem e cresce sem limite
    de profundidade, sorteando `colunas` colunas em cada nó. O número `sorteio` fixa todos os
    sorteios: o mesmo número dá a mesma floresta. Retorna as árvores e o peso de cada linha em
    cada árvore (0 = ficou de fora).
    """
    n = len(y)
    sorteios = np.random.SeedSequence(sorteio).spawn(arvores)
    modelos, pesos = [], []
    for s in sorteios:
        rng = np.random.default_rng(s)
        peso = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        modelos.append(treinar_arvore(f, y, peso=peso, min_folha=min_folha, colunas=colunas, rng=rng))
        pesos.append(peso)
    return {"arvores": modelos, "pesos": pesos}


def notas_das_arvores(modelo, f, arvores=None):
    """Uma linha por árvore, uma coluna por linha."""
    lista = modelo["arvores"] if arvores is None else modelo["arvores"][:arvores]
    return np.array([prever(a, f) for a in lista])


def nota_floresta(modelo, f):
    return notas_das_arvores(modelo, f).mean(axis=0)



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


