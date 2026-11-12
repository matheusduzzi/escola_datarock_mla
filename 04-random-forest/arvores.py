"""As árvores, a floresta e as medidas da aula «Random Forest».

Só pandas e NumPy. A qualidade sai igual toda vez que você roda. O tempo muda de máquina para
máquina, e de uma rodada para outra na mesma máquina.

ler(caminho): lê as entregas e tira maio e junho de 2026 logo na leitura. Esses dois meses são
    o teste e ficam fechados. O rótulo `atrasou` é 1 quando a situação não é `no_prazo`
    (atraso curto, atraso longo e extravio entram juntos).
separar(base): julho de 2025 a fevereiro de 2026 é o treino; março e abril de 2026 medem.
    As categorias e as faixas de corte saem só do treino. É a mesma divisão da aula 03.

As árvores trabalham com faixas, como na aula 03: cada coluna tem até 32 cortes possíveis,
tirados do treino. A pergunta de um nó é sempre "a coluna passa do corte?".

treinar_arvore(f, y, peso, min_folha, colunas, rng): árvore de classificação pelo Gini, sem
    limite de profundidade. Com `colunas`, cada nó sorteia essa quantidade de colunas e só
    procura a divisão entre elas. Coluna com um valor só dentro do nó não conta no sorteio.
treinar_floresta(f, y, arvores, colunas, amostra, min_folha, sorteio): a Random Forest. Cada
    árvore cresce numa amostra sorteada com reposição e sorteia as colunas em cada nó. O
    número `sorteio` fixa todos os sorteios: o mesmo número dá a mesma floresta.
nota_floresta(modelo, f, arvores): a média das notas das primeiras `arvores` árvores.
nota_fora_da_amostra(modelo, f_treino, arvores): para cada entrega do treino, a média só das
    árvores que não a viram.
rastro(caminho_dados, comando): o que é preciso para refazer o benchmark: o hash do arquivo
    de dados, as versões e a máquina.
"""
import hashlib
import os
import platform
import sys
import time

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
    """Retorna os nomes das colunas e, para treino e validação, (x, y, id da entrega)."""
    treino = base[base["data_despacho"] <= pd.Timestamp(FIM_TREINO)]
    validacao = base[base["data_despacho"] > pd.Timestamp(FIM_TREINO)]
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
    nomes = NUMERICAS + [f"{c}={v}" for c in CATEGORIAS for v in categorias[c]]

    def matriz(p):
        x = p[NUMERICAS].astype(float).copy()
        for c in CATEGORIAS:
            for v in categorias[c]:
                x[f"{c}={v}"] = (p[c] == v).astype(float)
        return x.to_numpy(dtype=float), p["atrasou"].to_numpy(dtype=int), p["id_entrega"].to_numpy()

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

    peso: quantas vezes cada entrega conta (0 = fica de fora). A floresta usa isso para repetir
    as entregas sorteadas. min_folha vale para o tamanho com as repetições.
    colunas: quantas colunas cada nó sorteia (None = todas, como no bagging da aula 03).
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

def treinar_floresta(f, y, arvores=100, colunas=4, amostra=1.0, min_folha=1, sorteio=0):
    """A Random Forest.

    Cada árvore sorteia, com reposição, round(amostra × n) entregas do treino e cresce sem
    limite de profundidade, sorteando `colunas` colunas em cada nó. colunas=None usa todas as
    colunas em todos os nós: é o bagging da aula 03.
    Retorna as árvores, o peso de cada entrega em cada árvore (0 = ficou de fora) e o tempo,
    em segundos, que cada árvore levou para crescer. O tempo é o do processador
    (time.process_time): ele não conta o tempo em que a máquina estava ocupada com outro programa.
    """
    n = len(y)
    tamanho = int(round(amostra * n))
    if not 1 <= tamanho <= n:
        raise ValueError(f"amostra precisa dar entre 1 e {n} entregas: {amostra}")
    sorteios = np.random.SeedSequence(sorteio).spawn(arvores)
    modelos, pesos, tempos = [], [], []
    for s in sorteios:
        rng = np.random.default_rng(s)
        inicio = time.process_time()
        peso = np.bincount(rng.integers(0, n, tamanho), minlength=n).astype(float)
        modelos.append(treinar_arvore(f, y, peso=peso, min_folha=min_folha, colunas=colunas, rng=rng))
        tempos.append(time.process_time() - inicio)
        pesos.append(peso)
    return {"arvores": modelos, "pesos": pesos, "tempos": tempos}


def notas_das_arvores(modelo, f, arvores=None):
    """Uma linha por árvore, uma coluna por entrega."""
    lista = modelo["arvores"] if arvores is None else modelo["arvores"][:arvores]
    return np.array([prever(a, f) for a in lista])


def nota_floresta(modelo, f, arvores=None):
    return notas_das_arvores(modelo, f, arvores).mean(axis=0)


def nota_fora_da_amostra(modelo, f_treino, arvores=None):
    """A média das árvores que não viram a entrega. NaN quando todas as árvores a viram."""
    k = len(modelo["arvores"]) if arvores is None else arvores
    notas = notas_das_arvores(modelo, f_treino, k)
    fora = np.array(modelo["pesos"][:k]) == 0
    soma = (notas * fora).sum(axis=0)
    quantas = fora.sum(axis=0)
    with np.errstate(all="ignore"):
        return np.where(quantas > 0, soma / np.maximum(quantas, 1), np.nan)


def rastro(caminho_dados, comando):
    """O que acompanha o benchmark: sem isso, ninguém refaz a conta nem compara o tempo."""
    return {
        "comando": comando,
        "dados": os.path.basename(str(caminho_dados)),
        "dados_sha256": hashlib.sha256(open(caminho_dados, "rb").read()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sistema": f"{platform.system()} {platform.release()}",
        "processador": platform.machine(),
        "nucleos": os.cpu_count(),
        "threads_usadas": 1,
        "carga_da_maquina": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
        "relogio": "time.process_time: tempo de processador do Python, só o crescimento das árvores",
    }


# ---------------------------------------------------------------- as referências e as medidas

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


def acerto_do_par(nota, y):
    """Sorteando uma que atrasou e uma no prazo, em quantos pares a que atrasou tem a nota maior."""
    nota = np.asarray(nota, dtype=float)
    y = np.asarray(y)
    ok = ~np.isnan(nota)
    nota, y = nota[ok], y[ok]
    posto = pd.Series(nota).rank().to_numpy()
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
