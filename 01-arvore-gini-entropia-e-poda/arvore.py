"""A árvore, o Gini, a entropia e a poda da aula «Árvore, Gini, entropia e poda».

Só pandas e NumPy. Tudo sai igual toda vez que você roda.

ler(caminho): lê as entregas e tira maio e junho de 2026 logo na leitura. Esses dois meses são
    o teste do curso e ficam fechados. O rótulo `atrasou` é 1 quando a situação não é `no_prazo`
    (atraso curto, atraso longo e extravio entram juntos).
separar(base): julho de 2025 a fevereiro de 2026 é o treino; março e abril de 2026 medem.
    As categorias saem só do treino.

gini(atrasou, total): 2 x p x (1 - p), com p = atrasou / total. É o mesmo que 1 - p² - (1 - p)².
entropia(atrasou, total): -p log2(p) - (1 - p) log2(1 - p). Vale 0 num nó puro.
depois(medida, n_sim, a_sim, n_nao, a_nao): a média da medida dos dois lados, pesada pelo
    tamanho de cada lado.
melhor_corte(x, y, medida, min_folha): para uma coluna, o corte que mais baixa a medida.
    O corte fica no meio de dois valores vizinhos que existem no treino.
treinar_arvore(x, y, medida, profundidade, min_folha): em cada nó, testa todas as colunas e todos
    os cortes e fica com a divisão que mais baixa a medida. Empate: a primeira coluna e o menor
    corte. Para quando o nó é puro, quando chega na profundidade ou quando não sobra divisão.
caminho_de_poda(arvore, total): a poda por custo-complexidade. Em cada passo, corta o galho que
    menos baixa o Gini por folha (o elo mais fraco) e anota o alfa em que ele cai.
podar(arvore, alfa, total): a árvore que sobra com esse alfa.
nota(arvore, x): a parte que atrasou na folha de cada entrega, no treino.
acerto_do_par(nota, y) e brier(nota, y): as medidas do curso ML supervisionado.
"""
import copy

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
    """Retorna os nomes das colunas e, para treino e validação, (x, y, tabela)."""
    treino = base[base["data_despacho"] <= pd.Timestamp(FIM_TREINO)]
    validacao = base[base["data_despacho"] > pd.Timestamp(FIM_TREINO)]
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
    nomes = NUMERICAS + [f"{c}={v}" for c in CATEGORIAS for v in categorias[c]]

    def matriz(p):
        x = p[NUMERICAS].astype(float).copy()
        for c in CATEGORIAS:
            for v in categorias[c]:
                x[f"{c}={v}"] = (p[c] == v).astype(float)
        return x.to_numpy(dtype=float), p["atrasou"].to_numpy(dtype=int), p.reset_index(drop=True)

    return nomes, {"treino": matriz(treino), "validacao": matriz(validacao)}


# ---------------------------------------------------------------- as duas medidas

def gini(atrasou, total):
    atrasou, total = np.asarray(atrasou, dtype=float), np.asarray(total, dtype=float)
    with np.errstate(all="ignore"):
        p = np.where(total > 0, atrasou / np.where(total > 0, total, 1), 0.0)
    return 2 * p * (1 - p)


def entropia(atrasou, total):
    atrasou, total = np.asarray(atrasou, dtype=float), np.asarray(total, dtype=float)
    with np.errstate(all="ignore"):
        p = np.where(total > 0, atrasou / np.where(total > 0, total, 1), 0.0)
        h = -(p * np.log2(np.where(p > 0, p, 1)) + (1 - p) * np.log2(np.where(p < 1, 1 - p, 1)))
    return np.where((p > 0) & (p < 1), h, 0.0)


MEDIDAS = {"gini": gini, "entropia": entropia}


def depois(medida, n_sim, a_sim, n_nao, a_nao):
    f = MEDIDAS[medida]
    n = n_sim + n_nao
    return (n_sim * f(a_sim, n_sim) + n_nao * f(a_nao, n_nao)) / n


# ---------------------------------------------------------------- a árvore

def melhor_corte(xj, y, medida="gini", min_folha=1):
    """(corte, valor da medida depois) da coluna xj, ou None se nenhum corte separa."""
    n, total = len(y), int(y.sum())
    ordem = np.argsort(xj, kind="mergesort")
    xs, ys = xj[ordem], y[ordem]
    esq_n = np.arange(1, n)
    esq_a = np.cumsum(ys)[:-1]
    pode = (xs[1:] != xs[:-1]) & (esq_n >= min_folha) & (n - esq_n >= min_folha)
    if not pode.any():
        return None
    filhos = depois(medida, esq_n, esq_a, n - esq_n, total - esq_a)
    filhos = np.where(pode, filhos, np.inf)
    k = int(np.argmax(filhos <= filhos.min() + 1e-12))       # empate: o menor corte
    return float((xs[k] + xs[k + 1]) / 2), float(filhos[k])


def candidatas(x, y, nomes, medida="gini", min_folha=1):
    """Uma linha por coluna: o melhor corte dela e os números de cada lado."""
    n, total = len(y), int(y.sum())
    antes = float(MEDIDAS[medida](total, n))
    linhas = []
    for j, nome in enumerate(nomes):
        r = melhor_corte(x[:, j], y, medida, min_folha)
        if r is None:
            continue
        corte, valor = r
        lado = x[:, j] > corte
        linhas.append({"coluna": nome, "corte": corte, "n_sim": int(lado.sum()), "a_sim": int(y[lado].sum()),
                       "n_nao": int((~lado).sum()), "a_nao": int(y[~lado].sum()),
                       "depois": valor, "queda": antes - valor})
    return sorted(linhas, key=lambda d: -d["queda"])


def treinar_arvore(x, y, medida="gini", profundidade=None, min_folha=1):
    """profundidade=None: cresce até cada folha ser pura ou não ter mais divisão."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    f = MEDIDAS[medida]

    def crescer(idx, nivel):
        n, a = len(idx), int(y[idx].sum())
        no = {"entregas": n, "atrasou": a, "nota": a / n}
        if a in (0, n) or (profundidade is not None and nivel >= profundidade) or n < 2 * min_folha:
            return no
        antes = float(f(a, n))
        melhor = None
        for j in range(x.shape[1]):
            r = melhor_corte(x[idx, j], y[idx], medida, min_folha)
            if r is None:
                continue
            queda = antes - r[1]
            if queda > 1e-12 and (melhor is None or queda > melhor[2] + 1e-12):
                melhor = (j, r[0], queda)
        if melhor is None:
            return no
        j, corte, _ = melhor
        vai_esq = x[idx, j] <= corte
        no.update(coluna=j, corte=corte,
                  esq=crescer(idx[vai_esq], nivel + 1), dir=crescer(idx[~vai_esq], nivel + 1))
        return no

    return crescer(np.arange(len(y)), 0)


def nota(arvore, x):
    x = np.asarray(x, dtype=float)
    saida = np.empty(len(x))

    def descer(no, idx):
        if "coluna" not in no:
            saida[idx] = no["nota"]
            return
        esq = x[idx, no["coluna"]] <= no["corte"]
        descer(no["esq"], idx[esq])
        descer(no["dir"], idx[~esq])

    descer(arvore, np.arange(len(x)))
    return saida


def folhas(arvore):
    if "coluna" not in arvore:
        return [arvore]
    return folhas(arvore["esq"]) + folhas(arvore["dir"])


def altura(arvore):
    if "coluna" not in arvore:
        return 0
    return 1 + max(altura(arvore["esq"]), altura(arvore["dir"]))


# ---------------------------------------------------------------- a poda por custo-complexidade
#
# R(nó) = (entregas do nó / entregas do treino) x Gini do nó. Para um galho que começa no nó t,
# R(galho) é a soma de R das folhas dele. Trocar o galho por uma folha aumenta R em
# R(t) - R(galho) e tira (folhas do galho - 1) folhas. O alfa do nó é essa troca por folha:
#     alfa(t) = (R(t) - R(galho)) / (folhas do galho - 1)
# A poda corta primeiro o nó de menor alfa, o elo mais fraco, e repete.

def _r(no, total):
    return no["entregas"] / total * float(gini(no["atrasou"], no["entregas"]))


def _resumo(no, total):
    """(R do galho, número de folhas, menor alfa dentro do galho)."""
    if "coluna" not in no:
        return _r(no, total), 1, np.inf
    re, fe, ae = _resumo(no["esq"], total)
    rd, fd, ad = _resumo(no["dir"], total)
    r_galho, n_folhas = re + rd, fe + fd
    alfa = (_r(no, total) - r_galho) / (n_folhas - 1)
    no["_alfa"] = alfa
    return r_galho, n_folhas, min(alfa, ae, ad)


def _cortar(no, alfa, total):
    if "coluna" not in no:
        return
    if no["_alfa"] <= alfa + 1e-15:
        for k in ("coluna", "corte", "esq", "dir", "_alfa"):
            no.pop(k, None)
        return
    _cortar(no["esq"], alfa, total)
    _cortar(no["dir"], alfa, total)


def _internos(no):
    if "coluna" not in no:
        return []
    return [no] + _internos(no["esq"]) + _internos(no["dir"])


def caminho_de_poda(arvore, total):
    """Lista de (alfa, folhas, R das folhas), da árvore inteira até sobrar só a raiz.

    Também anota em cada nó da `arvore` recebida o alfa em que o galho dele vira folha
    (`cai_em`). A poda de qualquer alfa sai dessa anotação, sem refazer as contas.
    """
    a = copy.deepcopy(arvore)
    originais = {id(c): o for c, o in zip(_internos(a), _internos(arvore))}
    r, n, menor = _resumo(a, total)
    caminho = [(0.0, n, r)]
    while n > 1:
        for no in _internos(a):
            if no["_alfa"] <= menor + 1e-15 and "cai_em" not in originais[id(no)]:
                for o in _internos(originais[id(no)]):     # o galho todo cai junto
                    o.setdefault("cai_em", menor)
        _cortar(a, menor, total)
        r, n, proximo = _resumo(a, total)
        caminho.append((menor, n, r))
        menor = proximo
    return caminho


def podar(arvore, alfa, total=None):
    """A árvore que sobra com esse alfa: todo nó que cai até esse alfa vira folha.

    Precisa de caminho_de_poda(arvore, total) antes, que anota onde cada nó cai.
    """
    def copiar(no):
        if "coluna" not in no or no["cai_em"] <= alfa + 1e-15:
            return {k: no[k] for k in ("entregas", "atrasou", "nota")}
        novo = {k: no[k] for k in ("entregas", "atrasou", "nota", "coluna", "corte", "cai_em")}
        novo["esq"], novo["dir"] = copiar(no["esq"]), copiar(no["dir"])
        return novo
    return copiar(arvore)


def dobras(n, k=5):
    """k pedaços seguidos das linhas 0..n-1 (as entregas estão em ordem de despacho)."""
    return np.array_split(np.arange(n), k)


def alfa_pelas_dobras(x, y, caminho, k=5):
    """Escolhe o alfa só com o treino, pelo Brier fora da dobra.

    Os alfas testados são os do caminho da árvore inteira; entre dois alfas vizinhos, usa a
    média geométrica deles. Para cada pedaço, cresce a
    árvore sem limite nos outros k - 1, poda com cada alfa e mede o Brier no pedaço de fora.
    Retorna (lista de (alfa testado, folhas na árvore inteira, Brier médio), alfa escolhido).
    """
    alfas = [a for a, _, _ in caminho]
    testados = [np.sqrt(alfas[i] * alfas[i + 1]) if alfas[i] > 0 else alfas[i + 1] / 2
                for i in range(len(alfas) - 1)] + [alfas[-1]]
    soma = np.zeros(len(testados))
    for fora in dobras(len(y), k):
        dentro = np.setdiff1d(np.arange(len(y)), fora)
        a = treinar_arvore(x[dentro], y[dentro])
        caminho_de_poda(a, len(dentro))
        for i, alfa in enumerate(testados):
            soma[i] += brier(nota(podar(a, alfa), x[fora]), y[fora]) * len(fora)
    media = soma / len(y)
    tabela = [(alfa, caminho[i][1], float(media[i])) for i, alfa in enumerate(testados)]
    i = int(np.argmin(media))
    return tabela, testados[i]


# ---------------------------------------------------------------- as medidas

def acerto_do_par(nota, y):
    posto = pd.Series(np.asarray(nota, dtype=float)).rank().to_numpy()
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((posto[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(nota, y):
    return float(np.mean((np.asarray(nota, dtype=float) - np.asarray(y)) ** 2))


# ---------------------------------------------------------------- texto em português

def num(x, casas=1):
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"


def corte_txt(x):
    texto = num(x, 2)
    return texto[:-3] if texto.endswith(",00") else texto.rstrip("0")


def pergunta(nome, corte):
    if "=" in nome:
        c, v = nome.split("=")
        return f"{c} é {v}"
    if nome == "vespera_feriado":
        return "véspera de feriado"
    return f"{nome} > {corte_txt(corte)}"


def desenhar(arvore, nomes, margem=""):
    """O texto da árvore: a pergunta de cada nó e, em cada folha, entregas e parte que atrasou."""
    if "coluna" not in arvore:
        n = arvore["entregas"]
        return [f"{margem}folha: {n} entrega{'s' if n != 1 else ''}, {pct(arvore['nota'])} atrasaram"]
    p = pergunta(nomes[arvore["coluna"]], arvore["corte"])
    linhas = []
    for resposta, filho in (("sim", arvore["dir"]), ("não", arvore["esq"])):
        linhas.append(f"{margem}{p}? {resposta}:")
        linhas += desenhar(filho, nomes, margem + "    ")
    return linhas
