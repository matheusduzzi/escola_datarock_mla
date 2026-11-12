"""O experimento da aula «Run, parâmetro, métrica e artefato».

Um boosting de árvores pequenas para a pergunta "essa entrega vai atrasar?", no pandas e no NumPy.
Cada vez que você roda, o programa grava uma run: uma pasta em runs/<nome>/ com o registro do que
entrou e do que saiu.

    python3 experimento.py rodar --nome paula --semente 0
    python3 experimento.py sementes --de 0 --ate 9
    python3 experimento.py comparar runs/paula runs/caio

rodar     treina, mede em março e abril e grava runs/<nome>/:
            run.json             o registro: código, dados, parâmetros, métricas, artefatos e ambiente
            modelo.json          as árvores treinadas (artefato)
            notas_validacao.csv  a nota de cada entrega de março e abril (artefato)
            experimento.py       a cópia do programa que rodou (artefato)
          Uma run gravada não muda: se a pasta já existe, o programa para sem escrever nada.
sementes  roda as mesmas contas com várias sementes, sem gravar, e mostra quanto a métrica varia.
comparar  lê o run.json de duas runs e mostra o que mudou entre elas.

Maio e junho de 2026 são o teste e saem logo na leitura: nenhuma conta usa essas entregas.
`atrasou` é 1 quando a situação não é `no_prazo` (atraso curto, atraso longo e extravio juntos).
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
DADOS_PADRAO = "dados/entregas.csv"
NUMERICAS = ["distancia_km", "peso_kg", "itens", "valor_frete", "prazo_prometido_dias",
             "vespera_feriado", "dia_semana_despacho", "valor_pedido"]
CATEGORIAS = ["transportadora", "regiao"]
MAX_FAIXAS = 64            # cada coluna vira até 64 faixas, com os cortes tirados do treino
LAMBDA = 1.0               # o +1 no denominador do valor de cada folha


# ------------------------------------------------------------------ dados

def sha256_arquivo(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def sha256_linhas(parte):
    return hashlib.sha256(parte.to_csv(index=False).encode("utf-8")).hexdigest()


def ler(caminho, fim_treino, fim_validacao):
    base = pd.read_csv(caminho, parse_dates=["data_despacho"])
    linhas = len(base)
    base = base[base["data_despacho"] <= pd.Timestamp(fim_validacao)].copy()   # maio e junho saem aqui
    treino = base[base["data_despacho"] <= pd.Timestamp(fim_treino)]
    validacao = base[base["data_despacho"] > pd.Timestamp(fim_treino)]
    registro = {"arquivo": str(caminho), "sha256": sha256_arquivo(caminho), "linhas": linhas}
    for nome, parte in (("treino", treino), ("validacao", validacao)):
        registro[nome] = {"de": str(parte["data_despacho"].min().date()),
                          "ate": str(parte["data_despacho"].max().date()),
                          "entregas": len(parte), "sha256": sha256_linhas(parte)}
    return treino, validacao, registro


def matriz(parte, categorias):
    x = parte[NUMERICAS].astype(float).copy()
    for c in CATEGORIAS:
        for v in categorias[c]:
            x[f"{c}={v}"] = (parte[c] == v).astype(float)
    y = (parte["situacao"] != "no_prazo").astype(int).to_numpy()
    return x.to_numpy(dtype=float), y, list(x.columns)


# ------------------------------------------------------------------ boosting

def cortes_do_treino(x):
    cortes = []
    for j in range(x.shape[1]):
        q = np.unique(np.quantile(x[:, j], np.linspace(0, 1, MAX_FAIXAS + 1)[1:-1]))
        cortes.append(q[q < x[:, j].max()])
    return cortes


def faixas(x, cortes):
    return np.column_stack([np.searchsorted(c, x[:, j], side="left") for j, c in enumerate(cortes)])


def arvore(fx, g, h, idx, cortes, profundidade, min_folha):
    """Uma árvore pelo ganho de gradiente; cada nó pergunta "a coluna passa do corte?"."""
    G, H = g[idx].sum(), h[idx].sum()
    folha = {"valor": float(G / (H + LAMBDA)), "entregas": int(len(idx))}
    if profundidade == 0 or len(idx) < 2 * min_folha:
        return folha
    melhor = None
    pai = G * G / (H + LAMBDA)
    for j, c in enumerate(cortes):
        if len(c) == 0:
            continue
        b = fx[idx, j]
        gl = np.cumsum(np.bincount(b, weights=g[idx], minlength=len(c) + 1))[:-1]
        hl = np.cumsum(np.bincount(b, weights=h[idx], minlength=len(c) + 1))[:-1]
        nl = np.cumsum(np.bincount(b, minlength=len(c) + 1))[:-1]
        ok = (nl >= min_folha) & (len(idx) - nl >= min_folha)
        if not ok.any():
            continue
        ganho = gl ** 2 / (hl + LAMBDA) + (G - gl) ** 2 / (H - hl + LAMBDA) - pai
        ganho[~ok] = -np.inf
        k = int(np.argmax(ganho))
        if ganho[k] > 0 and (melhor is None or ganho[k] > melhor[0]):
            melhor = (ganho[k], j, k)
    if melhor is None:
        return folha
    _, j, k = melhor
    esquerda = idx[fx[idx, j] <= k]
    direita = idx[fx[idx, j] > k]
    return {"coluna": j, "corte": float(cortes[j][k]),
            "sim": arvore(fx, g, h, direita, cortes, profundidade - 1, min_folha),
            "nao": arvore(fx, g, h, esquerda, cortes, profundidade - 1, min_folha)}


def prever_arvore(no, x):
    if "valor" in no:
        return np.full(len(x), no["valor"])
    passa = x[:, no["coluna"]] > no["corte"]
    saida = np.empty(len(x))
    saida[passa] = prever_arvore(no["sim"], x[passa])
    saida[~passa] = prever_arvore(no["nao"], x[~passa])
    return saida


def treinar(x, y, p):
    rng = np.random.default_rng(p["semente"])
    cortes = cortes_do_treino(x)
    fx = faixas(x, cortes)
    base = float(np.log(y.mean() / (1 - y.mean())))
    soma = np.full(len(y), base)
    arvores = []
    k = int(round(p["subamostra"] * len(y)))
    for _ in range(p["rodadas"]):
        nota = 1 / (1 + np.exp(-soma))
        g, h = y - nota, nota * (1 - nota)
        idx = np.sort(rng.choice(len(y), size=k, replace=False))
        a = arvore(fx, g, h, idx, cortes, p["profundidade"], p["min_folha"])
        arvores.append(a)
        soma += p["passo"] * prever_arvore(a, x)
    return {"base": base, "passo": p["passo"], "arvores": arvores}


def prever(modelo, x):
    soma = np.full(len(x), modelo["base"])
    for a in modelo["arvores"]:
        soma += modelo["passo"] * prever_arvore(a, x)
    return 1 / (1 + np.exp(-soma))


def acerto_do_par(nota, y):
    """Sorteando uma entrega que atrasou e uma no prazo, em quantos pares a que atrasou tem nota maior."""
    r = pd.Series(nota).rank().to_numpy()
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def experimento(p):
    treino, validacao, dados = ler(p["dados"], p["fim_treino"], p["fim_validacao"])
    categorias = {c: sorted(treino[c].unique()) for c in CATEGORIAS}
    xt, yt, colunas = matriz(treino, categorias)
    xv, yv, _ = matriz(validacao, categorias)
    modelo = treinar(xt, yt, p)
    modelo["colunas"] = colunas
    nv = prever(modelo, xv)
    metricas = {"acerto_do_par_validacao": round(acerto_do_par(nv, yv), 6),
                "brier_validacao": round(float(np.mean((nv - yv) ** 2)), 6),
                "acerto_do_par_treino": round(acerto_do_par(prever(modelo, xt), yt), 6),
                "atrasou_validacao": int(yv.sum())}
    notas = pd.DataFrame({"id_entrega": validacao["id_entrega"].to_numpy(), "nota": np.round(nv, 6)})
    return dados, metricas, modelo, notas


# ------------------------------------------------------------------ comandos

def parametros(a):
    return {"dados": a.dados, "fim_treino": a.fim_treino, "fim_validacao": a.fim_validacao,
            "rodadas": a.rodadas, "profundidade": a.profundidade, "passo": a.passo,
            "subamostra": a.subamostra, "min_folha": a.min_folha, "semente": getattr(a, "semente", None)}


def pct(v):
    return f"{100 * v:.2f}%".replace(".", ",")


def num(v, casas=4):
    return f"{v:.{casas}f}".replace(".", ",")


def rodar(a):
    pasta = AQUI / "runs" / a.nome
    if pasta.exists():
        print(f"a run '{a.nome}' já existe em runs/{a.nome}/. Uma run gravada não muda: escolha outro nome.")
        return 1
    p = parametros(a)
    inicio = time.time()
    dados, metricas, modelo, notas = experimento(p)
    duracao = time.time() - inicio
    pasta.mkdir(parents=True)
    (pasta / "modelo.json").write_text(json.dumps(modelo, indent=1) + "\n", encoding="utf-8")
    notas.to_csv(pasta / "notas_validacao.csv", index=False)
    shutil.copyfile(Path(__file__), pasta / "experimento.py")
    registro = {
        "run": {"nome": a.nome, "comando": "python3 experimento.py " + " ".join(sys.argv[1:]),
                "inicio": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(inicio)),
                "duracao_s": round(duracao, 1)},
        "codigo": {"arquivo": "experimento.py", "sha256": sha256_arquivo(Path(__file__))},
        "dados": dados,
        "parametros": {k: v for k, v in p.items() if k != "dados"},
        "metricas": metricas,
        "artefatos": [{"arquivo": n, "sha256": sha256_arquivo(pasta / n)}
                      for n in ("modelo.json", "notas_validacao.csv", "experimento.py")],
        "ambiente": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sistema": f"{platform.system()} {platform.machine()}"},
    }
    (pasta / "run.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"run '{a.nome}' gravada em runs/{a.nome}/")
    print(f"dados: {dados['arquivo']}, {dados['linhas']} linhas, sha256 {dados['sha256'][:12]}")
    for parte, rotulo in (("treino", "treino"), ("validacao", "validação")):
        d = dados[parte]
        print(f"   {rotulo}: {d['de']} a {d['ate']}, {d['entregas']} entregas, sha256 {d['sha256'][:12]}")
    print("parâmetros: " + ", ".join(f"{k}={v}" for k, v in registro["parametros"].items()))
    print(f"acerto do par: treino {pct(metricas['acerto_do_par_treino'])}, "
          f"validação {pct(metricas['acerto_do_par_validacao'])}; Brier na validação {num(metricas['brier_validacao'])}")
    print(f"entregas que atrasaram na validação: {metricas['atrasou_validacao']}")
    for art in registro["artefatos"]:
        print(f"artefato {art['arquivo']}: sha256 {art['sha256'][:12]}")
    return 0


def sementes(a):
    p = parametros(a)
    valores = []
    print(f"dados: {a.dados}; rodadas={a.rodadas}, passo={a.passo}, subamostra={a.subamostra}")
    print("semente  validação   Brier")
    for s in range(a.de, a.ate + 1):
        p["semente"] = s
        _, m, _, _ = experimento(p)
        valores.append(m["acerto_do_par_validacao"])
        print(f"{s:>7}  {pct(m['acerto_do_par_validacao']):>9}  {num(m['brier_validacao'])}")
    v = np.array(valores)
    print(f"{len(v)} sementes: de {pct(v.min())} a {pct(v.max())}, média {pct(v.mean())}, "
          f"diferença entre a maior e a menor {num(100 * (v.max() - v.min()), 2)} ponto")
    return 0


def achatar(d, prefixo=""):
    saida = {}
    for k, v in d.items():
        chave = f"{prefixo}{k}"
        if isinstance(v, dict):
            saida.update(achatar(v, chave + "."))
        else:
            saida[chave] = v
    return saida


def comparar(a):
    registros = []
    for pasta in (a.run_a, a.run_b):
        arq = Path(pasta) / "run.json"
        if not arq.is_file():
            print(f"não achei {arq}")
            return 1
        r = json.loads(arq.read_text(encoding="utf-8"))
        r["artefatos"] = {x["arquivo"]: x["sha256"] for x in r["artefatos"]}
        registros.append(achatar(r))
    ra, rb = registros
    na, nb = Path(a.run_a).name, Path(a.run_b).name
    ignorar = ("run.",)
    diferentes = [k for k in sorted(set(ra) | set(rb))
                  if not k.startswith(ignorar) and ra.get(k) != rb.get(k)]
    iguais = sum(1 for k in set(ra) & set(rb) if not k.startswith(ignorar) and ra[k] == rb[k])
    print(f"comparando {na} e {nb}: {iguais} campos iguais, {len(diferentes)} diferentes")
    if not diferentes:
        print("(fora nome, comando, hora e duração, os dois registros são iguais)")
        return 0
    largura = max([len(k) for k in diferentes] + [5])
    print(f"{'campo':<{largura}}  {na:<20}  {nb}")
    for k in diferentes:
        va, vb = ra.get(k, "(não tem)"), rb.get(k, "(não tem)")
        if isinstance(va, str) and len(va) == 64:
            va = va[:12]
        if isinstance(vb, str) and len(vb) == 64:
            vb = vb[:12]
        print(f"{k:<{largura}}  {str(va):<20}  {vb}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="O experimento da aula: rodar, sementes, comparar.")
    sub = p.add_subparsers(dest="comando", required=True)
    for nome in ("rodar", "sementes"):
        s = sub.add_parser(nome)
        s.add_argument("--dados", default=DADOS_PADRAO)
        s.add_argument("--fim-treino", default="2026-02-28")
        s.add_argument("--fim-validacao", default="2026-04-30")
        s.add_argument("--rodadas", type=int, default=30)
        s.add_argument("--profundidade", type=int, default=3)
        s.add_argument("--passo", type=float, default=0.1)
        s.add_argument("--subamostra", type=float, default=0.5)
        s.add_argument("--min-folha", type=int, default=20)
        if nome == "rodar":
            s.add_argument("--nome", required=True)
            s.add_argument("--semente", type=int, required=True)
        else:
            s.add_argument("--de", type=int, default=0)
            s.add_argument("--ate", type=int, default=9)
    c = sub.add_parser("comparar")
    c.add_argument("run_a")
    c.add_argument("run_b")
    a = p.parse_args(argv)
    return {"rodar": rodar, "sementes": sementes, "comparar": comparar}[a.comando](a)


if __name__ == "__main__":
    sys.exit(main())
