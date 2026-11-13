"""Busca de hiperparâmetros do boosting da lista da manhã: grade, busca aleatória e otimização bayesiana.

Na pasta da aula «Busca de hiperparâmetros» do kit, rode por exemplo:

    python3 busca.py atual
    python3 busca.py custo --espaco espaco.json --amostras 10
    python3 busca.py grade --espaco grade_caio.json --nome grade-caio
    python3 busca.py aleatoria --espaco espaco.json --orcamento 27 --semente 0
    python3 busca.py bayesiana --espaco espaco.json --orcamento 27 --semente 0
    python3 busca.py repetir --metodo aleatoria --espaco espaco.json --orcamento 27
    python3 busca.py meses buscas/grade-caio.csv
    python3 busca.py ruido buscas/grade-caio.csv

Cada avaliação treina o HistGradientBoostingClassifier do scikit-learn nas entregas de julho de 2025
a fevereiro de 2026 e mede o acerto do par (a área sob a curva ROC) em março e abril de 2026. Maio e
junho, o teste, saem na leitura e não entram em conta nenhuma. A planilha é a corrigida da aula 06.

Cada busca grava uma linha por avaliação em buscas/<nome>.csv. Uma busca gravada não é regravada.
"""
import os

# Uma thread só: o tempo de cada avaliação fica comparável entre as buscas e entre máquinas parecidas.
# Para usar todos os núcleos, rode com OMP_NUM_THREADS definido antes, por exemplo OMP_NUM_THREADS=4.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

AQUI = Path(__file__).resolve().parent
DADOS_PADRAO = "../06-run-parametro-metrica-e-artefato/dados/entregas.csv"
FIM_TREINO = "2026-02-28"
FIM_VALIDACAO = "2026-04-30"
NUMERICAS = ["distancia_km", "peso_kg", "itens", "valor_frete", "prazo_prometido_dias",
             "vespera_feriado", "dia_semana_despacho", "valor_pedido"]
CATEGORIAS = {"transportadora": ["Economica", "Propria", "Rapida"],
              "regiao": ["Centro-Oeste", "Nordeste", "Norte", "Sudeste", "Sul"]}
NOMES = ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]
# A configuração da lista de hoje, na linguagem do scikit-learn: passo 0,1, 30 rodadas,
# árvores de até 8 folhas (profundidade 3), folha mínima de 20 entregas, sem regularização.
ATUAL = {"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 8, "min_samples_leaf": 20,
         "l2_regularization": 0.0}
CANDIDATOS = 4000          # pontos sorteados a cada passo da bayesiana para escolher o próximo


# ------------------------------------------------------------------ dados e avaliação

def ler(caminho):
    base = pd.read_csv(caminho, parse_dates=["data_despacho"])
    base = base[base["data_despacho"] <= pd.Timestamp(FIM_VALIDACAO)]          # maio e junho saem aqui
    treino = base[base["data_despacho"] <= pd.Timestamp(FIM_TREINO)]
    validacao = base[base["data_despacho"] > pd.Timestamp(FIM_TREINO)]
    return treino, validacao


def matriz(parte):
    x = parte[NUMERICAS].astype(float).copy()
    for c, valores in CATEGORIAS.items():
        for v in valores:
            x[f"{c}={v}"] = (parte[c] == v).astype(float)
    y = (parte["situacao"] != "no_prazo").astype(int).to_numpy()   # atraso curto, longo e extravio
    return x.to_numpy(dtype=float), y


class Avaliador:
    def __init__(self, caminho):
        caminho = Path(caminho)
        if not caminho.is_file():
            sys.exit(f"não achei {caminho}. A planilha corrigida fica na pasta da aula 06, "
                     "06-run-parametro-metrica-e-artefato/, ao lado desta. Baixe o kit inteiro.")
        self.caminho = caminho
        treino, validacao = ler(caminho)
        self.xt, self.yt = matriz(treino)
        self.xv, self.yv = matriz(validacao)
        self.marco = (validacao["data_despacho"].dt.month == 3).to_numpy()
        self.n_treino, self.n_validacao = len(treino), len(validacao)

    def modelo(self, config):
        return HistGradientBoostingClassifier(early_stopping=False, random_state=0, **config)

    def notas(self, config):
        return self.modelo(config).fit(self.xt, self.yt).predict_proba(self.xv)[:, 1]

    def avaliar(self, config):
        t0 = time.process_time()
        nota = self.notas(config)
        segundos = time.process_time() - t0
        return {"auc": roc_auc_score(self.yv, nota),
                "auc_marco": roc_auc_score(self.yv[self.marco], nota[self.marco]),
                "auc_abril": roc_auc_score(self.yv[~self.marco], nota[~self.marco]),
                "segundos": segundos}


# ------------------------------------------------------------------ espaço de busca

def ler_espaco(caminho):
    espaco = json.loads(Path(caminho).read_text(encoding="utf-8"))
    for nome in espaco:
        if nome not in NOMES:
            sys.exit(f"{caminho}: '{nome}' não é um hiperparâmetro desse boosting. Use: {', '.join(NOMES)}")
    for nome in NOMES:
        regra = espaco.setdefault(nome, {"fixo": ATUAL[nome]})
        if not any(k in regra for k in ("fixo", "valores", "de")):
            sys.exit(f"{caminho}: '{nome}' precisa de 'fixo', 'valores' ou 'de' e 'ate'")
        if "de" in regra:
            if regra.get("escala", "linear") not in ("linear", "log"):
                sys.exit(f"{caminho}: a escala de '{nome}' é 'linear' ou 'log'")
            if regra.get("escala") == "log" and regra["de"] <= 0:
                sys.exit(f"{caminho}: '{nome}' em escala log precisa começar acima de zero")
            if regra["ate"] <= regra["de"]:
                sys.exit(f"{caminho}: em '{nome}', 'ate' precisa ser maior que 'de'")
    return espaco


def faixas(espaco):
    return [n for n in NOMES if "de" in espaco[n]]


def do_cubo(espaco, u):
    """Um ponto do cubo [0, 1]^d vira uma configuração (u com uma posição por hiperparâmetro com faixa)."""
    config = {}
    livres = iter(u)
    for nome in NOMES:
        regra = espaco[nome]
        if "fixo" in regra:
            config[nome] = regra["fixo"]
        elif "valores" in regra:
            sys.exit(f"'{nome}' tem uma lista de valores: isso é para a grade. Na busca aleatória "
                     "e na bayesiana, use 'de' e 'ate'.")
        else:
            t = next(livres)
            if regra.get("escala") == "log":
                v = math.exp(math.log(regra["de"]) + t * (math.log(regra["ate"]) - math.log(regra["de"])))
            else:
                v = regra["de"] + t * (regra["ate"] - regra["de"])
            config[nome] = int(round(v)) if regra.get("inteiro") else float(v)
    return config


def no_cubo(espaco, config):
    u = []
    for nome in faixas(espaco):
        regra, v = espaco[nome], config[nome]
        if regra.get("escala") == "log":
            u.append((math.log(v) - math.log(regra["de"])) / (math.log(regra["ate"]) - math.log(regra["de"])))
        else:
            u.append((v - regra["de"]) / (regra["ate"] - regra["de"]))
    return u


def combinacoes(espaco):
    listas = []
    for nome in NOMES:
        regra = espaco[nome]
        if "de" in regra:
            sys.exit(f"'{nome}' tem uma faixa: a grade precisa de uma lista de 'valores'.")
        listas.append(regra["valores"] if "valores" in regra else [regra["fixo"]])
    return [dict(zip(NOMES, c)) for c in itertools.product(*listas)]


def chave(config):
    return tuple(round(float(config[n]), 10) for n in NOMES)


# ------------------------------------------------------------------ os três métodos

def busca_grade(av, espaco):
    for config in combinacoes(espaco):
        yield config, av.avaliar(config)


def busca_aleatoria(av, espaco, orcamento, semente):
    rng = np.random.default_rng(semente)
    d = len(faixas(espaco))
    vistas = set()
    feitas = 0
    while feitas < orcamento:
        config = do_cubo(espaco, rng.random(d))
        if chave(config) in vistas:          # o arredondamento dos inteiros pode repetir um ponto
            continue
        vistas.add(chave(config))
        feitas += 1
        yield config, av.avaliar(config)


def melhoria_esperada(media, desvio, melhor):
    desvio = np.maximum(desvio, 1e-9)
    z = (media - melhor) / desvio
    return (media - melhor) * norm.cdf(z) + desvio * norm.pdf(z)


def busca_bayesiana(av, espaco, orcamento, semente, iniciais):
    """Processo gaussiano como modelo do acerto e melhoria esperada para escolher o próximo ponto."""
    rng = np.random.default_rng(semente)
    d = len(faixas(espaco))
    pontos, acertos, vistas = [], [], set()
    while len(pontos) < orcamento:
        if len(pontos) < iniciais:
            config = do_cubo(espaco, rng.random(d))
            if chave(config) in vistas:
                continue
        else:
            nucleo = (ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=[0.3] * d, length_scale_bounds=(1e-2, 1e1), nu=2.5)
                      + WhiteKernel(1e-2, (1e-6, 1e0)))
            gp = GaussianProcessRegressor(kernel=nucleo, normalize_y=True, n_restarts_optimizer=2,
                                          random_state=int(rng.integers(1_000_000)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(np.array(pontos), np.array(acertos))
            cand = rng.random((CANDIDATOS, d))
            media, desvio = gp.predict(cand, return_std=True)
            ordem = np.argsort(-melhoria_esperada(media, desvio, max(acertos)))
            for i in ordem:                        # o melhor candidato que ainda não foi avaliado
                config = do_cubo(espaco, cand[i])
                if chave(config) not in vistas:
                    break
        vistas.add(chave(config))
        r = av.avaliar(config)
        pontos.append(no_cubo(espaco, config))   # o ponto arredondado, o que foi de fato avaliado
        acertos.append(100 * r["auc"])
        yield config, r


# ------------------------------------------------------------------ saída

def pct(x, casas=2):
    return f"{100 * x:.{casas}f}%".replace(".", ",")


def num(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def duracao(segundos):
    if segundos < 120:
        return f"{num(segundos, 1)} s"
    return f"{num(segundos / 60, 1)} minutos"


def texto_config(c):
    lr = f"{c['learning_rate']:.3f}".rstrip("0").rstrip(".").replace(".", ",")
    l2 = f"{c['l2_regularization']:.3g}".replace(".", ",")
    return (f"passo {lr:<6} rodadas {c['max_iter']:>4}  folhas {c['max_leaf_nodes']:>3}  "
            f"folha mín. {c['min_samples_leaf']:>4}  L2 {l2}")


def gravar(nome, linhas):
    pasta = AQUI / "buscas"
    pasta.mkdir(exist_ok=True)
    arquivo = pasta / f"{nome}.csv"
    with open(arquivo, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["avaliacao"] + NOMES + ["auc", "auc_marco", "auc_abril", "segundos"])
        w.writeheader()
        for i, (c, r) in enumerate(linhas, 1):
            w.writerow({"avaliacao": i, **c, "auc": round(r["auc"], 6), "auc_marco": round(r["auc_marco"], 6),
                        "auc_abril": round(r["auc_abril"], 6), "segundos": round(r["segundos"], 3)})
    return arquivo


def checar_nome(nome):
    if (AQUI / "buscas" / f"{nome}.csv").exists():
        sys.exit(f"a busca '{nome}' já existe em buscas/{nome}.csv. Uma busca gravada não muda: "
                 "escolha outro nome.")


def rodar_busca(av, gerador, nome, titulo, orcamento=None):
    checar_nome(nome)
    atual = av.avaliar(ATUAL)
    print(titulo)
    print(f"treino: {av.n_treino} entregas até fevereiro; validação: {av.n_validacao} entregas de março e abril")
    print(f"configuração de hoje: acerto do par {pct(atual['auc'])}")
    linhas, melhor, t0 = [], -1.0, time.process_time()
    marcos = {5, 10, 15, 20, 27, 40, 50, 81, 100, 200, 432}
    for config, r in gerador:
        linhas.append((config, r))
        melhor = max(melhor, r["auc"])
        if len(linhas) in marcos and (orcamento is None or len(linhas) < orcamento):
            print(f"   depois de {len(linhas):>3} avaliações, a melhor: {pct(melhor)}")
    total = time.process_time() - t0
    arquivo = gravar(nome, linhas)
    ordem = sorted(linhas, key=lambda cr: -cr[1]["auc"])
    print(f"{len(linhas)} avaliações em {duracao(total)} de processador ({num(total / len(linhas), 2)} s por avaliação)")
    print("as 3 melhores:")
    for c, r in ordem[:3]:
        print(f"   {pct(r['auc'])}  {texto_config(c)}")
    c, r = ordem[0]
    dif = 100 * (r['auc'] - atual['auc'])
    print(f"melhor: {pct(r['auc'])}, {num(abs(dif))} ponto {'acima' if dif >= 0 else 'abaixo'} da configuração de hoje")
    print(f"gravada em buscas/{arquivo.name}")


# ------------------------------------------------------------------ comandos

def cmd_atual(a):
    av = Avaliador(a.dados)
    r = av.avaliar(ATUAL)
    print(f"configuração de hoje: {texto_config(ATUAL)}")
    print(f"acerto do par em março e abril: {pct(r['auc'])} (março {pct(r['auc_marco'])}, abril {pct(r['auc_abril'])})")
    print(f"tempo de uma avaliação: {num(r['segundos'], 2)} s de processador")


def cmd_custo(a):
    av = Avaliador(a.dados)
    espaco = ler_espaco(a.espaco)
    rng = np.random.default_rng(a.semente)
    if faixas(espaco):
        configs = [do_cubo(espaco, rng.random(len(faixas(espaco)))) for _ in range(a.amostras)]
        total_grade = None
    else:
        todas = combinacoes(espaco)
        total_grade = len(todas)
        configs = [todas[i] for i in rng.choice(len(todas), size=min(a.amostras, len(todas)), replace=False)]
    segundos = []
    print(f"{len(configs)} configurações sorteadas de {a.espaco}, segundos de processador numa thread:")
    for c in configs:
        r = av.avaliar(c)
        segundos.append(r["segundos"])
        print(f"   {num(r['segundos'], 2):>6} s  {texto_config(c)}")
    media = float(np.mean(segundos))
    print(f"média {num(media, 2)} s por avaliação; a mais lenta {num(max(segundos), 2)} s")
    if total_grade is not None:
        print(f"a grade tem {total_grade} combinações: cerca de {duracao(total_grade * media)}")
    if a.orcamento:
        print(f"{a.orcamento} avaliações: cerca de {duracao(a.orcamento * media)}; "
              f"no pior caso, {duracao(a.orcamento * max(segundos))}")


def cmd_grade(a):
    av = Avaliador(a.dados)
    espaco = ler_espaco(a.espaco)
    total = len(combinacoes(espaco))
    rodar_busca(av, busca_grade(av, espaco), a.nome, f"grade de {a.espaco}: {total} combinações")


def cmd_aleatoria(a):
    av = Avaliador(a.dados)
    espaco = ler_espaco(a.espaco)
    nome = a.nome or f"aleatoria-{a.orcamento}-s{a.semente}"
    rodar_busca(av, busca_aleatoria(av, espaco, a.orcamento, a.semente), nome,
                f"busca aleatória em {a.espaco}: {a.orcamento} avaliações, semente {a.semente}", a.orcamento)


def cmd_bayesiana(a):
    av = Avaliador(a.dados)
    espaco = ler_espaco(a.espaco)
    if not 2 <= a.iniciais < a.orcamento:
        sys.exit("--iniciais precisa ser pelo menos 2 e menor que o --orcamento")
    nome = a.nome or f"bayesiana-{a.orcamento}-s{a.semente}"
    rodar_busca(av, busca_bayesiana(av, espaco, a.orcamento, a.semente, a.iniciais), nome,
                f"otimização bayesiana em {a.espaco}: {a.orcamento} avaliações "
                f"({a.iniciais} sorteadas e {a.orcamento - a.iniciais} escolhidas pelo modelo), semente {a.semente}",
                a.orcamento)


def cmd_repetir(a):
    av = Avaliador(a.dados)
    espaco = ler_espaco(a.espaco)
    atual = av.avaliar(ATUAL)["auc"]
    print(f"{a.metodo} em {a.espaco}, {a.orcamento} avaliações, sementes {a.de} a {a.ate}"
          + (f" ({a.iniciais} sorteadas no início)" if a.metodo == "bayesiana" else "") + ", sem gravar")
    print(f"configuração de hoje: {pct(atual)}")
    print(f"{'semente':>7}  {'melhor':>8}  {'achou na':>8}  {'segundos':>8}")
    curvas, segs = [], []
    for s in range(a.de, a.ate + 1):
        if a.metodo == "aleatoria":
            g = busca_aleatoria(av, espaco, a.orcamento, s)
        else:
            g = busca_bayesiana(av, espaco, a.orcamento, s, a.iniciais)
        t0 = time.process_time()
        aucs = [r["auc"] for _, r in g]
        dt = time.process_time() - t0
        print(f"{s:>7}  {pct(max(aucs)):>8}  {int(np.argmax(aucs)) + 1:>8}  {num(dt, 1):>8}")
        curvas.append(np.maximum.accumulate(aucs))
        segs.append(dt)
    curvas = np.array(curvas)
    m = curvas[:, -1]
    print(f"{len(m)} sementes: melhor de {pct(m.min())} a {pct(m.max())}, mediana {pct(float(np.median(m)))}; "
          f"{num(float(np.mean(segs)), 1)} s de processador por busca")
    marcos = [n for n in (5, 10, 15, 20, 27, 40, 60, 80, 100) if n <= a.orcamento]
    print("a melhor até ali, mediana das sementes (a pior e a melhor semente entre parênteses):")
    for n in marcos:
        col = curvas[:, n - 1]
        print(f"   {n:>3} avaliações: {pct(float(np.median(col)))} ({pct(col.min())} a {pct(col.max())})")


def cmd_meses(a):
    d = pd.read_csv(a.arquivo)
    r = np.corrcoef(d["auc_marco"], d["auc_abril"])[0, 1]
    melhor_marco = d.loc[d["auc_marco"].idxmax()]
    melhor_abril = d.loc[d["auc_abril"].idxmax()]
    posicao = int((d["auc_abril"] > melhor_marco["auc_abril"]).sum()) + 1
    print(f"{a.arquivo}: {len(d)} avaliações")
    print(f"correlação entre o acerto de março e o de abril: {num(r)}")
    print(f"a melhor em março: {pct(melhor_marco['auc_marco'])} em março, {pct(melhor_marco['auc_abril'])} em abril "
          f"({posicao}ª de {len(d)} em abril)")
    print(f"a melhor em abril: {pct(melhor_abril['auc_abril'])} em abril")
    print(f"mediana das {len(d)} em abril: {pct(float(d['auc_abril'].median()))}")
    if len(d) >= 100:
        top_m = set(d.nlargest(10, "auc_marco").index)
        top_a = set(d.nlargest(10, "auc_abril").index)
        print(f"das 10 melhores em março, {len(top_m & top_a)} estão entre as 10 melhores em abril")
    hoje = Avaliador(a.dados).avaliar(ATUAL)
    print(f"a configuração de hoje: {pct(hoje['auc_marco'])} em março, {pct(hoje['auc_abril'])} em abril")
    juntos = d.loc[d["auc"].idxmax()]
    ganha = [m for m, c in (("março", "auc_marco"), ("abril", "auc_abril")) if juntos[c] > hoje[c]]
    print(f"a melhor nos dois meses juntos ({pct(juntos['auc'])}): {pct(juntos['auc_marco'])} em março, "
          f"{pct(juntos['auc_abril'])} em abril; ganha de hoje em "
          + (" e em ".join(ganha) if ganha else "nenhum dos dois"))


def cmd_efeitos(a):
    d = pd.read_csv(a.arquivo)
    print(f"{a.arquivo}: {len(d)} avaliações; acerto do par em março e abril, média e melhor por valor")
    variam = [n for n in NOMES if d[n].nunique() > 1]
    for nome in variam:
        g = d.groupby(nome)["auc"].agg(["mean", "max", "size"])
        valores = "  ".join(f"{num(v, 3).rstrip('0').rstrip(',')}: {pct(r['mean'], 1)} | {pct(r['max'], 1)}"
                            for v, r in g.iterrows())
        print(f"{nome:<18} {valores}")
        print(f"{'':<18} média vai {num(100 * (g['mean'].max() - g['mean'].min()))} ponto; "
              f"melhor vai {num(100 * (g['max'].max() - g['max'].min()))} ponto")
    if {"max_iter", "max_leaf_nodes"} <= set(variam):
        t = d.groupby(["max_iter", "max_leaf_nodes"])["segundos"].mean().unstack()
        print("segundos de processador por avaliação, rodadas × folhas:")
        print(f"{'rodadas':>9}" + "".join(f"{int(c):>7}" for c in t.columns))
        for i, linha in t.iterrows():
            print(f"{int(i):>9}" + "".join(f"{num(v):>7}" for v in linha))


def cmd_ruido(a):
    av = Avaliador(a.dados)
    d = pd.read_csv(a.arquivo)
    linha = d.loc[d["auc"].idxmax()]
    melhor = {n: (int(linha[n]) if n in ("max_iter", "max_leaf_nodes", "min_samples_leaf") else float(linha[n]))
              for n in NOMES}
    na, nb = av.notas(melhor), av.notas(ATUAL)
    y = av.yv
    dif = roc_auc_score(y, na) - roc_auc_score(y, nb)
    rng = np.random.default_rng(a.semente)
    difs, sozinha = [], []
    for _ in range(a.reamostras):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        x = roc_auc_score(y[i], na[i])
        difs.append(x - roc_auc_score(y[i], nb[i]))
        sozinha.append(x)
    difs, sozinha = np.array(difs), np.array(sozinha)
    lo, hi = np.percentile(difs, [5, 95])
    print(f"a melhor de {a.arquivo}: {texto_config(melhor)}")
    print(f"acerto do par: a melhor {pct(roc_auc_score(y, na))}, a de hoje {pct(roc_auc_score(y, nb))}, "
          f"diferença {num(100 * dif)} ponto")
    print(f"{len(difs)} reamostras das {len(y)} entregas de março e abril, sorteadas com reposição:")
    print(f"   o acerto da melhor anda de {pct(np.percentile(sozinha, 5))} a {pct(np.percentile(sozinha, 95))} "
          "(90% das reamostras)")
    print(f"   a diferença anda de {num(100 * lo)} a {num(100 * hi)} ponto (90% das reamostras); "
          f"a de hoje ganha em {pct(float((difs < 0).mean()), 0)} delas")


def main(argv=None):
    p = argparse.ArgumentParser(description="Busca de hiperparâmetros do boosting da lista da manhã.")
    p.add_argument("--dados", default=str(AQUI / DADOS_PADRAO), help="planilha de entregas")
    sub = p.add_subparsers(dest="comando", required=True)

    sub.add_parser("atual", help="mede a configuração de hoje")

    c = sub.add_parser("custo", help="mede o tempo de algumas avaliações do espaço")
    c.add_argument("--espaco", required=True)
    c.add_argument("--amostras", type=int, default=10)
    c.add_argument("--orcamento", type=int)
    c.add_argument("--semente", type=int, default=0)

    g = sub.add_parser("grade", help="avalia todas as combinações da lista de valores")
    g.add_argument("--espaco", required=True)
    g.add_argument("--nome", required=True)

    for metodo in ("aleatoria", "bayesiana"):
        m = sub.add_parser(metodo, help=f"busca {metodo} com um orçamento de avaliações")
        m.add_argument("--espaco", required=True)
        m.add_argument("--orcamento", type=int, required=True)
        m.add_argument("--semente", type=int, required=True)
        m.add_argument("--nome")
        if metodo == "bayesiana":
            m.add_argument("--iniciais", type=int, default=8)

    r = sub.add_parser("repetir", help="repete a busca com várias sementes, sem gravar")
    r.add_argument("--metodo", choices=["aleatoria", "bayesiana"], required=True)
    r.add_argument("--espaco", required=True)
    r.add_argument("--orcamento", type=int, required=True)
    r.add_argument("--iniciais", type=int, default=8)
    r.add_argument("--de", type=int, default=0)
    r.add_argument("--ate", type=int, default=9)

    me = sub.add_parser("meses", help="compara março e abril numa busca gravada")
    me.add_argument("arquivo")

    ef = sub.add_parser("efeitos", help="quanto cada hiperparâmetro mexe no acerto numa grade gravada")
    ef.add_argument("arquivo")

    ru = sub.add_parser("ruido", help="quanto a diferença da melhor para a de hoje muda com outras entregas")
    ru.add_argument("arquivo")
    ru.add_argument("--reamostras", type=int, default=1000)
    ru.add_argument("--semente", type=int, default=0)

    a = p.parse_args(argv)
    if a.comando in ("aleatoria", "bayesiana", "repetir", "custo") and getattr(a, "orcamento", 1) is not None \
            and getattr(a, "orcamento", 1) < 1:
        sys.exit("--orcamento precisa ser pelo menos 1")
    {"atual": cmd_atual, "custo": cmd_custo, "grade": cmd_grade, "aleatoria": cmd_aleatoria,
     "bayesiana": cmd_bayesiana, "repetir": cmd_repetir, "meses": cmd_meses, "efeitos": cmd_efeitos,
     "ruido": cmd_ruido}[a.comando](a)


if __name__ == "__main__":
    main()
