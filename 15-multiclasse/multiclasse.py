"""Contas da aula Multiclasse, do curso ML avançado.

    python3 multiclasse.py
    python3 multiclasse.py --dados caminho/para/entregas.csv

Treina a regressão logística de quatro situações (no prazo, atraso curto, atraso longo, extravio)
de três jeitos: multinomial, um contra o resto e um contra um. Treino: julho de 2025 a fevereiro
de 2026. Validação: março e abril de 2026. Maio e junho ficam guardados para o teste do curso.
Tudo no pandas e no NumPy, pelo método de Newton, sem sorteio: a saída sai igual toda vez.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
CLASSES = ["no_prazo", "atraso_curto", "atraso_longo", "extravio"]
NOMES = ["no prazo", "atraso curto", "atraso longo", "extravio"]
PRIORITARIAS = [2, 3]
COLUNAS = ["distância (mil km)", "Econômica", "Própria", "mais de 10 kg", "Econômica com mais de 10 kg",
           "véspera de feriado", "pico de fim de ano", "itens", "prazo prometido"]
L2, MAX_PASSOS, TOLERANCIA = 0.001, 100, 1e-10


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"


def colunas(df):
    data = pd.to_datetime(df["data_despacho"])
    econ = (df["transportadora"] == "Economica").astype(float)
    propria = (df["transportadora"] == "Propria").astype(float)
    pesado = (df["peso_kg"] > 10).astype(float)
    pico = ((data >= "2025-11-20") & (data <= "2025-12-24")).astype(float)
    x = [df["distancia_km"] / 1000, econ, propria, pesado, econ * pesado, df["vespera_feriado"].astype(float),
         pico, (df["itens"] - 2) / 3, (df["prazo_prometido_dias"] - 6) / 3]
    return np.column_stack([np.ones(len(df))] + [np.asarray(c, dtype=float) for c in x])


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def newton(x, y_alvo, notas):
    """Minimiza a perda logarítmica média + L2/2 * soma dos pesos ao quadrado (sem o intercepto).

    y_alvo tem uma coluna por classe da saída; notas(z) dá as notas e a derivada segunda por linha."""
    n, k = x.shape[1], y_alvo.shape[1]
    w = np.zeros((n, k))
    pena = np.full(n, L2)
    pena[0] = 0.0
    for _ in range(MAX_PASSOS):
        p = notas(x @ w)
        g = (x.T @ (p - y_alvo) / len(x) + pena[:, None] * w).ravel(order="F")
        if np.abs(g).max() < TOLERANCIA:
            return w
        h = np.zeros((n * k, n * k))
        for a in range(k):
            for b in range(k):
                d = p[:, a] * ((a == b) - p[:, b])
                h[a * n:(a + 1) * n, b * n:(b + 1) * n] = (x * d[:, None]).T @ x / len(x)
        h += np.diag(np.tile(pena, k))
        passo = np.linalg.lstsq(h, g, rcond=None)[0]
        w -= passo.reshape((n, k), order="F")
    raise SystemExit("O treino não convergiu em 100 passos. Avise na aula.")


def treinar_binaria(x, y):
    return newton(x, y[:, None], sigmoide)[:, 0]


def treinar_multinomial(x, y):
    return newton(x, np.eye(len(CLASSES))[y], softmax)


def matriz(real, prevista):
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(m, (real, prevista), 1)
    return m


def por_classe(m):
    acertos = np.diag(m).astype(float)
    previstas, reais = m.sum(axis=0), m.sum(axis=1)
    precisao = np.divide(acertos, previstas, out=np.zeros(len(CLASSES)), where=previstas > 0)
    recall = acertos / reais
    f1 = np.divide(2 * precisao * recall, precisao + recall, out=np.zeros(len(CLASSES)),
                   where=(precisao + recall) > 0)
    return precisao, recall, f1, reais, previstas


def medias(m, quais=None):
    quais = list(range(len(CLASSES))) if quais is None else quais
    p, r, f1, reais, _ = por_classe(m)
    q = np.array(quais)
    acertos = np.diag(m)[q].sum()
    p_micro = acertos / m[:, q].sum() if m[:, q].sum() else 0.0
    r_micro = acertos / m[q, :].sum()
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if p_micro + r_micro else 0.0
    peso = reais[q] / reais[q].sum()
    return {"micro": (p_micro, r_micro, f1_micro),
            "macro": (p[q].mean(), r[q].mean(), f1[q].mean()),
            "ponderada": ((p[q] * peso).sum(), (r[q] * peso).sum(), (f1[q] * peso).sum())}


def imprimir_matriz(m):
    print("real \\ prevista   " + "".join(f"{n:>13}" for n in NOMES) + "    total")
    for i, n in enumerate(NOMES):
        print(f"{n:<17}" + "".join(f"{v:>13}" for v in m[i]) + f"{m[i].sum():>9}")
    print(f"{'total previsto':<17}" + "".join(f"{v:>13}" for v in m.sum(axis=0)) + f"{m.sum():>9}")


def imprimir_classes(m):
    p, r, f1, reais, previstas = por_classe(m)
    print(f"{'situação':<15}{'reais':>7}{'previstas':>11}{'acertos':>9}{'precisão':>10}{'recall':>9}{'F1':>7}")
    for i, n in enumerate(NOMES):
        prec = num(p[i], 2) if previstas[i] else "sem"
        print(f"{n:<15}{reais[i]:>7}{previstas[i]:>11}{m[i, i]:>9}{prec:>10}{num(r[i], 2):>9}{num(f1[i], 2):>7}")
    print(f"   acerto: {m.trace()} de {m.sum()} = {pct(m.trace() / m.sum())}")


def imprimir_medias(m, quais=None, titulo="médias nas quatro situações"):
    md = medias(m, quais)
    print(f"{titulo:<34}{'precisão':>9}{'recall':>8}{'F1':>7}")
    for nome in ("micro", "macro", "ponderada"):
        p, r, f1 = md[nome]
        print(f"   {nome:<31}{num(p, 2):>9}{num(r, 2):>8}{num(f1, 2):>7}")


def main():
    ap = argparse.ArgumentParser(description="Contas da aula Multiclasse.")
    ap.add_argument("--dados", default=str(AQUI.parent / "dados" / "entregas" / "entregas.csv"),
                    help="caminho do entregas.csv (padrão: ../dados/entregas/entregas.csv)")
    args = ap.parse_args()
    caminho = Path(args.dados)
    if not caminho.is_file():
        raise SystemExit(f"Não achei {caminho}. Rode de dentro da pasta da aula ou passe --dados.")
    df = pd.read_csv(caminho)
    if sorted(df["situacao"].unique()) != sorted(CLASSES):
        raise SystemExit(f"A coluna situacao precisa ter as quatro situações: {', '.join(CLASSES)}.")
    data = pd.to_datetime(df["data_despacho"])
    treino = df[data < "2026-03-01"]
    valid = df[(data >= "2026-03-01") & (data < "2026-05-01")]
    codigo = {c: i for i, c in enumerate(CLASSES)}
    yt, yv = treino["situacao"].map(codigo).to_numpy(), valid["situacao"].map(codigo).to_numpy()
    xt, xv = colunas(treino), colunas(valid)

    with np.errstate(all="ignore"):
        w_multi = treinar_multinomial(xt, yt)
        w_ovr = np.column_stack([treinar_binaria(xt, (yt == k).astype(float)) for k in range(len(CLASSES))])
        pares = list(itertools.combinations(range(len(CLASSES)), 2))
        w_ovo = {}
        for a, b in pares:
            linhas = (yt == a) | (yt == b)
            w_ovo[(a, b)] = treinar_binaria(xt[linhas], (yt[linhas] == b).astype(float))
        nota_multi = softmax(xv @ w_multi)
        nota_ovr = sigmoide(xv @ w_ovr)
        votos, confianca = np.zeros((len(yv), 4)), np.zeros((len(yv), 4))
        for (a, b), w in w_ovo.items():
            pb = sigmoide(xv @ w)
            votos[:, b] += pb >= 0.5
            votos[:, a] += pb < 0.5
            confianca[:, b] += pb
            confianca[:, a] += 1 - pb
    for nome, arr in (("multinomial", nota_multi), ("um contra o resto", nota_ovr), ("um contra um", confianca)):
        if not np.isfinite(arr).all():
            raise SystemExit(f"O treino {nome} não convergiu: apareceu um valor infinito. Avise na aula.")

    print("1. As quatro situações")
    print(f"treino: {len(treino)} entregas de julho de 2025 a fevereiro de 2026")
    print(f"validação: {len(valid)} entregas de março e abril de 2026")
    print(f"{'situação':<15}{'treino':>8}{'':>8}{'validação':>11}{'':>8}")
    for i, n in enumerate(NOMES):
        nt, nv = int((yt == i).sum()), int((yv == i).sum())
        print(f"{n:<15}{nt:>8}{pct(nt / len(yt)):>8}{nv:>11}{pct(nv / len(yv)):>8}")
    print("   colunas do modelo: " + "; ".join(COLUNAS))

    print()
    print("2. Três estratégias, a mesma regra: fica a situação de maior nota")
    prev = {"multinomial": nota_multi.argmax(axis=1), "um contra o resto": nota_ovr.argmax(axis=1),
            "um contra um": (votos + confianca / 100).argmax(axis=1)}
    print(f"{'estratégia':<20}{'modelos':>8}{'acerto':>8}" + "".join(f"{n:>13}" for n in NOMES))
    for nome, modelos in (("multinomial", 1), ("um contra o resto", 4), ("um contra um", 6)):
        m = matriz(yv, prev[nome])
        print(f"{nome:<20}{modelos:>8}{pct(m.trace() / m.sum()):>8}" + "".join(f"{v:>13}" for v in m.sum(axis=0)))
    print("   nas colunas das situações: quantas entregas cada estratégia pôs em cada uma")
    soma = nota_ovr.sum(axis=1)
    print(f"   soma das quatro notas por entrega: multinomial sempre {pct(nota_multi.sum(axis=1).mean(), 0)}; "
          f"um contra o resto de {pct(soma.min())} a {pct(soma.max())}")
    i = int(soma.argmax())
    ent = valid.iloc[i]
    print(f"   entrega {ent['id_entrega']}: {ent['transportadora']}, {num(ent['peso_kg'])} kg, "
          f"{num(ent['distancia_km'])} km, véspera de feriado: {'sim' if ent['vespera_feriado'] else 'não'}; "
          f"situação real: {NOMES[yv[i]]}")
    print(f"   {'':<20}" + "".join(f"{n:>13}" for n in NOMES) + f"{'soma':>9}")
    for nome, arr in (("multinomial", nota_multi), ("um contra o resto", nota_ovr)):
        print(f"   {nome:<20}" + "".join(f"{pct(v):>13}" for v in arr[i]) + f"{pct(arr[i].sum()):>9}")

    print()
    print("3. Modelo multinomial, maior nota, nas entregas de março e abril")
    m_a = matriz(yv, prev["multinomial"])
    imprimir_matriz(m_a)
    imprimir_classes(m_a)
    imprimir_medias(m_a)
    p_mac, r_mac, _ = medias(m_a)["macro"]
    print(f"   F1 tirado da precisão macro e do recall macro: 2 × {num(p_mac, 2)} × {num(r_mac, 2)} ÷ "
          f"({num(p_mac, 2)} + {num(r_mac, 2)}) = {num(2 * p_mac * r_mac / (p_mac + r_mac), 2)}")
    m_base = matriz(yv, np.zeros(len(yv), dtype=int))
    md = medias(m_base)
    print(f"   sempre 'no prazo': acerto {pct(m_base.trace() / m_base.sum())}; F1 micro {num(md['micro'][2], 2)}, "
          f"macro {num(md['macro'][2], 2)}, ponderada {num(md['ponderada'][2], 2)}")

    print()
    print("4. O mesmo modelo, regra da nota acima do normal")
    j = int(nota_multi[:, 0].argmin())
    print(f"   na maior nota, o extravio não ganha: a nota de extravio mais alta é {pct(nota_multi[:, 3].max())}; "
          f"a menor nota de no prazo é {pct(nota_multi[j, 0])} (entrega {valid.iloc[j]['id_entrega']})")
    taxa = np.bincount(yt, minlength=len(CLASSES)) / len(yt)
    print("   fica a situação em que a nota mais passa da taxa do treino: nota ÷ taxa, a maior vence")
    print("   taxa do treino: " + "; ".join(f"{n} {pct(t)}" for n, t in zip(NOMES, taxa)))
    m_b = matriz(yv, (nota_multi / taxa).argmax(axis=1))
    imprimir_matriz(m_b)
    imprimir_classes(m_b)
    imprimir_medias(m_b)

    print()
    print("5. Só atraso longo e extravio")
    for nome, m in (("maior nota", m_a), ("nota acima do normal", m_b)):
        imprimir_medias(m, PRIORITARIAS, titulo=f"regra {nome}")


if __name__ == "__main__":
    main()
