"""Contas da aula Multiclasse na Loja Exemplo, do curso ML avançado.

    python3 custos.py
    python3 custos.py --custos custos_logistica_extravio_500.csv
    python3 custos.py --custos custos_aviso.csv
    python3 custos.py --dados caminho/para/entregas.csv

Treina a mesma regressão logística multinomial da aula 15 (quatro situações, mesmas colunas, mesmo
treino de julho de 2025 a fevereiro de 2026) e mede, nas entregas de março e abril de 2026, quanto
custa cada regra de decisão com a matriz de custo do arquivo. Maio e junho ficam guardados para o
teste do curso. Tudo no pandas e no NumPy, pelo método de Newton, sem sorteio: a saída sai igual
toda vez.

A matriz de custo tem uma linha por situação real e uma coluna por situação prevista, em reais por
entrega. A primeira coluna do CSV se chama "real".
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
CLASSES = ["no_prazo", "atraso_curto", "atraso_longo", "extravio"]
NOMES = ["no prazo", "atraso curto", "atraso longo", "extravio"]
L2, MAX_PASSOS, TOLERANCIA = 0.001, 100, 1e-10
EXEMPLO = 9717


def num(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"


def reais(x, centavos=False):
    """R$ com ponto de milhar; sem centavos, a não ser que peça."""
    if centavos:
        return "R$ " + f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return "R$ " + f"{int(round(x)):,}".replace(",", ".")


def colunas(df):
    """As mesmas nove colunas da aula 15, que a loja sabe no despacho."""
    data = pd.to_datetime(df["data_despacho"])
    econ = (df["transportadora"] == "Economica").astype(float)
    propria = (df["transportadora"] == "Propria").astype(float)
    pesado = (df["peso_kg"] > 10).astype(float)
    pico = ((data >= "2025-11-20") & (data <= "2025-12-24")).astype(float)
    x = [df["distancia_km"] / 1000, econ, propria, pesado, econ * pesado, df["vespera_feriado"].astype(float),
         pico, (df["itens"] - 2) / 3, (df["prazo_prometido_dias"] - 6) / 3]
    return np.column_stack([np.ones(len(df))] + [np.asarray(c, dtype=float) for c in x])


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def treinar_multinomial(x, y):
    """Minimiza a perda logarítmica média + L2/2 * soma dos pesos ao quadrado (sem o intercepto)."""
    n, k = x.shape[1], len(CLASSES)
    alvo = np.eye(k)[y]
    w = np.zeros((n, k))
    pena = np.full(n, L2)
    pena[0] = 0.0
    for _ in range(MAX_PASSOS):
        p = softmax(x @ w)
        g = (x.T @ (p - alvo) / len(x) + pena[:, None] * w).ravel(order="F")
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


def ler_custos(caminho):
    """Lê a matriz de custo e recusa arquivo incompleto, com nome errado ou valor que não é número."""
    if not caminho.is_file():
        raise SystemExit(f"Não achei {caminho}. Rode de dentro da pasta da aula ou passe --custos.")
    df = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    if list(df.columns[:1]) != ["real"]:
        raise SystemExit(f"{caminho.name}: a primeira coluna precisa se chamar 'real' (a situação real de cada linha).")
    faltam = [c for c in CLASSES if c not in df.columns[1:]]
    sobram = [c for c in df.columns[1:] if c not in CLASSES]
    if faltam or sobram or len(df.columns) != 5:
        raise SystemExit(f"{caminho.name}: as colunas previstas precisam ser {', '.join(CLASSES)}. "
                         f"Faltam: {', '.join(faltam) or 'nenhuma'}. Sobram: {', '.join(sobram) or 'nenhuma'}.")
    linhas = df["real"].str.strip().tolist()
    if sorted(linhas) != sorted(CLASSES) or len(linhas) != 4:
        raise SystemExit(f"{caminho.name}: precisa de uma linha para cada situação real: {', '.join(CLASSES)}.")
    m = np.zeros((4, 4))
    for _, linha in df.iterrows():
        i = CLASSES.index(linha["real"].strip())
        for j, c in enumerate(CLASSES):
            valor = str(linha[c]).strip().replace(",", ".")
            if valor == "":
                raise SystemExit(f"{caminho.name}: célula vazia na linha {linha['real'].strip()}, coluna {c}. Preencha as 16 células.")
            try:
                m[i, j] = float(valor)
            except ValueError:
                raise SystemExit(f"{caminho.name}: '{linha[c]}' na linha {linha['real'].strip()}, coluna {c} não é número.")
            if not math.isfinite(m[i, j]):
                raise SystemExit(f"{caminho.name}: valor infinito na linha {linha['real'].strip()}, coluna {c}.")
    return m


def matriz(real, prevista):
    m = np.zeros((4, 4), dtype=int)
    np.add.at(m, (real, prevista), 1)
    return m


def imprimir_grade(m, formato, totais=True):
    print("real \\ prevista".ljust(17) + "".join(f"{n:>13}" for n in NOMES) + (f"{'total':>11}" if totais else ""))
    for i, n in enumerate(NOMES):
        print(f"{n:<17}" + "".join(f"{formato(v):>13}" for v in m[i]) + (f"{formato(m[i].sum()):>11}" if totais else ""))
    if totais:
        print(f"{'total':<17}" + "".join(f"{formato(v):>13}" for v in m.sum(axis=0)) + f"{formato(m.sum()):>11}")


def main():
    ap = argparse.ArgumentParser(description="Contas da aula Multiclasse na Loja Exemplo.")
    ap.add_argument("--dados", default=str(AQUI.parent / "dados" / "entregas" / "entregas.csv"),
                    help="caminho do entregas.csv (padrão: ../dados/entregas/entregas.csv)")
    ap.add_argument("--custos", default=str(AQUI / "custos_logistica.csv"),
                    help="matriz de custo em reais por entrega (padrão: custos_logistica.csv)")
    ap.add_argument("--decidir-com", default=None,
                    help="outra matriz só para escolher a situação; o custo continua medido com --custos")
    args = ap.parse_args()
    custo = ler_custos(Path(args.custos))
    outra = ler_custos(Path(args.decidir_com)) if args.decidir_com else None
    caminho = Path(args.dados)
    if not caminho.is_file():
        raise SystemExit(f"Não achei {caminho}. Rode de dentro da pasta da aula ou passe --dados.")
    df = pd.read_csv(caminho)
    if sorted(df["situacao"].unique()) != sorted(CLASSES):
        raise SystemExit(f"A coluna situacao precisa ter as quatro situações: {', '.join(CLASSES)}.")
    data = pd.to_datetime(df["data_despacho"])
    treino = df[data < "2026-03-01"]
    valid = df[(data >= "2026-03-01") & (data < "2026-05-01")].reset_index(drop=True)
    codigo = {c: i for i, c in enumerate(CLASSES)}
    yt, yv = treino["situacao"].map(codigo).to_numpy(), valid["situacao"].map(codigo).to_numpy()
    mes = pd.to_datetime(valid["data_despacho"]).dt.month.to_numpy()

    with np.errstate(all="ignore"):
        w = treinar_multinomial(colunas(treino), yt)
        nota = softmax(colunas(valid) @ w)
        esperado = nota @ custo
        esperado_outra = nota @ outra if outra is not None else None
    if not np.isfinite(esperado).all():
        raise SystemExit("O treino não convergiu: apareceu um valor infinito. Avise na aula.")
    taxa = np.bincount(yt, minlength=4) / len(yt)

    print(f"1. A matriz de custo: {Path(args.custos).name}")
    print("   reais por entrega; linha = situação real, coluna = situação prevista")
    imprimir_grade(custo, lambda v: num(v, 0), totais=False)
    print(f"validação: {len(valid)} entregas de março e abril de 2026; "
          + "; ".join(f"{n} {int((yv == i).sum())}" for i, n in enumerate(NOMES)))

    regras = {
        "sempre no prazo": np.zeros(len(yv), dtype=int),
        "maior nota": nota.argmax(axis=1),
        "nota acima do normal": (nota / taxa).argmax(axis=1),
        "menor custo esperado": esperado.argmin(axis=1),
    }
    nome_outra = None
    if outra is not None:
        nome_outra = f"decidida com {Path(args.decidir_com).stem}"
        regras[nome_outra] = esperado_outra.argmin(axis=1)
    print()
    print("2. O custo de cada regra, em março e abril")
    largura = max(22, max(len(n) for n in regras) + 2)
    print(f"{'regra':<{largura}}" + "".join(f"{n:>13}" for n in NOMES) + f"{'março':>11}{'abril':>11}{'total':>11}")
    totais = {}
    for nome, prev in regras.items():
        c = custo[yv, prev]
        totais[nome] = c.sum()
        print(f"{nome:<{largura}}" + "".join(f"{v:>13}" for v in np.bincount(prev, minlength=4))
              + f"{reais(c[mes == 3].sum()):>11}{reais(c[mes == 4].sum()):>11}{reais(c.sum()):>11}")
    ideal = custo.min(axis=1)[yv]
    print(f"{'quem acerta tudo':<{largura}}" + "".join(f"{v:>13}" for v in np.bincount(yv, minlength=4))
          + f"{reais(ideal[mes == 3].sum()):>11}{reais(ideal[mes == 4].sum()):>11}{reais(ideal.sum()):>11}")
    print("   nas colunas das situações: quantas entregas cada regra pôs em cada uma")
    if nome_outra:
        print(f"   a regra '{nome_outra}' escolhe pela outra matriz, e o custo dela é medido com {Path(args.custos).name}")
    base = totais["sempre no prazo"]
    for nome in [n for n in regras if n != "sempre no prazo"]:
        dif = base - totais[nome]
        print(f"   {nome}: {reais(abs(dif))} a {'menos' if dif >= 0 else 'mais'} que sempre no prazo")
    print(f"   quem acerta tudo: {reais(base - ideal.sum())} a menos que sempre no prazo")

    prev = regras["menor custo esperado"]
    m = matriz(yv, prev)
    gasto = np.zeros((4, 4))
    np.add.at(gasto, (yv, prev), custo[yv, prev])
    print()
    print("3. A regra do menor custo esperado, situação por situação")
    print("entregas")
    imprimir_grade(m, lambda v: str(int(v)))
    print("reais gastos em cada célula")
    imprimir_grade(gasto, reais)
    print(f"{'situação':<15}{'reais':>7}{'acertos':>9}{'recall':>9}{'gasto':>11}{'por entrega':>13}{'se acertasse tudo':>19}")
    for i, n in enumerate(NOMES):
        r = int(m[i].sum())
        print(f"{n:<15}{r:>7}{int(m[i, i]):>9}{num(m[i, i] / r, 2):>9}{reais(gasto[i].sum()):>11}"
              f"{reais(gasto[i].sum() / r, centavos=True):>13}{reais(custo[i].min() * r):>19}")

    print()
    print("4. Por dentro da regra")
    j = int(np.flatnonzero(valid["id_entrega"].to_numpy() == EXEMPLO)[0]) if EXEMPLO in set(valid["id_entrega"]) else 0
    ent = valid.iloc[j]
    print(f"   entrega {ent['id_entrega']}: {ent['transportadora']}, {num(ent['peso_kg'])} kg, {num(ent['distancia_km'])} km, "
          f"véspera de feriado: {'sim' if ent['vespera_feriado'] else 'não'}; situação real: {NOMES[yv[j]]}")
    print("   notas:          " + "; ".join(f"{n} {pct(v)}" for n, v in zip(NOMES, nota[j])))
    print("   custo esperado: " + "; ".join(f"{n} R$ {num(v, 2)}" for n, v in zip(NOMES, esperado[j]))
          + f" -> fica {NOMES[int(esperado[j].argmin())]}")
    print(f"   nota de extravio mais alta em março e abril: {pct(nota[:, 3].max())}")
    print("   em cada coluna que a regra usou: a nota média da situação prevista e quantas eram dela de verdade")
    for k in range(4):
        dentro = prev == k
        if dentro.any():
            print(f"   prevista {NOMES[k]:<13} {int(dentro.sum()):>5} entregas; nota média de {NOMES[k]} "
                  f"{pct(nota[dentro, k].mean())}; eram {NOMES[k]}: {pct((yv[dentro] == k).mean())}")
    empates = int((np.isclose(esperado, esperado.min(axis=1, keepdims=True)).sum(axis=1) > 1).sum())
    print(f"   entregas com duas ou mais colunas de custo esperado igual ao menor: {empates} (fica a primeira da ordem)")


if __name__ == "__main__":
    main()
