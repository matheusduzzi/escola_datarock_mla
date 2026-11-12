"""Valores de Shapley de uma tabela de notas com três perguntas, nas entregas da Loja Exemplo.

    python3 shap_pequeno.py                          # as contas da aula
    python3 shap_pequeno.py --entrega 8403           # as mesmas contas para outra entrega
    python3 shap_pequeno.py --entrega 8403 --referencia 3
    python3 shap_pequeno.py --dados ../dados/entregas/entregas.csv

Só usa o pandas e o NumPy. A conta é a exata: com 3 colunas, são 8 combinações e 6 ordens,
e o programa faz todas. Nada é sorteado, então a saída sai igual toda vez.
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd

DADOS = "../dados/entregas/entregas.csv"
INICIO_VALIDACAO = "2026-03-01"
INICIO_TESTE = "2026-05-01"
ENTREGA_DA_AULA = 8324
REFERENCIA_DA_AULA = 3

# Cada tabela de notas faz três perguntas de sim ou não sobre a entrega.
PERGUNTAS = {
    "economica": ("econômica", lambda d: d["transportadora"].eq("Economica")),
    "pesado": ("pesado", lambda d: d["peso_kg"].gt(10)),
    "frete_alto": ("frete alto", lambda d: d["valor_frete"].gt(40)),
    "longe": ("longe", lambda d: d["distancia_km"].gt(1500)),
}
TABELA_FRETE = ("economica", "pesado", "frete_alto")
TABELA_DISTANCIA = ("economica", "pesado", "longe")


def br(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


def pct(x, casas=1):
    return br(100 * x, casas) + "%"


def pontos(x):
    """Diferença de nota em pontos percentuais, com sinal."""
    v = round(100 * x, 1)
    if v == 0:
        return "0,0"
    return ("+" if v > 0 else "−") + br(abs(v))


def sim(v):
    return "sim" if v else "não"


def ler(caminho):
    d = pd.read_csv(caminho, parse_dates=["data_despacho"])
    d["atrasou"] = d["situacao"].ne("no_prazo").astype(int)
    for nome, (_, regra) in PERGUNTAS.items():
        d[nome] = regra(d).astype(int)
    return d


def separar(d):
    treino = d[d["data_despacho"] < INICIO_VALIDACAO]
    validacao = d[(d["data_despacho"] >= INICIO_VALIDACAO) & (d["data_despacho"] < INICIO_TESTE)]
    return treino, validacao


def treinar_tabela(treino, colunas):
    """A nota de cada combinação de respostas é a parte do treino que atrasou nela."""
    g = treino.groupby(list(colunas))["atrasou"].agg(["size", "mean"])
    return {chave: (int(n), float(m)) for chave, (n, m) in g.iterrows()}


def nota(tabela, respostas):
    """Nota de uma combinação de respostas (tupla de 0 e 1 na ordem das colunas)."""
    return tabela[tuple(int(r) for r in respostas)][1]


def misturar(x, r, quais):
    """Combinação com as colunas em `quais` vindas da entrega x e as outras da referência r."""
    return tuple(x[i] if i in quais else r[i] for i in range(len(x)))


def valor(tabela, x, refs, quais):
    """Nota média quando as colunas em `quais` vêm da entrega e as outras vêm de cada referência."""
    return float(np.mean([nota(tabela, misturar(x, r, quais)) for r in refs]))


def shapley_por_ordens(tabela, x, refs):
    """Para cada ordem das colunas, quanto a nota muda quando cada coluna troca para o valor da entrega.
    O valor de Shapley de uma coluna é a média dessas mudanças nas 6 ordens."""
    k = len(x)
    linhas = []
    for ordem in itertools.permutations(range(k)):
        ja, mudancas = set(), [0.0] * k
        for i in ordem:
            antes = valor(tabela, x, refs, ja)
            ja = ja | {i}
            mudancas[i] = valor(tabela, x, refs, ja) - antes
        linhas.append((ordem, mudancas))
    phi = [float(np.mean([m[i] for _, m in linhas])) for i in range(k)]
    return linhas, phi


def respostas(linha, colunas):
    return tuple(int(linha[c]) for c in colunas)


def acerto_do_par(nota_, y):
    """Sorteando uma entrega que atrasou e uma que chegou no prazo, em quantos pares a que atrasou
    tem a nota maior. Empate vale meio."""
    nota_, y = np.asarray(nota_, dtype=float), np.asarray(y)
    ordem = pd.Series(nota_).rank().to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((ordem[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def descrever(linha):
    return (f"{linha['transportadora']}, {br(linha['peso_kg'])} kg, frete R$ {br(linha['valor_frete'], 2)}, "
            f"{br(linha['distancia_km'])} km")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Valores de Shapley de uma tabela de notas com três perguntas.")
    ap.add_argument("--dados", default=DADOS, help="caminho do entregas.csv")
    ap.add_argument("--entrega", type=int, default=ENTREGA_DA_AULA, help="id da entrega explicada")
    ap.add_argument("--referencia", type=int, default=REFERENCIA_DA_AULA, help="id da entrega de referência")
    a = ap.parse_args(argv)

    d = ler(a.dados)
    treino, validacao = separar(d)
    por_id = d.set_index("id_entrega")
    for nome, ident in (("--entrega", a.entrega), ("--referencia", a.referencia)):
        if ident not in por_id.index:
            sys.exit(f"{nome} {ident}: não existe entrega com esse id (vão de 1 a {len(d)})")
    primeira_do_teste = int(d.loc[d["data_despacho"] >= INICIO_TESTE, "id_entrega"].min())
    for nome, ident in (("--entrega", a.entrega), ("--referencia", a.referencia)):
        if ident >= primeira_do_teste:
            sys.exit(f"{nome} {ident}: é de maio ou junho, o teste. Escolha uma entrega até abril.")

    tab = treinar_tabela(treino, TABELA_FRETE)
    rotulos = [PERGUNTAS[c][0] for c in TABELA_FRETE]

    print("1. A tabela de notas")
    print(f"treino: {len(treino)} entregas de julho de 2025 a fevereiro de 2026, "
          f"{int(treino['atrasou'].sum())} atrasaram ({pct(treino['atrasou'].mean())})")
    print(f"validação: {len(validacao)} entregas de março e abril de 2026, "
          f"{int(validacao['atrasou'].sum())} atrasaram ({pct(validacao['atrasou'].mean())})")
    print("   econômica: transportadora Economica; pesado: mais de 10 kg; frete alto: frete acima de R$ 40")
    print(f"{'econômica':<11}{'pesado':<8}{'frete alto':<12}{'entregas':>8}{'nota':>9}")
    for chave in sorted(tab):
        n, m = tab[chave]
        print(f"{sim(chave[0]):<11}{sim(chave[1]):<8}{sim(chave[2]):<12}{n:>8}{pct(m, 2):>9}")
    nv = [nota(tab, respostas(l, TABELA_FRETE)) for _, l in validacao.iterrows()]
    print(f"   acerto do par em março e abril: {pct(acerto_do_par(nv, validacao['atrasou']))} "
          f"(a mesma nota para todos dá 50,0%)")

    xl, rl = por_id.loc[a.entrega], por_id.loc[a.referencia]
    x, r = respostas(xl, TABELA_FRETE), respostas(rl, TABELA_FRETE)
    print()
    print("2. Uma entrega contra uma referência")
    print(f"entrega {a.entrega}: {descrever(xl)}; atrasou: {sim(xl['atrasou'])}; nota {pct(nota(tab, x), 2)}")
    print(f"referência {a.referencia}: {descrever(rl)}; atrasou: {sim(rl['atrasou'])}; nota {pct(nota(tab, r), 2)}")
    print(f"{'colunas que vêm da entrega':<31}{'econômica':<11}{'pesado':<8}{'frete alto':<12}{'nota':>7}")
    for tam in range(4):
        for quais in itertools.combinations(range(3), tam):
            nome = ", ".join(rotulos[i] for i in quais) if quais else "nenhuma"
            m = misturar(x, r, set(quais))
            print(f"{nome:<31}{sim(m[0]):<11}{sim(m[1]):<8}{sim(m[2]):<12}{pct(nota(tab, m), 2):>7}")
    linhas, phi = shapley_por_ordens(tab, x, [r])
    print(f"{'ordem em que as colunas trocam':<31}{'econômica':>10}{'pesado':>8}{'frete alto':>12}")
    for ordem, m in linhas:
        nome = ", ".join(rotulos[i] for i in ordem)
        print(f"{nome:<31}{pontos(m[0]):>10}{pontos(m[1]):>8}{pontos(m[2]):>12}")
    print(f"{'média das 6 ordens':<31}{pontos(phi[0]):>10}{pontos(phi[1]):>8}{pontos(phi[2]):>12}")
    print(f"   soma: {pontos(sum(phi))} pontos = nota da entrega {pct(nota(tab, x), 2)} − nota da referência "
          f"{pct(nota(tab, r), 2)}")

    print()
    print("3. A mesma entrega, três referências")
    refs = [
        (f"a entrega {a.referencia}", [r]),
        (f"as {len(treino)} entregas do treino", [respostas(l, TABELA_FRETE) for _, l in treino.iterrows()]),
    ]
    eco = treino[treino["economica"] == 1]
    refs.append((f"as {len(eco)} da Econômica no treino", [respostas(l, TABELA_FRETE) for _, l in eco.iterrows()]))
    print(f"{'referência':<33}{'valor base':>11}{'econômica':>10}{'pesado':>8}{'frete alto':>12}{'soma':>7}")
    for nome, rs in refs:
        base = valor(tab, x, rs, set())
        _, ph = shapley_por_ordens(tab, x, rs)
        print(f"{nome:<33}{pct(base):>11}{pontos(ph[0]):>10}{pontos(ph[1]):>8}{pontos(ph[2]):>12}"
              f"{pontos(sum(ph)):>7}")
    print(f"   nota da entrega {a.entrega}: {pct(nota(tab, x))}; valor base + soma = nota, nas três linhas")

    print()
    print("4. Frete alto e distância, no treino")
    print(f"{'distância':<20}{'frete até R$ 40':>22}{'frete acima de R$ 40':>24}")
    for longe, nome in ((0, "até 1500 km"), (1, "acima de 1500 km")):
        partes = []
        for fa in (0, 1):
            g = treino[(treino["longe"] == longe) & (treino["frete_alto"] == fa)]["atrasou"]
            partes.append(f"{len(g)} entregas, {pct(g.mean())}")
        print(f"{nome:<20}{partes[0]:>22}{partes[1]:>24}")
    print(f"   correlação entre distância e frete: {br(treino['distancia_km'].corr(treino['valor_frete']), 2)}")
    tab2 = treinar_tabela(treino, TABELA_DISTANCIA)
    x2 = respostas(xl, TABELA_DISTANCIA)
    rs2 = [respostas(l, TABELA_DISTANCIA) for _, l in treino.iterrows()]
    _, ph2 = shapley_por_ordens(tab2, x2, rs2)
    nv2 = [nota(tab2, respostas(l, TABELA_DISTANCIA)) for _, l in validacao.iterrows()]
    _, ph1 = shapley_por_ordens(tab, x, refs[1][1])
    print(f"   a entrega {a.entrega} em duas tabelas, referência: as {len(treino)} entregas do treino")
    print(f"{'tabela':<31}{'econômica':>10}{'pesado':>8}{'frete alto':>12}{'longe':>7}{'nota':>7}{'par':>7}")
    print(f"{'econômica, pesado, frete alto':<31}{pontos(ph1[0]):>10}{pontos(ph1[1]):>8}{pontos(ph1[2]):>12}"
          f"{'':>7}{pct(nota(tab, x)):>7}{pct(acerto_do_par(nv, validacao['atrasou'])):>7}")
    print(f"{'econômica, pesado, longe':<31}{pontos(ph2[0]):>10}{pontos(ph2[1]):>8}{'':>12}"
          f"{pontos(ph2[2]):>7}{pct(nota(tab2, x2)):>7}{pct(acerto_do_par(nv2, validacao['atrasou'])):>7}")
    print("   par: acerto do par em março e abril; longe: mais de 1500 km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
