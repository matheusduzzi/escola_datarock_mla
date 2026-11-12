"""As contas da aula «Random Forest»: o benchmark da floresta nas entregas da Loja Exemplo.

    python3 floresta.py                    # leva alguns minutos
    python3 floresta.py --saida outra/pasta

As partes 1 a 6 saem iguais toda vez. A parte 7 é o tempo, que é da sua máquina.
No fim, o programa salva benchmark.csv (uma linha por configuração e sorteio) e
benchmark-rastro.json (de onde vieram os dados, as versões e a máquina).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import arvores as a

DADOS = Path(__file__).resolve().parent.parent / "dados" / "entregas" / "entregas.csv"
SORTEIOS = [0, 1, 2, 3, 4]
ARVORES_PARTE3 = [1, 5, 10, 25, 50, 100, 200]
COLUNAS = [None, 8, 4, 2, 1]
AMOSTRAS = [1.0, 0.5, 0.25]


def rotulo(colunas, amostra=1.0, min_folha=1, arvores=100):
    c = "todas as 16" if colunas is None else str(colunas)
    return f"colunas {c}, amostra {int(round(100 * amostra))}%, folha {min_folha}, {arvores} árvore{'s' if arvores > 1 else ''}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark da Random Forest nas entregas.")
    parser.add_argument("--dados", default=str(DADOS))
    parser.add_argument("--saida", default=".")
    args = parser.parse_args()
    saida = Path(args.saida)

    base = a.ler(args.dados)
    nomes, d = a.separar(base)
    xt, yt, _ = d["treino"]
    xv, yv, idv = d["validacao"]
    cortes = a.faixas(xt)
    ft, fv = a.em_faixas(xt, cortes), a.em_faixas(xv, cortes)
    linhas_csv = []

    print("1. As entregas")
    print(f"treino: {len(yt)} entregas de julho de 2025 a fevereiro de 2026, {yt.sum()} atrasaram ({a.pct(yt.mean())})")
    print(f"validação: {len(yv)} entregas de março e abril de 2026, {yv.sum()} atrasaram ({a.pct(yv.mean())})")
    print("maio e junho, o teste, ficam fechados: nenhuma conta usa essas entregas")
    print(f"   {len(nomes)} colunas: {len(a.NUMERICAS)} números, 3 transportadoras e 5 regiões em colunas de 0 e 1")

    print()
    print("2. As referências: acerto do par na validação")
    arvore = a.treinar_arvore(ft, yt)
    logistica = a.treinar_logistica(xt, yt)
    refs = [("a mesma nota para todos", np.full(len(yv), yt.mean())),
            ("uma árvore sem limite", a.prever(arvore, fv)),
            ("regressão logística", a.nota_logistica(logistica, xv))]
    print(f"{'referência':<34}{'validação':>10}{'Brier':>8}")
    for nome, nota in refs:
        print(f"{nome:<34}{a.pct(a.acerto_do_par(nota, yv)):>10}{a.num(a.brier(nota, yv), 3):>8}")
    print(f"   a árvore sem limite tem {a.folhas(arvore)} folhas e acerta {a.pct(a.acerto_do_par(a.prever(arvore, ft), yt))} no treino")

    # ---- as florestas: cada configuração com os cinco sorteios
    florestas = {}

    def floresta(colunas, amostra=1.0, min_folha=1, arvores=100):
        chave = (colunas, amostra, min_folha)
        if chave not in florestas or len(florestas[chave][0]["arvores"]) < arvores:
            florestas[chave] = [a.treinar_floresta(ft, yt, arvores, colunas, amostra, min_folha, s) for s in SORTEIOS]
        return florestas[chave]

    def medir(modelos, arvores):
        nv = [a.nota_floresta(m, fv, arvores) for m in modelos]
        return {"validacao": [a.acerto_do_par(n, yv) for n in nv],
                "brier": [a.brier(n, yv) for n in nv],
                "fora": [a.acerto_do_par(a.nota_fora_da_amostra(m, ft, arvores), yt) for m in modelos],
                "notas": np.array(nv),
                "segundos": [sum(m["tempos"][:arvores]) for m in modelos],
                "folhas": [np.mean([a.folhas(x) for x in m["arvores"][:arvores]]) for m in modelos]}

    def guardar(parte, colunas, amostra, min_folha, arvores, r):
        for i, s in enumerate(SORTEIOS):
            linhas_csv.append({"parte": parte, "configuracao": rotulo(colunas, amostra, min_folha, arvores),
                               "arvores": arvores, "colunas": 16 if colunas is None else colunas,
                               "amostra": amostra, "min_folha": min_folha, "sorteio": s,
                               "validacao": round(r["validacao"][i], 6), "fora_da_amostra": round(r["fora"][i], 6),
                               "brier": round(r["brier"][i], 6), "folhas_por_arvore": round(r["folhas"][i], 1),
                               "segundos_treino": round(r["segundos"][i], 4)})

    print()
    print("3. Mais árvores: a floresta muda menos de um sorteio para outro")
    print("   colunas 4 por nó, amostra do tamanho do treino, folha mínima 1; sorteios 0 a 4")
    print(f"{'árvores':<9}{'média':>8}{'menor':>8}{'maior':>8}{'diferença':>11}{'Brier':>8}{'a nota muda':>13}")
    modelos4 = floresta(4, arvores=200)
    tempos_arvores = {}
    for k in ARVORES_PARTE3:
        r = medir(modelos4, k)
        v = np.array(r["validacao"])
        muda = float(r["notas"].std(axis=0).mean())
        print(f"{k:<9}{a.pct(v.mean()):>8}{a.pct(v.min()):>8}{a.pct(v.max()):>8}{a.num(100 * (v.max() - v.min())):>11}"
              f"{a.num(np.mean(r['brier']), 3):>8}{a.num(muda, 3):>13}")
        guardar(3, 4, 1.0, 1, k, r)
        tempos_arvores[k] = float(np.median(r["segundos"]))
    print("   diferença: maior menos menor, em pontos")
    print("   a nota muda: quanto a nota de uma mesma entrega varia entre os 5 sorteios (desvio, média das 1888)")
    exemplo = medir(modelos4, 100)["notas"][:, :3]
    for j in range(3):
        print(f"   entrega {idv[j]}: " + "  ".join(a.num(x, 2) for x in exemplo[:, j]) + f"   atrasou: {'sim' if yv[j] else 'não'}")

    print()
    print("4. Quantas colunas cada nó sorteia: 100 árvores, 5 sorteios")
    print(f"{'colunas por nó':<22}{'média':>8}{'diferença':>11}{'uma árvore':>12}{'parecidas':>11}{'folhas':>8}")
    custo = []
    for col in COLUNAS:
        modelos = floresta(col, arvores=100) if col != 4 else modelos4
        r = medir(modelos, 100)
        v = np.array(r["validacao"])
        uma = np.mean([a.acerto_do_par(a.prever(m["arvores"][0], fv), yv) for m in modelos])
        n20 = a.notas_das_arvores(modelos[0], fv, 20)
        corr = np.corrcoef(n20)
        parecidas = float(corr[np.triu_indices(20, 1)].mean())
        nome = "todas as 16 (bagging)" if col is None else str(col)
        print(f"{nome:<22}{a.pct(v.mean()):>8}{a.num(100 * (v.max() - v.min())):>11}{a.pct(uma):>12}"
              f"{a.num(parecidas, 2):>11}{a.num(np.mean(r['folhas']), 0):>8}")
        if col != 4:
            guardar(4, col, 1.0, 1, 100, r)
        custo.append((f"colunas {16 if col is None else col}, amostra 100%", r["segundos"], r["folhas"]))
    print("   uma árvore: a primeira árvore de cada sorteio, sozinha (média dos 5)")
    print("   parecidas: correlação média entre as notas de 20 árvores do mesmo sorteio, nas entregas de março e abril")

    print()
    print("5. O tamanho da amostra de cada árvore: colunas 4, 100 árvores, 5 sorteios")
    print(f"{'amostra':<22}{'média':>8}{'diferença':>11}{'fora da amostra':>17}{'folhas':>8}")
    for am in AMOSTRAS:
        modelos = modelos4 if am == 1.0 else floresta(4, am, arvores=100)
        r = medir(modelos, 100)
        v = np.array(r["validacao"])
        print(f"{str(int(am * 100)) + '% do treino':<22}{a.pct(v.mean()):>8}{a.num(100 * (v.max() - v.min())):>11}"
              f"{a.pct(np.mean(r['fora'])):>17}{a.num(np.mean(r['folhas']), 0):>8}")
        if am != 1.0:
            guardar(5, 4, am, 1, 100, r)
            custo.append((f"colunas 4, amostra {int(am * 100)}%", r["segundos"], r["folhas"]))
    print("   cada árvore sorteia com reposição essa parte do treino; folhas: média por árvore")

    print()
    print("6. Fora da amostra contra a validação: colunas 4, amostra 100%, média dos 5 sorteios")
    print(f"{'árvores':<9}{'fora da amostra':>17}{'validação':>11}{'entregas com nota':>19}")
    for k in [1, 5, 10, 25, 100, 200]:
        r = medir(modelos4, k)
        cobertas = np.mean([np.isfinite(a.nota_fora_da_amostra(m, ft, k)).sum() for m in modelos4])
        print(f"{k:<9}{a.pct(np.mean(r['fora'])):>17}{a.pct(np.mean(r['validacao'])):>11}{a.num(cobertas, 0):>19}")
    deixadas = np.mean([(np.array(m["pesos"]) == 0).mean() for m in modelos4])
    print(f"   cada árvore deixa de fora, em média, {a.pct(deixadas)} das {len(yt)} entregas do treino")
    print("   fora da amostra: julho a fevereiro, nas árvores que não viram a entrega; validação: março e abril")

    print()
    rastro = a.rastro(args.dados, "python3 floresta.py")
    print("7. O custo nessa máquina: muda de máquina para máquina")
    print(f"   {rastro['sistema']} {rastro['processador']}, {rastro['nucleos']} núcleos, 1 usado; "
          f"Python {rastro['python']}, NumPy {rastro['numpy']}, pandas {rastro['pandas']}")
    if rastro["carga_da_maquina"] is not None:
        print(f"   carga da máquina no fim: {a.num(rastro['carga_da_maquina'], 2)} (média de programas esperando, último minuto)")
    print(f"{'configuração (100 árvores)':<30}{'segundos':>10}{'por árvore':>12}{'nós por árvore':>16}")
    for nome, segundos, folhas_ in custo:
        s = float(np.median(segundos))
        print(f"{nome:<30}{a.num(s, 1):>10}{a.num(s / 100, 3):>12}{a.num(2 * np.mean(folhas_) - 1, 0):>16}")
    print("   segundos: tempo de processador, mediana dos 5 sorteios, só o crescimento das árvores")
    print("   colunas 4, amostra 100%, por número de árvores: "
          + "; ".join(f"{k}: {a.num(tempos_arvores[k], 1)} s" for k in (1, 10, 50, 100, 200)))
    inicio = time.process_time()
    a.nota_floresta(modelos4[0], fv, 100)
    print(f"   dar nota às {len(yv)} entregas de março e abril com 100 árvores: {a.num(time.process_time() - inicio, 2)} s")

    saida.mkdir(parents=True, exist_ok=True)
    with open(saida / "benchmark.csv", "w", encoding="utf-8", newline="") as arq:
        w = csv.DictWriter(arq, fieldnames=list(linhas_csv[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(linhas_csv)
    rastro["sorteios"] = SORTEIOS
    rastro["divisao"] = {"treino": "2025-07-01 a 2026-02-28", "validacao": "2026-03-01 a 2026-04-30",
                         "fechado": "maio e junho de 2026"}
    rastro["linhas_benchmark"] = len(linhas_csv)
    (saida / "benchmark-rastro.json").write_text(json.dumps(rastro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"   salvos: benchmark.csv ({len(linhas_csv)} linhas, uma por configuração e sorteio) e benchmark-rastro.json")
    print(f"   dados: {rastro['dados']}, sha256 {rastro['dados_sha256'][:12]}")


if __name__ == "__main__":
    sys.exit(main())
