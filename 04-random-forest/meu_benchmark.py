"""Exercício da aula «Random Forest»: a folha mínima com um limite de tempo.

Escreva as duas funções marcadas com NotImplementedError. Depois rode, de dentro da pasta da aula:

    python3 testar_meu_benchmark.py
    python3 meu_benchmark.py            # leva cerca de um minuto

O resto do programa já está pronto: ele treina as florestas, mede cada uma, chama as suas
funções e salva meu_benchmark.csv e meu_benchmark-rastro.json.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import arvores as a

DADOS = Path(__file__).resolve().parent.parent / "dados" / "entregas" / "entregas.csv"
MIN_FOLHAS = [1, 5, 20, 50, 100]
SORTEIOS = [0, 1, 2, 3, 4]
OBRIGATORIAS = ["config", "sorteio", "validacao", "fora_da_amostra", "segundos"]


def resumir(linhas):
    """Uma linha por configuração, na ordem em que cada configuração aparece pela primeira vez.

    Cada item de `linhas` é um dicionário com as chaves de OBRIGATORIAS: o nome da
    configuração, o número do sorteio, o acerto do par na validação e fora da amostra (de 0 a 1)
    e os segundos que a floresta levou para crescer.

    Cada linha do resumo é um dicionário com as chaves:
        "config":          o nome da configuração
        "sorteios":        quantos sorteios ela tem
        "media":           a média do acerto na validação
        "menor", "maior":  o menor e o maior acerto na validação
        "diferenca":       maior menos menor
        "fora_da_amostra": a média do acerto fora da amostra
        "segundos":        a mediana dos segundos (np.median)

    Pare com ValueError se:
        - linhas estiver vazia;
        - faltar alguma chave de OBRIGATORIAS em alguma linha;
        - o mesmo sorteio aparecer duas vezes na mesma configuração;
        - alguma configuração tiver menos de 2 sorteios (com um só, não dá para ver a variação);
        - algum acerto estiver fora de 0 a 1, ou algum tempo não for maior que zero.
    """
    raise NotImplementedError("escreva resumir no meu_benchmark.py")


def escolher(resumo, limite_segundos):
    """A configuração para a lista da madrugada.

    1. Ficam só as configurações com "segundos" até limite_segundos.
    2. Entre elas, a melhor é a de maior "media".
    3. Empate: toda configuração que cabe e tem média até "diferenca" da melhor abaixo dela
       está empatada com a melhor (a diferença entre sorteios da melhor ainda cobre essa
       distância). Entre as empatadas, fica a de menos segundos.

    Pare com ValueError se o resumo estiver vazio, se o limite não for maior que zero ou se
    nenhuma configuração couber no limite.
    """
    raise NotImplementedError("escreva escolher no meu_benchmark.py")


def main():
    parser = argparse.ArgumentParser(description="Folha mínima da floresta com limite de tempo.")
    parser.add_argument("--dados", default=str(DADOS))
    parser.add_argument("--saida", default=".")
    args = parser.parse_args()
    saida = Path(args.saida)

    base = a.ler(args.dados)
    _, d = a.separar(base)
    xt, yt, _ = d["treino"]
    xv, yv, _ = d["validacao"]
    cortes = a.faixas(xt)
    ft, fv = a.em_faixas(xt, cortes), a.em_faixas(xv, cortes)

    linhas = []
    for mf in MIN_FOLHAS:
        for s in SORTEIOS:
            m = a.treinar_floresta(ft, yt, arvores=100, colunas=4, amostra=1.0, min_folha=mf, sorteio=s)
            linhas.append({"config": f"folha {mf}", "sorteio": s, "min_folha": mf, "arvores": 100,
                           "colunas": 4, "amostra": 1.0,
                           "validacao": a.acerto_do_par(a.nota_floresta(m, fv), yv),
                           "fora_da_amostra": a.acerto_do_par(a.nota_fora_da_amostra(m, ft), yt),
                           "segundos": sum(m["tempos"])})

    resumo = resumir(linhas)
    print("1. A qualidade: colunas 4, amostra 100%, 100 árvores, sorteios 0 a 4")
    print(f"{'config':<12}{'média':>8}{'menor':>8}{'maior':>8}{'diferença':>11}{'fora da amostra':>17}")
    for r in resumo:
        print(f"{r['config']:<12}{a.pct(r['media']):>8}{a.pct(r['menor']):>8}{a.pct(r['maior']):>8}"
              f"{a.num(100 * r['diferenca']):>11}{a.pct(r['fora_da_amostra']):>17}")
    logistica = a.treinar_logistica(xt, yt)
    print(f"   regressão logística na validação: {a.pct(a.acerto_do_par(a.nota_logistica(logistica, xv), yv))}")

    referencia = next(r for r in resumo if r["config"] == "folha 1")
    limite = referencia["segundos"] / 2
    print()
    print("2. O tempo nessa máquina e o limite: metade do tempo da folha 1")
    print(f"{'config':<12}{'segundos':>10}{'cabe':>7}")
    for r in resumo:
        print(f"{r['config']:<12}{a.num(r['segundos'], 1):>10}{'sim' if r['segundos'] <= limite else 'não':>7}")
    print(f"   limite: {a.num(limite, 1)} s")

    escolhida = escolher(resumo, limite)
    print()
    print(f"3. A escolhida: {escolhida['config']}, {a.pct(escolhida['media'])} na validação "
          f"(de {a.pct(escolhida['menor'])} a {a.pct(escolhida['maior'])} entre os sorteios)")

    saida.mkdir(parents=True, exist_ok=True)
    with open(saida / "meu_benchmark.csv", "w", encoding="utf-8", newline="") as arq:
        w = csv.DictWriter(arq, fieldnames=list(linhas[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(linhas)
    rastro = a.rastro(args.dados, "python3 meu_benchmark.py")
    rastro.update(sorteios=SORTEIOS, min_folhas=MIN_FOLHAS, limite_segundos=limite, escolhida=escolhida["config"])
    (saida / "meu_benchmark-rastro.json").write_text(json.dumps(rastro, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    print(f"   salvos: meu_benchmark.csv ({len(linhas)} linhas) e meu_benchmark-rastro.json "
          f"(dados sha256 {rastro['dados_sha256'][:12]})")


if __name__ == "__main__":
    main()
