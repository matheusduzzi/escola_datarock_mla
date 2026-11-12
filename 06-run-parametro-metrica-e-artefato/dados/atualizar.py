"""Monta a planilha de entregas depois da revisão do SAC de 3 de julho de 2026.

Parte do arquivo do curso, `../dados/entregas/entregas.csv`, e troca a situação de 48 entregas de
março e abril de 2026 que constavam como `no_prazo` e que o SAC confirmou como `extravio` ou
`atraso_longo`. O resto do arquivo fica igual, linha por linha.

    python3 dados/atualizar.py            # da pasta da aula; escreve dados/entregas.csv
    python3 dados/atualizar.py --saida x.csv

Só biblioteca padrão. Rodar duas vezes dá o mesmo arquivo, byte a byte.
"""
import argparse
import csv
import pathlib
import random

AQUI = pathlib.Path(__file__).resolve().parent
ORIGEM = AQUI.parent.parent / "dados" / "entregas" / "entregas.csv"
SORTEIO = 703          # fixa quais entregas o SAC revisou, para o arquivo sair sempre igual
CORRIGIDAS = 48


def atualizar(origem, saida):
    with open(origem, newline="", encoding="utf-8") as f:
        leitor = csv.reader(f)
        cabecalho = next(leitor)
        linhas = list(leitor)
    col_data, col_sit = cabecalho.index("data_despacho"), cabecalho.index("situacao")
    candidatas = [i for i, l in enumerate(linhas)
                  if "2026-03-01" <= l[col_data] <= "2026-04-30" and l[col_sit] == "no_prazo"]
    rng = random.Random(SORTEIO)
    escolhidas = sorted(rng.sample(candidatas, CORRIGIDAS))
    for k, i in enumerate(escolhidas):
        linhas[i][col_sit] = "extravio" if k % 3 == 0 else "atraso_longo"
    with open(saida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f, lineterminator="\n")
        escritor.writerow(cabecalho)
        escritor.writerows(linhas)
    return len(linhas), len(escolhidas)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--origem", default=str(ORIGEM))
    p.add_argument("--saida", default=str(AQUI / "entregas.csv"))
    a = p.parse_args()
    total, trocadas = atualizar(a.origem, a.saida)
    print(f"entregas={total} corrigidas_pelo_sac={trocadas}")
