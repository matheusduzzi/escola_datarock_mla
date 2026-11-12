"""Gera as entregas da Loja Exemplo usadas no curso.

Só biblioteca padrão (Python 3.9 ou mais novo). Semente fixa: rodar duas vezes dá o mesmo
arquivo, byte a byte.

    python3 gerar.py                 # escreve nesta pasta
    python3 gerar.py --saida /tmp/x  # escreve em outra pasta

Arquivo:
- entregas.csv: o que a loja sabe na hora do despacho, mais o rótulo `situacao`
  (no_prazo, atraso_curto, atraso_longo ou extravio).

Nas aulas, você treina e avalia primeiro e só depois lê este programa para comparar.
"""
import argparse
import csv
import datetime
import math
import pathlib
import random

SEED = 2611
N_ENTREGAS = 12000
INICIO = datetime.date(2025, 7, 1)
FIM = datetime.date(2026, 6, 30)

REGIOES = [("Sudeste", 45, (15, 900)), ("Sul", 18, (300, 1400)), ("Centro-Oeste", 9, (700, 1700)),
           ("Nordeste", 20, (1400, 2800)), ("Norte", 8, (2200, 3600))]
# A frota própria só atende o Sudeste.
TRANSPORTADORAS = [("Rapida", 35), ("Economica", 45), ("Propria", 20)]
EFEITO_TRANSPORTADORA = {"Rapida": -0.3, "Economica": 0.35, "Propria": -0.5}
FERIADOS = [datetime.date(2025, 9, 7), datetime.date(2025, 10, 12), datetime.date(2025, 11, 2),
            datetime.date(2025, 11, 15), datetime.date(2025, 11, 20), datetime.date(2025, 12, 25),
            datetime.date(2026, 1, 1), datetime.date(2026, 4, 3), datetime.date(2026, 4, 21),
            datetime.date(2026, 5, 1)]
PICO = (datetime.date(2025, 11, 20), datetime.date(2025, 12, 24))


def _sorteio(rng, opcoes):
    return rng.choices([o[0] for o in opcoes], weights=[o[1] for o in opcoes])[0]


def _sigmoide(x):
    return 1 / (1 + math.exp(-x))


def _datas(rng):
    dias = [INICIO + datetime.timedelta(days=d) for d in range((FIM - INICIO).days + 1)]
    pesos = [1.6 if PICO[0] <= d <= PICO[1] else (0.6 if d.weekday() == 6 else 1.0) for d in dias]
    return sorted(rng.choices(dias, weights=pesos, k=N_ENTREGAS))


def gerar(saida):
    rng = random.Random(SEED)
    saida.mkdir(parents=True, exist_ok=True)
    faixa = {r: f for r, _, f in REGIOES}
    linhas = []
    for i, data in enumerate(_datas(rng), start=1):
        regiao = _sorteio(rng, REGIOES)
        lo, hi = faixa[regiao]
        distancia = round(lo + (hi - lo) * rng.random() ** 1.3, 1)
        if regiao == "Sudeste":
            transportadora = _sorteio(rng, TRANSPORTADORAS)
        else:
            transportadora = _sorteio(rng, TRANSPORTADORAS[:2])
        peso = round(min(40.0, max(0.1, math.exp(rng.gauss(0.9, 1.0)))), 1)
        itens = max(1, int(round(peso / 2.2 + rng.gauss(0.8, 0.9))))
        frete = (8 + 0.012 * distancia + 1.1 * peso) * (1.6 if transportadora == "Rapida" else 1.0)
        frete = round(frete * rng.uniform(0.93, 1.07), 2)
        prazo = int(2 + distancia / 450 + {"Rapida": 0, "Economica": 3, "Propria": -1}[transportadora] + rng.randint(0, 2))
        prazo = max(1, prazo)
        vespera = 1 if any(0 < (f - data).days <= 2 for f in FERIADOS) else 0
        pico = 1 if PICO[0] <= data <= PICO[1] else 0
        cod_rota = rng.randint(1, 400)
        valor_pedido = round(math.exp(rng.gauss(5.0, 0.7)), 2)
        longe = 1 if distancia > 1500 else 0
        pesado = 1 if peso > 10 else 0
        economica = 1 if transportadora == "Economica" else 0
        norte_nordeste = 1 if regiao in ("Norte", "Nordeste") else 0

        logito_extravio = -5.2 + 0.0004 * distancia + 0.7 * economica
        if rng.random() < _sigmoide(logito_extravio):
            situacao = "extravio"
        else:
            logito = (-1.75
                      + 0.00045 * distancia
                      + 0.5 * longe
                      + EFEITO_TRANSPORTADORA[transportadora]
                      + 1.5 * economica * pesado
                      + 0.1 * (transportadora == "Rapida") * pesado
                      + (1.1 if norte_nordeste else 0.3) * vespera
                      + 0.7 * pico
                      + 0.04 * (itens - 2)
                      - 0.08 * (prazo - 6)
                      + rng.gauss(0, 0.5))
            if rng.random() < _sigmoide(logito):
                logito_longo = -1.0 + 0.7 * longe + 1.0 * economica * pesado + 0.6 * vespera * norte_nordeste + 0.4 * pico
                situacao = "atraso_longo" if rng.random() < _sigmoide(logito_longo) else "atraso_curto"
            else:
                situacao = "no_prazo"

        linhas.append({
            "id_entrega": i, "data_despacho": data.isoformat(), "dia_semana_despacho": data.weekday(),
            "regiao": regiao, "distancia_km": "%.1f" % distancia, "transportadora": transportadora,
            "peso_kg": "%.1f" % peso, "itens": itens, "valor_frete": "%.2f" % frete,
            "prazo_prometido_dias": prazo, "vespera_feriado": vespera, "cod_rota": cod_rota,
            "valor_pedido": "%.2f" % valor_pedido, "situacao": situacao,
        })

    colunas = list(linhas[0])
    with open(saida / "entregas.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colunas, lineterminator="\n")
        w.writeheader()
        w.writerows(linhas)
    conta = {s: sum(1 for l in linhas if l["situacao"] == s) for s in ("no_prazo", "atraso_curto", "atraso_longo", "extravio")}
    return {"entregas": len(linhas), **conta}


def main():
    parser = argparse.ArgumentParser(description="Gera o CSV de entregas da Loja Exemplo.")
    parser.add_argument("--saida", default=str(pathlib.Path(__file__).resolve().parent),
                        help="pasta de saída (padrão: esta pasta)")
    args = parser.parse_args()
    resumo = gerar(pathlib.Path(args.saida))
    print(" ".join("%s=%s" % (k, v) for k, v in resumo.items()))


if __name__ == "__main__":
    main()
