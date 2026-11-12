"""Gera os pedidos da Loja Exemplo usados no curso.

Só biblioteca padrão (Python 3.9 ou mais novo). Semente fixa: rodar duas vezes dá os mesmos
arquivos, byte a byte.

    python3 gerar.py                 # escreve nesta pasta
    python3 gerar.py --saida /tmp/x  # escreve em outra pasta

Arquivos:
- pedidos.csv: o que a loja sabe no momento do pedido, mais o rótulo `cancelado` (0 ou 1).
- pedidos_export_completo.csv: os mesmos pedidos com três colunas que outros sistemas
  acrescentam ao export.

Nas aulas, você prepara e avalia primeiro e só depois lê este programa para comparar.
"""
import argparse
import csv
import datetime
import math
import pathlib
import random

SEED = 42
N_PEDIDOS = 2400
INICIO = datetime.date(2026, 1, 1)
FIM = datetime.date(2026, 6, 30)
FRACAO_DEVOLUCAO = 0.04  # cancelados depois de entregues: dias_ate_entrega > 0

CATEGORIAS = [("Eletro", 22, (60, 400)), ("Casa", 24, (20, 250)), ("Papelaria", 18, (5, 120)),
              ("Esporte", 20, (15, 900)), ("Beleza", 16, (12, 300))]
PAGAMENTOS = [("pix", 45), ("cartao", 38), ("boleto", 17)]
EFEITO_CATEGORIA = {"Eletro": 0.25, "Casa": 0.0, "Papelaria": -0.2, "Esporte": 0.35, "Beleza": -0.1}


def _sorteio(rng, opcoes):
    return rng.choices([o[0] for o in opcoes], weights=[o[1] for o in opcoes])[0]


def gerar(saida):
    rng = random.Random(SEED)
    saida.mkdir(parents=True, exist_ok=True)
    dias_periodo = (FIM - INICIO).days
    faixa = {c: f for c, _, f in CATEGORIAS}

    linhas = []
    for i in range(1, N_PEDIDOS + 1):
        data = INICIO + datetime.timedelta(days=rng.randint(0, dias_periodo))
        categoria = _sorteio(rng, CATEGORIAS)
        lo, hi = faixa[categoria]
        valor = round(rng.uniform(lo, hi) * rng.choice([1, 1, 1, 2, 3]), 2)
        pagamento = _sorteio(rng, PAGAMENTOS)
        prazo = rng.randint(2, 15)
        cliente_novo = 1 if rng.random() < 0.35 else 0
        anteriores = 0 if cliente_novo else min(int(rng.expovariate(1 / 4.0)) + 1, 30)
        canal = "app" if rng.random() < 0.55 else "site"

        logito = (-2.35
                  + 1.45 * (pagamento == "boleto")
                  + 0.13 * (prazo - 8)
                  + 0.85 * cliente_novo
                  - 0.09 * min(anteriores, 12)
                  + 0.0016 * (valor - 250)
                  + EFEITO_CATEGORIA[categoria]
                  + 0.05 * (canal == "site")
                  + rng.gauss(0, 0.6))
        cancelado = 1 if rng.random() < 1 / (1 + math.exp(-logito)) else 0

        if cancelado and rng.random() >= FRACAO_DEVOLUCAO:
            dias_entrega = 0
        else:
            dias_entrega = max(1, prazo + rng.randint(-2, 4))
        base_sac = 0.9 if cancelado else 0.55
        contatos_sac = sum(1 for _ in range(4) if rng.random() < base_sac / 2.2)

        linhas.append({
            "id_pedido": i, "data_pedido": data.isoformat(), "categoria": categoria,
            "forma_pagamento": pagamento, "valor": "%.2f" % valor, "prazo_prometido_dias": prazo,
            "cliente_novo": cliente_novo, "pedidos_anteriores": anteriores, "canal": canal,
            "dias_ate_entrega": dias_entrega, "contatos_sac": contatos_sac, "cancelado": cancelado,
        })

    base = ["id_pedido", "data_pedido", "categoria", "forma_pagamento", "valor",
            "prazo_prometido_dias", "cliente_novo", "pedidos_anteriores", "cancelado"]
    completo = ["id_pedido", "data_pedido", "categoria", "forma_pagamento", "valor",
                "prazo_prometido_dias", "cliente_novo", "pedidos_anteriores", "canal",
                "dias_ate_entrega", "contatos_sac", "cancelado"]

    def escrever(nome, colunas):
        with open(saida / nome, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=colunas, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(linhas)

    escrever("pedidos.csv", base)
    escrever("pedidos_export_completo.csv", completo)
    cancelados = sum(l["cancelado"] for l in linhas)
    return {
        "pedidos": len(linhas), "cancelados": cancelados,
        "cancelados_com_entrega": sum(1 for l in linhas if l["cancelado"] and l["dias_ate_entrega"] > 0),
        "entregues_com_dias_zero": sum(1 for l in linhas if not l["cancelado"] and l["dias_ate_entrega"] == 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Gera os CSVs de pedidos da Loja Exemplo.")
    parser.add_argument("--saida", default=str(pathlib.Path(__file__).resolve().parent),
                        help="pasta de saída (padrão: esta pasta)")
    args = parser.parse_args()
    resumo = gerar(pathlib.Path(args.saida))
    print(" ".join("%s=%s" % (k, v) for k, v in resumo.items()))


if __name__ == "__main__":
    main()
