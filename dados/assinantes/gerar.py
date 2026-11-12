"""Gera os clientes do Sistema Exemplo usados no case de sucesso do cliente.

Só biblioteca padrão (Python 3.9 ou mais novo). Semente fixa: rodar duas vezes dá os mesmos
arquivos, byte a byte.

    python3 gerar.py                 # escreve nesta pasta
    python3 gerar.py --saida /tmp/x  # escreve em outra pasta

O Sistema Exemplo vende um programa de gestão para pequenas lojas por assinatura mensal.
A cada trimestre, a equipe de sucesso do cliente tira uma foto de quem está ativo.

Arquivos:
- assinantes.csv: uma linha por cliente ativo em cada data de referência, com o que a equipe
  sabe nesse dia, mais o rótulo `cancelou_90d` (0 ou 1).
- assinantes_export_completo.csv: as mesmas linhas com duas colunas que o financeiro
  acrescenta ao export.

Nas aulas, você treina e avalia primeiro e só depois lê este programa para comparar.
"""
import argparse
import csv
import datetime
import math
import pathlib
import random

SEED = 1708
REFERENCIAS = [datetime.date(2025, 7, 1), datetime.date(2025, 10, 1), datetime.date(2026, 1, 1), datetime.date(2026, 4, 1)]
CLIENTES_INICIAIS = 1800
NOVOS_POR_TRIMESTRE = 330
SEGMENTOS = [("moda", 30), ("mercado", 22), ("farmacia", 14), ("pet", 16), ("papelaria", 18)]
PLANOS = [("basico", 58), ("pro", 42)]
# O plano rede começa a ser vendido em janeiro de 2026.
PLANOS_2026 = [("basico", 48), ("pro", 34), ("rede", 18)]
EFEITO_PLANO = {"basico": 0.3, "pro": 0.0, "rede": -0.2}
# Em janeiro de 2026 o preço do plano básico subiu.
AUMENTO_BASICO = datetime.date(2026, 1, 1)
EFEITO_CONTATO = -0.4
MOTIVOS = [("preco", 30), ("trocou de sistema", 25), ("fechou a loja", 15), ("falta de uso", 20), ("suporte", 10)]


def _sigmoide(x):
    return 1 / (1 + math.exp(-x))


def _sorteio(rng, opcoes):
    return rng.choices([o[0] for o in opcoes], weights=[o[1] for o in opcoes])[0]


def _novo_cliente(rng, ident, inicio, ref):
    plano = _sorteio(rng, PLANOS_2026 if ref >= AUMENTO_BASICO else PLANOS)
    return {"id_cliente": ident, "inicio": inicio, "plano": plano, "segmento": _sorteio(rng, SEGMENTOS),
            "usuarios": {"basico": 1, "pro": rng.randint(2, 5), "rede": rng.randint(5, 15)}[plano],
            "engajamento": rng.gauss(0, 1), "logins_antes": None}


def gerar(saida):
    rng = random.Random(SEED)
    saida.mkdir(parents=True, exist_ok=True)
    ativos = []
    for i in range(1, CLIENTES_INICIAIS + 1):
        inicio = REFERENCIAS[0] - datetime.timedelta(days=rng.randint(30, 1800))
        ativos.append(_novo_cliente(rng, i, inicio, inicio))
    proximo_id = CLIENTES_INICIAIS + 1
    linhas = []
    for ref in REFERENCIAS:
        ficam = []
        for c in ativos:
            c["engajamento"] = 0.8 * c["engajamento"] + rng.gauss(0, 0.6)
            meses = max(1, (ref - c["inicio"]).days // 30)
            logins = max(0, int(round(c["usuarios"] * 18 * math.exp(0.45 * c["engajamento"]) * rng.uniform(0.8, 1.2))))
            antes = c["logins_antes"] if c["logins_antes"] is not None else max(1, int(logins * rng.uniform(0.85, 1.2)))
            variacao = round(100 * (logins - antes) / max(antes, 1), 1)
            c["logins_antes"] = logins
            chamados = sum(1 for _ in range(6) if rng.random() < (0.12 + 0.08 * (c["engajamento"] < -0.8)))
            atraso = 0 if rng.random() < 0.86 else rng.randint(3, 45)
            respondeu = rng.random() < (0.62 if c["engajamento"] > -0.5 else 0.35)
            nps = max(0, min(10, int(round(7.2 + 1.3 * c["engajamento"] + rng.gauss(0, 1.6))))) if respondeu else None
            desconto = 1 if rng.random() < 0.15 else 0
            # A equipe liga para quem os sinais dela apontam como risco.
            regra_equipe = -3.4 + 2.4 * (variacao <= -30) + 1.6 * (atraso > 10) + 1.3 * (chamados >= 3) + 0.9 * (nps is not None and nps <= 6)
            contato = 1 if rng.random() < _sigmoide(regra_equipe) else 0
            logito = (-2.2
                      - 0.55 * c["engajamento"]
                      + 0.8 * (variacao <= -30)
                      + 0.035 * atraso
                      + 0.5 * (chamados >= 3)
                      + (0.6 if nps is None else -0.15 * (nps - 7))
                      - 0.02 * min(meses, 36)
                      + EFEITO_PLANO[c["plano"]]
                      + 0.55 * (c["plano"] == "basico" and ref >= AUMENTO_BASICO)
                      + EFEITO_CONTATO * contato
                      + rng.gauss(0, 0.4))
            cancelou = 1 if rng.random() < _sigmoide(logito) else 0
            if cancelou:
                data_canc = (ref + datetime.timedelta(days=rng.randint(1, 90))).isoformat()
                motivo = _sorteio(rng, MOTIVOS)
            else:
                data_canc, motivo = "", ""
            linhas.append({
                "id_cliente": c["id_cliente"], "data_referencia": ref.isoformat(), "plano": c["plano"],
                "segmento": c["segmento"], "meses_de_casa": meses, "usuarios_contratados": c["usuarios"],
                "logins_30d": logins, "variacao_logins_pct": "%.1f" % variacao, "chamados_30d": chamados,
                "nps_ultimo": "" if nps is None else nps, "atraso_pagamento_dias": atraso,
                "desconto_ativo": desconto, "recebeu_contato_30d": contato, "cancelou_90d": cancelou,
                "data_cancelamento": data_canc, "motivo_cancelamento": motivo,
            })
            if not cancelou:
                ficam.append(c)
        ativos = ficam
        for _ in range(NOVOS_POR_TRIMESTRE):
            inicio = ref + datetime.timedelta(days=rng.randint(1, 90))
            ativos.append(_novo_cliente(rng, proximo_id, inicio, inicio))
            proximo_id += 1

    base = ["id_cliente", "data_referencia", "plano", "segmento", "meses_de_casa", "usuarios_contratados",
            "logins_30d", "variacao_logins_pct", "chamados_30d", "nps_ultimo", "atraso_pagamento_dias",
            "desconto_ativo", "recebeu_contato_30d", "cancelou_90d"]
    completo = base[:-1] + ["data_cancelamento", "motivo_cancelamento", "cancelou_90d"]

    def escrever(nome, colunas):
        with open(saida / nome, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=colunas, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(linhas)

    escrever("assinantes.csv", base)
    escrever("assinantes_export_completo.csv", completo)
    return {"linhas": len(linhas), "clientes": len({l["id_cliente"] for l in linhas}),
            "cancelaram": sum(l["cancelou_90d"] for l in linhas),
            "contatados": sum(l["recebeu_contato_30d"] for l in linhas)}


def main():
    parser = argparse.ArgumentParser(description="Gera os CSVs de clientes do Sistema Exemplo.")
    parser.add_argument("--saida", default=str(pathlib.Path(__file__).resolve().parent),
                        help="pasta de saída (padrão: esta pasta)")
    args = parser.parse_args()
    resumo = gerar(pathlib.Path(args.saida))
    print(" ".join("%s=%s" % (k, v) for k, v in resumo.items()))


if __name__ == "__main__":
    main()
