"""Modelo de cancelamento, versão 2 do Diego (depois da revisão da versão 1).

    python3 candidato.py                    # escolhe em janeiro, abre abril uma vez, escreve registro.csv
    python3 candidato.py --saida outra/     # escreve registro.csv em outra pasta

Regressão logística do caso.py da pasta de cima, com as colunas e a guarda do caso.py.
Tudo sai igual toda vez.
"""
import argparse
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
import caso as c  # noqa: E402

DADOS = AQUI.parent.parent / "dados" / "assinantes" / "assinantes.csv"
ALVO = "cancelou_90d"
# a força da penalidade da regressão logística: três valores, escolhidos na foto de janeiro
CONFIGS = [{"penalidade": p} for p in (0.1, 1.0, 10.0)]
MODELO = "logistica"
LISTA = 150


def ler(caminho=DADOS):
    return pd.read_csv(caminho, dtype={"data_referencia": str})


def colunas(base, treino):
    """As colunas do caso.py: o NPS vazio vira coluna, a ligação fica fora.

    A mediana do NPS e as categorias de plano e segmento saem só das linhas de treino.
    """
    preparo = c.Preparo(base[treino])
    return pd.DataFrame(preparo.matriz(base), columns=preparo.nomes, index=base.index)


def dividir(base):
    """Treino nas fotos de 2025, escolha na de janeiro de 2026. Abril fica de teste."""
    return base["data_referencia"].isin(c.TREINO).to_numpy(), (base["data_referencia"] == c.VALIDACAO).to_numpy()


def treinar(x, y, config):
    return c.treinar_logistica(np.asarray(x, dtype=float), np.asarray(y), **config)


def nota(modelo, x):
    return c.nota_logistica(modelo, np.asarray(x, dtype=float))


def main(argv=None):
    p = argparse.ArgumentParser(description="Treina e mede a versão 2 do candidato do Diego.")
    p.add_argument("--dados", default=str(DADOS))
    p.add_argument("--saida", default=str(AQUI))
    args = p.parse_args(argv)
    base = ler(args.dados)
    sha = hashlib.sha256(Path(args.dados).read_bytes()).hexdigest()
    y = base[ALVO].to_numpy()
    treino, janeiro = dividir(base)
    x = colunas(base, treino)
    print(f"dados: {Path(args.dados).name}, sha256 {sha}, {len(base)} linhas")
    print(f"ambiente: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}")
    print(f"colunas ({x.shape[1]}): {', '.join(x.columns)}")
    linhas = []
    for i, config in enumerate(CONFIGS, start=1):
        n = nota(treinar(x[treino], y[treino], config), x[janeiro])
        linhas.append({"rodada": i, "modelo": MODELO, "penalidade": config["penalidade"],
                       "treino": "fotos 2025-07-01 e 2025-10-01", "avaliado_em": f"validação ({c.VALIDACAO})",
                       "linhas_avaliadas": int(janeiro.sum()), "acerto_do_par": round(c.acerto_do_par(n, y[janeiro]), 6),
                       "dos_150_primeiros": round(c.quantos_na_lista(n, y[janeiro], LISTA), 1), "escolhido": 0})
    melhor = max(range(len(linhas)), key=lambda i: (linhas[i]["dos_150_primeiros"], linhas[i]["acerto_do_par"]))
    linhas[melhor]["escolhido"] = 1
    escolhida = CONFIGS[melhor]
    # a versão entregue: a escolhida, treinada nas três fotos com rótulo; abril aberto uma vez
    passado = base["data_referencia"].isin(c.TREINO + [c.VALIDACAO]).to_numpy()
    abril = (base["data_referencia"] == c.TESTE).to_numpy()
    xp = colunas(base, passado)
    n = nota(treinar(xp[passado], y[passado], escolhida), xp[abril])
    fora, _, _ = c.guarda(c.Preparo(base[passado]), base[abril].drop(columns=[ALVO]).reset_index(drop=True))
    topo = c.lista(n, LISTA, fora)
    linhas.append({"rodada": len(linhas) + 1, "modelo": MODELO, "penalidade": escolhida["penalidade"],
                   "treino": "fotos 2025-07-01, 2025-10-01 e 2026-01-01", "avaliado_em": f"teste ({c.TESTE}), sem os separados pela guarda",
                   "linhas_avaliadas": int(abril.sum()), "acerto_do_par": round(c.acerto_do_par(n, y[abril]), 6),
                   "dos_150_primeiros": float(y[abril][topo].sum()), "escolhido": 1})
    registro = pd.DataFrame(linhas)
    for r in registro.itertuples():
        print(f"   rodada {r.rodada}: penalidade {c.num(r.penalidade)}, {r.avaliado_em}: acerto do par "
              f"{c.pct(r.acerto_do_par)}, dos 150 primeiros {c.num(r.dos_150_primeiros)}"
              + ("   <- escolhida" if r.escolhido else ""))
    print(f"   guarda em abril: {int(fora.sum())} clientes separados para revisão à mão")
    Path(args.saida).mkdir(parents=True, exist_ok=True)
    registro.to_csv(Path(args.saida) / "registro.csv", index=False)
    print(f"registro.csv: {len(registro)} rodadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
