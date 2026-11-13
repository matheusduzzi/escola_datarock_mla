"""O modelo atual: a floresta da aula «Sucesso do cliente», no formato de pacote de candidato.

    python3 candidato.py                    # refaz a escolha em janeiro e o teste de abril, escreve registro.csv
    python3 candidato.py --saida outra/     # escreve registro.csv em outra pasta

As mesmas contas do sucesso.py da aula 17: cinco sorteios da floresta e a regressão logística
na foto de janeiro, a floresta do sorteio 0 escolhida e o teste de abril aberto uma vez.
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
FLORESTA = {"arvores": 100, "colunas": 4, "min_folha": 20}
CONFIGS = [{"modelo": "floresta", "sorteio": s} for s in range(5)] + [{"modelo": "logistica"}]
ESCOLHIDA = 0   # a floresta do sorteio 0: empate em janeiro, desempate pela guarda (aula 17, seção 7)
LISTA = 150


def ler(caminho=DADOS):
    return pd.read_csv(caminho, dtype={"data_referencia": str})


def colunas(base, treino):
    """As 16 colunas do caso.py, com a mediana do NPS e as categorias tiradas só do treino."""
    preparo = c.Preparo(base[treino])
    return pd.DataFrame(preparo.matriz(base), columns=preparo.nomes, index=base.index)


def dividir(base):
    """Treino nas fotos de 2025, escolha na de janeiro de 2026. Abril fica de teste."""
    return base["data_referencia"].isin(c.TREINO).to_numpy(), (base["data_referencia"] == c.VALIDACAO).to_numpy()


def treinar(x, y, config):
    x = np.asarray(x, dtype=float)
    if config["modelo"] == "logistica":
        return {"logistica": c.treinar_logistica(x, np.asarray(y))}
    cortes = c.faixas(x)
    return {"cortes": cortes,
            "floresta": c.treinar_floresta(c.em_faixas(x, cortes), np.asarray(y), sorteio=config["sorteio"], **FLORESTA)}


def nota(modelo, x):
    x = np.asarray(x, dtype=float)
    if "logistica" in modelo:
        return c.nota_logistica(modelo["logistica"], x)
    return c.nota_floresta(modelo["floresta"], c.em_faixas(x, modelo["cortes"]))


def descrever(config):
    return "regressão logística" if config["modelo"] == "logistica" else f"floresta, sorteio {config['sorteio']}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Refaz a escolha e o teste do modelo atual.")
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
        linhas.append({"rodada": i, "modelo": descrever(config), "treino": "fotos 2025-07-01 e 2025-10-01",
                       "avaliado_em": f"validação ({c.VALIDACAO})", "linhas_avaliadas": int(janeiro.sum()),
                       "acerto_do_par": round(c.acerto_do_par(n, y[janeiro]), 6),
                       "dos_150_primeiros": round(c.quantos_na_lista(n, y[janeiro], LISTA), 1),
                       "escolhido": int(i - 1 == ESCOLHIDA)})
    passado = base["data_referencia"].isin(c.TREINO + [c.VALIDACAO]).to_numpy()
    abril = (base["data_referencia"] == c.TESTE).to_numpy()
    xp = colunas(base, passado)
    n = nota(treinar(xp[passado], y[passado], CONFIGS[ESCOLHIDA]), xp[abril])
    fora, _, _ = c.guarda(c.Preparo(base[passado]), base[abril].drop(columns=[ALVO]).reset_index(drop=True))
    topo = c.lista(n, LISTA, fora)
    linhas.append({"rodada": len(linhas) + 1, "modelo": descrever(CONFIGS[ESCOLHIDA]),
                   "treino": "fotos 2025-07-01, 2025-10-01 e 2026-01-01",
                   "avaliado_em": f"teste ({c.TESTE}), sem os separados pela guarda", "linhas_avaliadas": int(abril.sum()),
                   "acerto_do_par": round(c.acerto_do_par(n, y[abril]), 6),
                   "dos_150_primeiros": float(y[abril][topo].sum()), "escolhido": 1})
    registro = pd.DataFrame(linhas)
    for r in registro.itertuples():
        print(f"   rodada {r.rodada}: {r.modelo}, {r.avaliado_em}: acerto do par {c.pct(r.acerto_do_par)}, "
              f"dos 150 primeiros {c.num(r.dos_150_primeiros)}" + ("   <- escolhida" if r.escolhido else ""))
    print(f"   guarda em abril: {int(fora.sum())} clientes separados para revisão à mão")
    Path(args.saida).mkdir(parents=True, exist_ok=True)
    registro.to_csv(Path(args.saida) / "registro.csv", index=False)
    print(f"registro.csv: {len(registro)} rodadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
