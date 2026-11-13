"""Modelo de cancelamento, versão do Diego (candidato para o piloto de julho de 2026).

    python3 candidato.py                    # treina, mede e escreve registro.csv nesta pasta
    python3 candidato.py --saida outra/     # escreve registro.csv em outra pasta

Usa a floresta do caso.py da pasta de cima (o mesmo do time), porque o ambiente não tem
scikit-learn. Tudo sai igual toda vez: a divisão e a floresta têm número de sorteio fixo.
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

DADOS = AQUI.parent.parent / "dados" / "assinantes" / "assinantes_export_completo.csv"
ALVO = "cancelou_90d"
SORTEIO_DIVISAO = 42
SORTEIO_FLORESTA = 0
ARVORES = 100
# as seis configurações que eu testei
CONFIGS = [{"colunas": k, "min_folha": m} for k in (4, 8) for m in (1, 5, 20)]
MODELO = "floresta"
LISTA = 150


def ler(caminho=DADOS):
    return pd.read_csv(caminho, dtype={"data_referencia": str})


def colunas(base, treino=None):
    """As colunas do modelo, para todas as linhas da base.

    Tiro o alvo e as colunas do financeiro que falam do cancelamento. Faço as contas com a base
    inteira, então o treino não muda nada aqui.
    """
    x = base.drop(columns=[ALVO, "data_cancelamento", "motivo_cancelamento", "id_cliente", "data_referencia"],
                  errors="ignore")
    # NPS vazio: preencho com a média
    x["nps_ultimo"] = x["nps_ultimo"].fillna(base["nps_ultimo"].mean())
    # tempo de relacionamento: em quantas fotos o cliente aparece
    x["fotos_do_cliente"] = base.groupby("id_cliente")["id_cliente"].transform("size")
    x = pd.get_dummies(x, columns=["plano", "segmento"], dtype=float)
    return x.astype(float)


def dividir(base):
    """80% das linhas para treino e 20% para teste, ao acaso."""
    rng = np.random.default_rng(SORTEIO_DIVISAO)
    ordem = rng.permutation(len(base))
    teste = np.zeros(len(base), dtype=bool)
    teste[ordem[: len(base) // 5]] = True
    return ~teste, teste


def treinar(x, y, config):
    x = np.asarray(x, dtype=float)
    cortes = c.faixas(x)
    modelo = c.treinar_floresta(c.em_faixas(x, cortes), np.asarray(y), arvores=ARVORES,
                                sorteio=SORTEIO_FLORESTA, **config)
    return {"cortes": cortes, "floresta": modelo}


def nota(modelo, x):
    return c.nota_floresta(modelo["floresta"], c.em_faixas(np.asarray(x, dtype=float), modelo["cortes"]))


def main(argv=None):
    p = argparse.ArgumentParser(description="Treina e mede o candidato do Diego.")
    p.add_argument("--dados", default=str(DADOS))
    p.add_argument("--saida", default=str(AQUI))
    args = p.parse_args(argv)
    base = ler(args.dados)
    sha = hashlib.sha256(Path(args.dados).read_bytes()).hexdigest()
    treino, teste = dividir(base)
    x, y = colunas(base, treino), base[ALVO].to_numpy()
    print(f"dados: {Path(args.dados).name}, sha256 {sha}, {len(base)} linhas")
    print(f"ambiente: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}")
    print(f"colunas ({x.shape[1]}): {', '.join(x.columns)}")
    print(f"treino: {treino.sum()} linhas; teste: {teste.sum()} linhas, {int(y[teste].sum())} cancelaram")
    linhas = []
    for i, config in enumerate(CONFIGS, start=1):
        modelo = treinar(x[treino], y[treino], config)
        n = nota(modelo, x[teste])
        linhas.append({"rodada": i, "modelo": MODELO, "colunas_por_no": config["colunas"],
                       "folha_minima": config["min_folha"], "arvores": ARVORES, "sorteio": SORTEIO_FLORESTA,
                       "treino": "80% das linhas, ao acaso", "avaliado_em": "teste (20% das linhas, ao acaso)",
                       "linhas_avaliadas": int(teste.sum()), "acerto_do_par": round(c.acerto_do_par(n, y[teste]), 6),
                       "dos_150_primeiros": round(c.quantos_na_lista(n, y[teste], LISTA), 1)})
    registro = pd.DataFrame(linhas)
    melhor = int(registro["acerto_do_par"].idxmax())
    registro["escolhido"] = (registro.index == melhor).astype(int)
    for r in registro.itertuples():
        print(f"   rodada {r.rodada}: {r.colunas_por_no} colunas por nó, folha mínima {r.folha_minima:>2}: "
              f"acerto do par {c.pct(r.acerto_do_par)}, dos 150 primeiros {c.num(r.dos_150_primeiros)}"
              + ("   <- escolhida" if r.escolhido else ""))
    Path(args.saida).mkdir(parents=True, exist_ok=True)
    registro.to_csv(Path(args.saida) / "registro.csv", index=False)
    print(f"registro.csv: {len(registro)} rodadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
