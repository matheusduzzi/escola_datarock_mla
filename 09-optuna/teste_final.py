"""Abre maio e junho uma vez, para a configuração que o estudo exportado decidiu, e grava o recibo.

    python3 teste_final.py estudos/plano-estudo.json

Antes de ler maio e junho, ele confere o estudo exportado: o estudo terminou, a tabela das tentativas,
o espaço, o código e a planilha são os mesmos do estudo. Qualquer diferença para tudo sem abrir o teste.

Depois treina a candidata e a configuração de hoje com as entregas de julho de 2025 a abril de 2026 e mede
as duas em maio e junho. O recibo fica em estudos/teste-final.json. Se ele já existe, o teste não abre de novo.
"""
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
AULA08 = AQUI.parent / "08-busca-de-hiperparametros"
if not (AULA08 / "busca.py").is_file():
    sys.exit(f"não achei {AULA08.name}/busca.py. A pasta da aula 08 fica ao lado desta. Baixe o kit inteiro.")
sys.path.insert(0, str(AULA08))

import busca  # noqa: E402  (fixa uma thread para o boosting antes do scikit-learn)
from busca import ATUAL, NOMES, matriz, pct, num, texto_config  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RECIBO = AQUI / "estudos" / "teste-final.json"
FIM_VALIDACAO = pd.Timestamp("2026-04-30")


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def parar(msg):
    sys.exit(f"o teste não abriu: {msg}")


def conferir_estudo(arq, dados):
    if not arq.is_file():
        parar(f"não achei {arq}. Exporte antes: python3 estudo.py exportar --nome <nome>")
    try:
        reg = json.loads(arq.read_text(encoding="utf-8"))
        hashes, plano, decisao = reg["sha256"], reg["plano"], reg["decisao"]
        candidata = {n: decisao["candidata"][n] for n in NOMES}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        parar(f"{arq.name} não é um estudo exportado pelo estudo.py ({type(e).__name__}: {e})")
    if plano.get("tentativas_ao_mesmo_tempo") != 1 or not isinstance(plano.get("semente"), int):
        parar("o estudo não diz a semente ou rodou tentativas ao mesmo tempo; ele não se repete")
    for nome, esperado in hashes.items():
        caminho = (AQUI.parent / nome) if nome.startswith("08-") else (AQUI / nome)
        if not caminho.is_file():
            parar(f"o estudo cita {nome}, que não existe mais")
        if sha256(caminho) != esperado:
            parar(f"{nome} mudou depois do estudo (sha256 diferente). O teste mede o que o estudo escolheu, "
                  "com o espaço, o código e a tabela que ele usou")
    tabela = AQUI / next(n for n in hashes if n.endswith("-tentativas.csv"))
    with open(tabela, encoding="utf-8") as f:
        linhas = [linha for linha in csv.DictReader(f) if linha["estado"] == "COMPLETE"]
    if len(linhas) < plano["orcamento"] or reg.get("tentativas") != len(linhas):
        parar(f"o estudo tem {len(linhas)} de {plano['orcamento']} tentativas completas")
    if sha256(dados) != reg["dados"]["sha256"]:
        parar("a planilha de entregas mudou depois do estudo (sha256 diferente)")
    # a decisão do JSON precisa sair da tabela pela regra, com a de hoje medida de novo na validação
    hoje = busca.Avaliador(dados).avaliar(ATUAL)
    melhor = max(linhas, key=lambda linha: float(linha["auc"]))
    ganha = float(melhor["auc_marco"]) > hoje["auc_marco"] and float(melhor["auc_abril"]) > hoje["auc_abril"]
    tipo = {n: (int if n in ("max_iter", "max_leaf_nodes", "min_samples_leaf") else float) for n in NOMES}
    pela_regra = {n: tipo[n](melhor[n]) for n in NOMES} if ganha else dict(ATUAL)
    if pela_regra != candidata:
        parar("a candidata do JSON não é a que a regra tira da tabela das tentativas")
    return reg, candidata


def main(argv=None):
    p = argparse.ArgumentParser(description="Abre o teste final (maio e junho) uma vez.")
    p.add_argument("estudo", help="o JSON gravado por python3 estudo.py exportar")
    p.add_argument("--dados", default=str(AULA08 / busca.DADOS_PADRAO))
    a = p.parse_args(argv)

    if RECIBO.exists():
        r = json.loads(RECIBO.read_text(encoding="utf-8"))
        parar(f"o teste final já foi aberto em {r['aberto_em']}, para o estudo '{r['estudo']}' "
              f"(recibo em estudos/{RECIBO.name}). Ele abre uma vez. Os números dele estão no recibo")
    arq = Path(a.estudo)
    if not arq.is_absolute():
        arq = (Path.cwd() / arq).resolve()
    dados = Path(a.dados)
    reg, candidata = conferir_estudo(arq, dados)

    base = pd.read_csv(dados, parse_dates=["data_despacho"])
    treino = base[base["data_despacho"] <= FIM_VALIDACAO]
    teste = base[base["data_despacho"] > FIM_VALIDACAO]
    xt, yt = matriz(treino)
    xs, ys = matriz(teste)
    maio = (teste["data_despacho"].dt.month == 5).to_numpy()

    notas = {}

    def medir(nome, config):
        nota = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **config) \
            .fit(xt, yt).predict_proba(xs)[:, 1]
        notas[nome] = nota
        return {"auc": roc_auc_score(ys, nota), "auc_maio": roc_auc_score(ys[maio], nota[maio]),
                "auc_junho": roc_auc_score(ys[~maio], nota[~maio])}

    mesma = candidata == ATUAL
    resultados = {"hoje": medir("hoje", ATUAL)}
    faixa = None
    if not mesma:
        resultados["candidata"] = medir("candidata", candidata)
        rng = np.random.default_rng(0)       # 1000 sorteios das entregas do teste, com reposição
        difs = []
        for _ in range(1000):
            i = rng.integers(0, len(ys), len(ys))
            difs.append(roc_auc_score(ys[i], notas["candidata"][i]) - roc_auc_score(ys[i], notas["hoje"][i]))
        difs = np.array(difs)
        faixa = {"de": float(np.percentile(difs, 5)), "ate": float(np.percentile(difs, 95)),
                 "hoje_ganha_em": float((difs < 0).mean())}
    recibo = {"aberto_em": datetime.datetime.now().isoformat(timespec="seconds"),
              "estudo": reg["estudo"], "estudo_sha256": sha256(arq),
              "treino": f"{len(treino)} entregas, julho de 2025 a abril de 2026",
              "teste": f"{len(teste)} entregas, maio e junho de 2026",
              "candidata": candidata, "vai_para_o_teste": reg["decisao"]["vai_para_o_teste"],
              "resultados": {k: {m: round(v, 6) for m, v in r.items()} for k, r in resultados.items()},
              "diferenca_em_1000_reamostras": None if faixa is None else {k: round(v, 6) for k, v in faixa.items()}}
    RECIBO.parent.mkdir(exist_ok=True)
    RECIBO.write_text(json.dumps(recibo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"estudo '{reg['estudo']}' conferido: {reg['tentativas']} tentativas, espaço, código, tabela e planilha iguais")
    print(f"treino: {len(treino)} entregas de julho a abril; teste: {len(teste)} entregas de maio e junho")
    for nome, r in resultados.items():
        rotulo = "configuração de hoje" if nome == "hoje" else "candidata do estudo"
        cfg = ATUAL if nome == "hoje" else candidata
        print(f"{rotulo}: {pct(r['auc'])} (maio {pct(r['auc_maio'])}, junho {pct(r['auc_junho'])})  {texto_config(cfg)}")
    if not mesma:
        dif = 100 * (resultados["candidata"]["auc"] - resultados["hoje"]["auc"])
        print(f"diferença no teste: {num(dif)} ponto (candidata menos hoje)")
        print(f"em 1000 reamostras das {len(ys)} entregas do teste, a diferença anda de {num(100 * faixa['de'])} "
              f"a {num(100 * faixa['ate'])} ponto (90% delas); a de hoje ganha em {pct(faixa['hoje_ganha_em'], 0)}")
    print(f"recibo gravado em estudos/{RECIBO.name}. O teste não abre de novo")


if __name__ == "__main__":
    main()
