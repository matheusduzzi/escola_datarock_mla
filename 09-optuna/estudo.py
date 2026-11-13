"""O plano de busca da aula 08 rodado no Optuna: estudo semeado, retomável e exportável.

Na pasta da aula «Optuna» do kit, rode por exemplo:

    python3 estudo.py rodar --nome plano --semente 0
    python3 estudo.py repetir --semente 0 --vezes 2
    python3 estudo.py repetir --semente 0 --vezes 2 --jobs 4
    python3 estudo.py rodar --nome interrompido --semente 0 --parar-em 20
    python3 estudo.py rodar --nome interrompido --semente 0
    python3 estudo.py comparar plano interrompido
    python3 estudo.py sementes --de 0 --ate 4
    python3 estudo.py exportar --nome plano

Cada tentativa treina o boosting da aula 08 nas entregas de julho de 2025 a fevereiro de 2026 e mede o
acerto do par em março e abril de 2026. A leitura e a avaliação são as do busca.py da aula 08: maio e
junho saem na leitura e não entram em conta nenhuma. Quem abre maio e junho é o teste_final.py, uma vez.

Cada estudo fica em estudos/<nome>.db (SQLite), com o sampler guardado em estudos/<nome>-sampler.pkl.
"""
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
AULA08 = AQUI.parent / "08-busca-de-hiperparametros"
if not (AULA08 / "busca.py").is_file():
    sys.exit(f"não achei {AULA08.name}/busca.py. A pasta da aula 08 fica ao lado desta. Baixe o kit inteiro.")
sys.path.insert(0, str(AULA08))

# O busca.py fixa uma thread para o boosting antes de importar o scikit-learn.
import busca  # noqa: E402
from busca import ATUAL, NOMES, Avaliador, ler_espaco, pct, num, texto_config  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

ESPACO = AQUI / "espaco.json"
PASTA = AQUI / "estudos"
ORCAMENTO = 40      # o plano da aula 08: 40 tentativas
SORTEADAS = 8       # as 8 primeiras sorteadas, as outras escolhidas pelo TPE
INTEIROS = ("max_iter", "max_leaf_nodes", "min_samples_leaf")


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def sugerir(trial, espaco):
    """Uma configuração sugerida pelo Optuna dentro das faixas do espaco.json."""
    config = {}
    for nome in NOMES:
        regra = espaco[nome]
        if "fixo" in regra:
            config[nome] = regra["fixo"]
        elif "de" not in regra:
            sys.exit(f"'{nome}' precisa de 'fixo' ou de 'de' e 'ate' no espaco.json")
        elif regra.get("inteiro"):
            config[nome] = trial.suggest_int(nome, int(regra["de"]), int(regra["ate"]),
                                             log=regra.get("escala") == "log")
        else:
            config[nome] = trial.suggest_float(nome, float(regra["de"]), float(regra["ate"]),
                                               log=regra.get("escala") == "log")
    return config


def objetivo(av, espaco):
    def f(trial):
        r = av.avaliar(sugerir(trial, espaco))
        trial.set_user_attr("auc_marco", r["auc_marco"])
        trial.set_user_attr("auc_abril", r["auc_abril"])
        trial.set_user_attr("segundos", r["segundos"])
        return r["auc"]
    return f


def novo_sampler(semente, sorteadas):
    return optuna.samplers.TPESampler(seed=semente, n_startup_trials=sorteadas)


def completas(estudo):
    return [t for t in estudo.trials if t.state == optuna.trial.TrialState.COMPLETE]


def config_de(t):
    return {n: t.params.get(n, ATUAL[n]) for n in NOMES}


def resumo(estudo, hoje, total=None):
    ts = completas(estudo)
    ordem = sorted(ts, key=lambda t: -t.value)
    print("as 3 melhores:")
    for t in ordem[:3]:
        print(f"   {pct(t.value)}  tentativa {t.number:>2}  {texto_config(config_de(t))}")
    m = ordem[0]
    dif = 100 * (m.value - hoje["auc"])
    lado = "acima" if dif >= 0 else "abaixo"
    print(f"melhor: {pct(m.value)}, {num(abs(dif))} ponto {lado} da configuração de hoje")


# ------------------------------------------------------------------ comandos

def cmd_rodar(a):
    av = Avaliador(a.dados)
    espaco = ler_espaco(ESPACO)
    PASTA.mkdir(exist_ok=True)
    banco = PASTA / f"{a.nome}.db"
    arq_sampler = PASTA / f"{a.nome}-sampler.pkl"
    existia = banco.exists()
    marca = {"semente": a.semente, "orcamento": a.orcamento, "sorteadas": a.sorteadas,
             "espaco_sha256": sha256(ESPACO)}
    if existia:
        estudo = optuna.load_study(study_name=a.nome, storage=f"sqlite:///{banco}")
        for k, v in marca.items():
            if estudo.user_attrs.get(k) != v:
                sys.exit(f"o estudo '{a.nome}' foi criado com {k} = {estudo.user_attrs.get(k)}, e agora veio {v}. "
                         "Um estudo não muda de plano no meio. Para outro plano, use outro --nome.")
        if arq_sampler.exists() and not a.esquecer_sampler:
            with open(arq_sampler, "rb") as f:
                estudo.sampler = pickle.load(f)
        else:
            estudo.sampler = novo_sampler(a.semente, a.sorteadas)
    else:
        estudo = optuna.create_study(study_name=a.nome, storage=f"sqlite:///{banco}", direction="maximize",
                                     sampler=novo_sampler(a.semente, a.sorteadas))
        for k, v in marca.items():
            estudo.set_user_attr(k, v)
        estudo.set_user_attr("dados_sha256", sha256(av.caminho))
    feitas = len(completas(estudo))
    alvo = min(a.orcamento, a.parar_em) if a.parar_em else a.orcamento
    if feitas >= a.orcamento:
        sys.exit(f"o estudo '{a.nome}' já tem as {a.orcamento} tentativas do plano. Para exportar: "
                 f"python3 estudo.py exportar --nome {a.nome}")
    hoje = av.avaliar(ATUAL)
    print(f"estudo '{a.nome}' em estudos/{banco.name}: " + (f"retomado com {feitas} tentativas feitas" if existia
                                                             else "criado agora"))
    print(f"sampler TPE, semente {a.semente}, {a.sorteadas} sorteadas no começo; orçamento {a.orcamento} tentativas"
          + ("; sampler novo, o guardado foi ignorado" if existia and a.esquecer_sampler else ""))
    print(f"treino: {av.n_treino} entregas até fevereiro; validação: {av.n_validacao} entregas de março e abril")
    print(f"configuração de hoje: acerto do par {pct(hoje['auc'])}")

    def guardar(estudo_, _trial):
        with open(arq_sampler, "wb") as f:
            pickle.dump(estudo_.sampler, f)
        n = len(completas(estudo_))
        if n in (5, 10, 20, 30) and n < alvo:
            print(f"   depois de {n:>2} tentativas, a melhor: {pct(estudo_.best_value)}")

    t0 = time.process_time()
    estudo.optimize(objetivo(av, espaco), n_trials=alvo - feitas, callbacks=[guardar])
    total = time.process_time() - t0
    agora = len(completas(estudo))
    print(f"{agora - feitas} tentativas nessa rodada, em {num(total, 1)} s de processador; "
          f"o estudo tem {agora} de {a.orcamento}")
    resumo(estudo, hoje)
    if agora < a.orcamento:
        print(f"parado antes do fim. Para continuar: python3 estudo.py rodar --nome {a.nome} --semente {a.semente}")


def rodar_em_memoria(av, espaco, semente, jobs, orcamento, sorteadas):
    estudo = optuna.create_study(direction="maximize", sampler=novo_sampler(semente, sorteadas))
    estudo.optimize(objetivo(av, espaco), n_trials=orcamento, n_jobs=jobs)
    return estudo


def sequencia(estudo):
    return [tuple(round(float(t.params[n]), 12) for n in sorted(t.params))
            for t in sorted(completas(estudo), key=lambda t: t.number)]


def cmd_repetir(a):
    av = Avaliador(a.dados)
    espaco = ler_espaco(ESPACO)
    ritmo = "uma tentativa por vez" if a.jobs == 1 else f"{a.jobs} tentativas ao mesmo tempo"
    print(f"semente {a.semente}, {a.vezes} rodadas de {a.orcamento} tentativas, {ritmo}, sem gravar")
    seqs, melhores = [], []
    for i in range(a.vezes):
        e = rodar_em_memoria(av, espaco, a.semente, a.jobs, a.orcamento, a.sorteadas)
        seqs.append(sequencia(e))
        melhores.append(e.best_value)
        print(f"   rodada {i + 1}: melhor {pct(e.best_value)}  {texto_config(config_de(e.best_trial))}")
    for i in range(1, a.vezes):
        iguais = sum(x == y for x, y in zip(seqs[0], seqs[i]))
        print(f"rodada 1 e rodada {i + 1}: {iguais} de {a.orcamento} tentativas com a mesma configuração, na mesma ordem")


def cmd_comparar(a):
    seqs = []
    for nome in (a.a, a.b):
        banco = PASTA / f"{nome}.db"
        if not banco.exists():
            sys.exit(f"não achei estudos/{nome}.db")
        seqs.append(sequencia(optuna.load_study(study_name=nome, storage=f"sqlite:///{banco}")))
    iguais = sum(x == y for x, y in zip(*seqs))
    primeira = next((i for i, (x, y) in enumerate(zip(*seqs)) if x != y), None)
    print(f"{a.a}: {len(seqs[0])} tentativas; {a.b}: {len(seqs[1])} tentativas")
    print(f"{iguais} com a mesma configuração, na mesma ordem"
          + (f"; a primeira diferente é a tentativa {primeira}" if primeira is not None else ""))


def embaralhado(caminho, pasta):
    """Uma cópia da planilha com a situação de maio e junho trocada de lugar entre as entregas."""
    base = pd.read_csv(caminho)
    teste = pd.to_datetime(base["data_despacho"]) > pd.Timestamp(busca.FIM_VALIDACAO)
    base.loc[teste, "situacao"] = np.random.default_rng(0).permutation(base.loc[teste, "situacao"].to_numpy())
    copia = Path(pasta) / "entregas-teste-embaralhado.csv"
    base.to_csv(copia, index=False)
    return copia, int(teste.sum())


def cmd_sementes(a):
    with tempfile.TemporaryDirectory() as pasta:
        if a.embaralhar_teste:
            copia, n = embaralhado(a.dados, pasta)
            print(f"planilha com a situação das {n} entregas de maio e junho embaralhada (cópia temporária)")
            a.dados = str(copia)
        sementes(a)


def sementes(a):
    av = Avaliador(a.dados)
    espaco = ler_espaco(ESPACO)
    hoje = av.avaliar(ATUAL)
    print(f"plano com as sementes {a.de} a {a.ate}, {a.orcamento} tentativas cada, sem gravar")
    print(f"configuração de hoje: {pct(hoje['auc'])} (março {pct(hoje['auc_marco'])}, abril {pct(hoje['auc_abril'])})")
    print(f"{'semente':>7}  {'melhor':>7}  {'março':>7}  {'abril':>7}  {'pela regra':<11}  configuração")
    vals = []
    for s in range(a.de, a.ate + 1):
        e = rodar_em_memoria(av, espaco, s, 1, a.orcamento, a.sorteadas)
        t = e.best_trial
        ma, ab = t.user_attrs["auc_marco"], t.user_attrs["auc_abril"]
        ganha = ma > hoje["auc_marco"] and ab > hoje["auc_abril"]
        vals.append(t.value)
        print(f"{s:>7}  {pct(t.value):>7}  {pct(ma):>7}  {pct(ab):>7}  {'candidata' if ganha else 'fica a de hoje':<11}"
              f"  {texto_config(config_de(t))}")
    v = np.array(vals)
    if len(v) > 1:
        print(f"{len(v)} sementes: a melhor vai de {pct(v.min())} a {pct(v.max())}, mediana {pct(float(np.median(v)))}")


def cmd_exportar(a):
    banco = PASTA / f"{a.nome}.db"
    if not banco.exists():
        sys.exit(f"não achei estudos/{a.nome}.db. Rode antes: python3 estudo.py rodar --nome {a.nome} --semente 0")
    estudo = optuna.load_study(study_name=a.nome, storage=f"sqlite:///{banco}")
    ua = estudo.user_attrs
    ts = completas(estudo)
    if len(ts) < ua["orcamento"]:
        sys.exit(f"o estudo '{a.nome}' tem {len(ts)} de {ua['orcamento']} tentativas. Um estudo pela metade não é "
                 f"exportado. Para terminar: python3 estudo.py rodar --nome {a.nome} --semente {ua['semente']}")
    if ua["espaco_sha256"] != sha256(ESPACO):
        sys.exit("o espaco.json mudou depois que o estudo foi criado. O estudo não é exportado com outro espaço.")
    av = Avaliador(a.dados)
    if ua["dados_sha256"] != sha256(av.caminho):
        sys.exit("a planilha de entregas mudou depois que o estudo foi criado. O estudo não é exportado.")
    hoje = av.avaliar(ATUAL)

    # a tabela das tentativas, sem horários: o mesmo estudo exporta o mesmo arquivo
    campos = ["tentativa", "estado", "auc"] + NOMES + ["auc_marco", "auc_abril", "segundos"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=campos, lineterminator="\n")
    w.writeheader()
    for t in sorted(estudo.trials, key=lambda t: t.number):
        w.writerow({"tentativa": t.number, "estado": t.state.name,
                    "auc": "" if t.value is None else round(t.value, 6),
                    **{n: t.params.get(n, "") for n in NOMES},
                    "auc_marco": round(t.user_attrs.get("auc_marco", float("nan")), 6),
                    "auc_abril": round(t.user_attrs.get("auc_abril", float("nan")), 6),
                    "segundos": round(t.user_attrs.get("segundos", float("nan")), 3)})
    arq_csv = PASTA / f"{a.nome}-tentativas.csv"
    arq_csv.write_text(buf.getvalue(), encoding="utf-8")

    m = estudo.best_trial
    ma, ab = m.user_attrs["auc_marco"], m.user_attrs["auc_abril"]
    ganha_marco, ganha_abril = ma > hoje["auc_marco"], ab > hoje["auc_abril"]
    candidata = config_de(m) if (ganha_marco and ganha_abril) else dict(ATUAL)
    registro = {
        "estudo": a.nome,
        "plano": {"orcamento": ua["orcamento"], "sorteadas": ua["sorteadas"], "semente": ua["semente"],
                  "sampler": f"TPESampler(seed={ua['semente']}, n_startup_trials={ua['sorteadas']})",
                  "tentativas_ao_mesmo_tempo": 1,
                  "regra": "a melhor da validação substitui a de hoje só se ganhar em março e em abril, cada mês"},
        "espaco": ler_espaco(ESPACO) | {},
        "dados": {"arquivo": "06-run-parametro-metrica-e-artefato/dados/entregas.csv",
                  "sha256": ua["dados_sha256"],
                  "treino": f"{av.n_treino} entregas, julho de 2025 a fevereiro de 2026",
                  "validacao": f"{av.n_validacao} entregas, março e abril de 2026",
                  "teste": "maio e junho de 2026, fora do estudo"},
        "sha256": {"espaco.json": sha256(ESPACO), "estudo.py": sha256(__file__),
                   "08-busca-de-hiperparametros/busca.py": sha256(AULA08 / "busca.py"),
                   f"estudos/{arq_csv.name}": sha256(arq_csv)},
        "versoes": {"python": platform.python_version(), "optuna": optuna.__version__,
                    "scikit-learn": sklearn.__version__, "numpy": np.__version__},
        "tentativas": len(ts),
        "hoje": {"config": ATUAL, "auc": round(hoje["auc"], 6), "auc_marco": round(hoje["auc_marco"], 6),
                 "auc_abril": round(hoje["auc_abril"], 6)},
        "melhor": {"tentativa": m.number, "config": config_de(m), "auc": round(m.value, 6),
                   "auc_marco": round(ma, 6), "auc_abril": round(ab, 6)},
        "decisao": {"ganha_em_marco": ganha_marco, "ganha_em_abril": ganha_abril,
                    "vai_para_o_teste": "a melhor do estudo" if candidata != ATUAL else "a configuração de hoje",
                    "candidata": candidata},
    }
    arq_json = PASTA / f"{a.nome}-estudo.json"
    arq_json.write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"estudo '{a.nome}': {len(ts)} tentativas completas, semente {ua['semente']}, Optuna {optuna.__version__}")
    print(f"gravei estudos/{arq_csv.name} ({len(estudo.trials)} linhas) e estudos/{arq_json.name}")
    print(f"a melhor: tentativa {m.number}, {pct(m.value)}; março {pct(ma)} (hoje {pct(hoje['auc_marco'])}), "
          f"abril {pct(ab)} (hoje {pct(hoje['auc_abril'])})")
    print("pela regra: " + ("ganha nos dois meses, vai para o teste como candidata" if candidata != ATUAL
                            else "não ganha nos dois meses, fica a configuração de hoje"))
    print(f"sha256 do estudo exportado: {sha256(arq_json)[:16]}…")


def main(argv=None):
    p = argparse.ArgumentParser(description="O plano de busca da aula 08 no Optuna.")
    p.add_argument("--dados", default=str(AULA08 / busca.DADOS_PADRAO), help="planilha de entregas")
    sub = p.add_subparsers(dest="comando", required=True)

    r = sub.add_parser("rodar", help="cria ou retoma um estudo gravado em estudos/<nome>.db")
    r.add_argument("--nome", required=True)
    r.add_argument("--semente", type=int, required=True)
    r.add_argument("--orcamento", type=int, default=ORCAMENTO)
    r.add_argument("--sorteadas", type=int, default=SORTEADAS)
    r.add_argument("--parar-em", type=int, help="para quando o estudo tiver esse número de tentativas")
    r.add_argument("--esquecer-sampler", action="store_true",
                   help="ao retomar, começa um sampler novo em vez de carregar o guardado")

    rp = sub.add_parser("repetir", help="roda o plano mais de uma vez com a mesma semente, sem gravar")
    rp.add_argument("--semente", type=int, required=True)
    rp.add_argument("--vezes", type=int, default=2)
    rp.add_argument("--jobs", type=int, default=1, help="tentativas ao mesmo tempo")
    rp.add_argument("--orcamento", type=int, default=ORCAMENTO)
    rp.add_argument("--sorteadas", type=int, default=SORTEADAS)

    c = sub.add_parser("comparar", help="compara as configurações de dois estudos gravados, tentativa por tentativa")
    c.add_argument("a")
    c.add_argument("b")

    s = sub.add_parser("sementes", help="roda o plano com várias sementes, sem gravar")
    s.add_argument("--de", type=int, default=0)
    s.add_argument("--ate", type=int, default=4)
    s.add_argument("--orcamento", type=int, default=ORCAMENTO)
    s.add_argument("--sorteadas", type=int, default=SORTEADAS)
    s.add_argument("--embaralhar-teste", action="store_true",
                   help="roda numa cópia da planilha com a situação de maio e junho embaralhada")

    e = sub.add_parser("exportar", help="grava a tabela das tentativas e o registro do estudo, com a decisão")
    e.add_argument("--nome", required=True)

    a = p.parse_args(argv)
    for campo in ("orcamento", "vezes", "jobs"):
        if getattr(a, campo, 1) < 1:
            sys.exit(f"--{campo} precisa ser pelo menos 1")
    if hasattr(a, "sorteadas") and not 1 <= a.sorteadas < a.orcamento:
        sys.exit("--sorteadas precisa ser pelo menos 1 e menor que o --orcamento")
    if getattr(a, "parar_em", None) is not None and a.parar_em < 1:
        sys.exit("--parar-em precisa ser pelo menos 1")
    {"rodar": cmd_rodar, "repetir": cmd_repetir, "comparar": cmd_comparar, "sementes": cmd_sementes,
     "exportar": cmd_exportar}[a.comando](a)


if __name__ == "__main__":
    main()
