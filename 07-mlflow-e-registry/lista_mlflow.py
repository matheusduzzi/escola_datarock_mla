"""A lista da manhã no MLflow: run, dados de referência, modelo com ambiente e versão candidata.

Usa o boosting da aula 06 (`../06-run-parametro-metrica-e-artefato/experimento.py`) e grava tudo num
MLflow local: o banco `mlflow.db` e a pasta `mlartifacts/`, os dois na pasta dessa aula. Sem servidor
e sem conta.

    python3 lista_mlflow.py registrar --nome paula --semente 2
    python3 lista_mlflow.py repetir paula
    python3 lista_mlflow.py versao paula
    python3 lista_mlflow.py candidata 1 --motivo "run repetida, contrato completo"
    python3 lista_mlflow.py versoes
    python3 lista_mlflow.py avaliar models:/lista-da-manha@candidata
    python3 lista_mlflow.py comparar paula paula-repeticao

registrar  treina, mede em março e abril e grava uma run com parâmetros, métricas, a faixa das
           sementes 0 a 9, os hashes do código e dos dados, o treino e a validação como dados de
           referência, as notas de março e abril e o modelo com o ambiente.
repetir    treina de novo com o que a run gravou e diz se o modelo saiu igual, byte a byte.
versao     registra o modelo de uma run como nova versão do modelo `lista-da-manha`.
candidata  aponta o alias `candidata` para uma versão. Antes, olha se a run dela tem tudo que o
           contrato da aula 06 pede e se foi repetida. Se faltar alguma coisa, não mexe no alias.
versoes    mostra as versões, o alias e o que cada uma tem.
avaliar    carrega um modelo pelo MLflow e mede em março e abril.
comparar   mostra o que mudou entre duas runs.

Se a variável MLFLOW_TRACKING_URI estiver definida, o programa usa o MLflow dela.
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.WARNING)

AQUI = Path(__file__).resolve().parent
AULA_06 = AQUI.parent / "06-run-parametro-metrica-e-artefato"
EXPERIMENTO = "lista-da-manha"
MODELO = "lista-da-manha"
ALIAS = "candidata"
SEMENTES_DA_FAIXA = range(0, 10)
PARAMETROS = ("fim_treino", "fim_validacao", "rodadas", "profundidade", "passo", "subamostra",
              "min_folha", "semente")
TAGS_DO_CONTRATO = ("codigo.experimento_sha256", "codigo.lista_mlflow_sha256", "codigo.modelo_lista_sha256",
                    "dados.arquivo", "dados.sha256", "dados.treino_sha256", "dados.validacao_sha256",
                    "artefato.modelo_json_sha256", "regra_de_repetir")
METRICAS_DO_CONTRATO = ("acerto_do_par_validacao", "brier_validacao", "faixa_sementes_min",
                        "faixa_sementes_max", "faixa_sementes_media")
REGRA = "mesmo ambiente: mesmo sha256 do modelo.json; outro ambiente: mesmo acerto do par com duas casas"


def carregar_aula_06():
    arquivo = AULA_06 / "experimento.py"
    if not arquivo.is_file():
        sys.exit(f"não achei {arquivo}. Essa aula usa a pasta da aula 06 do kit, ao lado desta.")
    spec = importlib.util.spec_from_file_location("experimento", arquivo)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


exp = carregar_aula_06()


def conectar():
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{AQUI / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        mlflow.create_experiment(EXPERIMENTO, artifact_location=(AQUI / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENTO)
    return MlflowClient()


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def pct(v):
    return f"{100 * v:.2f}%".replace(".", ",")


def num(v, casas=4):
    return f"{v:.{casas}f}".replace(".", ",")


def caminho_dos_dados(texto):
    """Os caminhos de dados são lidos a partir da pasta da aula 06, como no experimento.py."""
    p = Path(texto)
    return p if p.is_absolute() else (AULA_06 / p).resolve()


def run_por_nome(cliente, nome):
    achadas = cliente.search_runs([mlflow.get_experiment_by_name(EXPERIMENTO).experiment_id],
                                  filter_string=f"attributes.run_name = '{nome}'")
    return achadas[0] if achadas else None


def treinar_e_medir(p):
    caminho = caminho_dos_dados(p["dados"])
    q = dict(p, dados=str(caminho))
    dados, metricas, modelo, notas = exp.experimento(q)
    treino, validacao, _ = exp.ler(caminho, p["fim_treino"], p["fim_validacao"])
    return dados, metricas, modelo, notas, treino, validacao


def gravar_run(nome, p, faixa, tags_extras=None):
    """Treina e grava a run. Retorna (run_id, métricas, sha256 do modelo.json)."""
    dados, metricas, modelo, notas, treino, validacao = treinar_e_medir(p)
    with tempfile.TemporaryDirectory() as tmp:
        arq_modelo = Path(tmp) / "modelo.json"
        arq_modelo.write_text(json.dumps(modelo, indent=1) + "\n", encoding="utf-8")
        arq_notas = Path(tmp) / "notas_validacao.csv"
        notas.to_csv(arq_notas, index=False)
        sha_modelo = sha256(arq_modelo)
        with mlflow.start_run(run_name=nome) as run:
            mlflow.log_params({k: p[k] for k in PARAMETROS})
            mlflow.log_metrics(metricas)
            if faixa:
                mlflow.log_metrics({"faixa_sementes_min": min(faixa), "faixa_sementes_max": max(faixa),
                                    "faixa_sementes_media": round(float(np.mean(faixa)), 6)})
            mlflow.set_tags({
                "codigo.experimento_sha256": sha256(AULA_06 / "experimento.py"),
                "codigo.lista_mlflow_sha256": sha256(Path(__file__)),
                "codigo.modelo_lista_sha256": sha256(AQUI / "modelo_lista.py"),
                "dados.arquivo": p["dados"],
                "dados.sha256": dados["sha256"],
                "dados.treino_sha256": dados["treino"]["sha256"],
                "dados.validacao_sha256": dados["validacao"]["sha256"],
                "artefato.modelo_json_sha256": sha_modelo,
                "faixa_sementes": "0 a 9" if faixa else "não medida",
                "regra_de_repetir": REGRA,
                **(tags_extras or {}),
            })
            for parte, df, contexto in (("treino", treino, "training"), ("validacao", validacao, "validation")):
                ds = mlflow.data.from_pandas(df, source=p["dados"], name=parte, targets="situacao")
                mlflow.log_input(ds, context=contexto)
            mlflow.log_artifact(str(arq_notas))
            exemplo = validacao[exp.NUMERICAS + exp.CATEGORIAS].head(3)
            mlflow.pyfunc.log_model(name="modelo", python_model=str(AQUI / "modelo_lista.py"),
                                    artifacts={"modelo": str(arq_modelo)}, input_example=exemplo)
    return run.info.run_id, metricas, sha_modelo, dados


def parametros_da_linha(a):
    return {"dados": a.dados, "fim_treino": a.fim_treino, "fim_validacao": a.fim_validacao,
            "rodadas": a.rodadas, "profundidade": a.profundidade, "passo": a.passo,
            "subamostra": a.subamostra, "min_folha": a.min_folha, "semente": a.semente}


# ------------------------------------------------------------------ comandos

def registrar(a):
    conectar()
    if run_por_nome(MlflowClient(), a.nome):
        print(f"a run '{a.nome}' já existe no MLflow. Uma run gravada não muda: escolha outro nome.")
        return 1
    p = parametros_da_linha(a)
    faixa = []
    for s in SEMENTES_DA_FAIXA:
        _, m, _, _ = exp.experimento(dict(p, dados=str(caminho_dos_dados(p["dados"])), semente=s))
        faixa.append(m["acerto_do_par_validacao"])
    run_id, m, sha_modelo, dados = gravar_run(a.nome, p, faixa)
    print(f"run '{a.nome}' gravada no MLflow, experimento '{EXPERIMENTO}', id {run_id[:8]}")
    print(f"dados de referência: treino {dados['treino']['entregas']} entregas, "
          f"validação {dados['validacao']['entregas']} entregas, arquivo sha256 {dados['sha256'][:12]}")
    print(f"acerto do par na validação {pct(m['acerto_do_par_validacao'])}; Brier {num(m['brier_validacao'])}")
    print(f"sementes 0 a 9: de {pct(min(faixa))} a {pct(max(faixa))}, média {pct(np.mean(faixa))}")
    print(f"modelo com ambiente gravado; modelo.json sha256 {sha_modelo[:12]}")
    return 0


def repetir(a):
    cliente = conectar()
    original = run_por_nome(cliente, a.nome)
    if original is None:
        print(f"não achei a run '{a.nome}' no MLflow.")
        return 1
    faltam = [k for k in PARAMETROS if k not in original.data.params] + \
             [k for k in ("dados.arquivo", "dados.sha256", "artefato.modelo_json_sha256") if k not in original.data.tags]
    if faltam:
        print(f"a run '{a.nome}' não gravou o que precisa para ser repetida: {', '.join(faltam)}")
        return 1
    t, pr = original.data.tags, original.data.params
    arquivo = t["dados.arquivo"]
    agora = sha256(caminho_dos_dados(arquivo))
    if agora != t["dados.sha256"]:
        print(f"os dados mudaram: a run leu {arquivo} com sha256 {t['dados.sha256'][:12]}, hoje o arquivo tem {agora[:12]}.")
        print("Repetir com outros dados não repete a run.")
        return 1
    p = {"dados": arquivo, "fim_treino": pr["fim_treino"], "fim_validacao": pr["fim_validacao"],
         "rodadas": int(pr["rodadas"]), "profundidade": int(pr["profundidade"]), "passo": float(pr["passo"]),
         "subamostra": float(pr["subamostra"]), "min_folha": int(pr["min_folha"]), "semente": int(pr["semente"])}
    nome = f"{a.nome}-repeticao"
    if run_por_nome(cliente, nome):
        print(f"a run '{nome}' já existe no MLflow. Uma run gravada não muda.")
        return 1
    run_id, m, sha_modelo, _ = gravar_run(nome, p, faixa=None, tags_extras={"repete_run": original.info.run_id})
    igual = sha_modelo == t["artefato.modelo_json_sha256"]
    acerto_antes = original.data.metrics["acerto_do_par_validacao"]
    duas_casas = round(100 * acerto_antes, 2) == round(100 * m["acerto_do_par_validacao"], 2)
    resultado = "igual" if igual else ("mesmo acerto com duas casas" if duas_casas else "diferente")
    cliente.set_tag(run_id, "repeticao", resultado)
    print(f"run '{nome}' gravada, id {run_id[:8]}, repetindo '{a.nome}' ({original.info.run_id[:8]})")
    print(f"modelo.json: antes {t['artefato.modelo_json_sha256'][:12]}, agora {sha_modelo[:12]}")
    print(f"acerto do par na validação: antes {pct(acerto_antes)}, agora {pct(m['acerto_do_par_validacao'])}")
    print(f"resultado: {resultado}")
    return 0 if resultado != "diferente" else 1


def versao(a):
    cliente = conectar()
    run = run_por_nome(cliente, a.nome)
    if run is None:
        print(f"não achei a run '{a.nome}' no MLflow.")
        return 1
    modelos = cliente.search_logged_models(experiment_ids=[run.info.experiment_id],
                                           filter_string=f"source_run_id = '{run.info.run_id}'")
    if not modelos:
        print(f"a run '{a.nome}' não tem modelo gravado.")
        return 1
    mv = mlflow.register_model(f"models:/{modelos[0].model_id}", MODELO)
    print(f"modelo '{MODELO}', versão {mv.version}, da run '{a.nome}' ({run.info.run_id[:8]})")
    return 0


def o_que_falta(cliente, mv):
    """O que a run de uma versão não tem do contrato da aula 06, mais a repetição, por item do contrato."""
    if not mv.run_id:
        return {"run": ["a versão não aponta para nenhuma run"]}
    run = cliente.get_run(mv.run_id)
    contextos = {t.value for e in run.inputs.dataset_inputs for t in e.tags if t.key == "mlflow.data.context"}
    repeticoes = cliente.search_runs([run.info.experiment_id],
                                     filter_string=f"tags.repete_run = '{run.info.run_id}'")
    faltam = {
        "dados de referência": [nome for c, nome in (("training", "treino"), ("validation", "validação"))
                                if c not in contextos],
        "parâmetros": [k for k in PARAMETROS if k not in run.data.params],
        "métricas": [k for k in METRICAS_DO_CONTRATO if k not in run.data.metrics],
        "tags": [k for k in TAGS_DO_CONTRATO if k not in run.data.tags],
        "repetição": [] if any(r.data.tags.get("repeticao") in ("igual", "mesmo acerto com duas casas")
                               for r in repeticoes) else ["nenhuma run repetiu essa com o mesmo resultado"],
    }
    return {k: v for k, v in faltam.items() if v}


def candidata(a):
    cliente = conectar()
    try:
        mv = cliente.get_model_version(MODELO, str(a.versao))
    except mlflow.exceptions.MlflowException:
        print(f"o modelo '{MODELO}' não tem a versão {a.versao}.")
        return 1
    if not a.motivo.strip():
        print("diga o motivo em --motivo: quem move o alias diz por quê.")
        return 1
    faltam = o_que_falta(cliente, mv)
    try:
        antes = str(cliente.get_model_version_by_alias(MODELO, ALIAS).version)
    except mlflow.exceptions.MlflowException:
        antes = None
    if faltam:
        print(f"a versão {a.versao} não pode ser a candidata. Falta:")
        for item, nomes in faltam.items():
            print(f"  {item}: {', '.join(nomes)}")
        print(f"o alias '{ALIAS}' continua {'na versão ' + antes if antes else 'sem versão'}.")
        return 1
    cliente.set_registered_model_alias(MODELO, ALIAS, mv.version)
    cliente.set_model_version_tag(MODELO, mv.version, "motivo_candidata", a.motivo.strip())
    cliente.set_model_version_tag(MODELO, mv.version, "contrato", "completo e repetido")
    de = f"da versão {antes} " if antes and antes != str(mv.version) else ""
    print(f"alias '{ALIAS}' movido {de}para a versão {mv.version} (run {cliente.get_run(mv.run_id).info.run_name})")
    print(f"motivo: {a.motivo.strip()}")
    return 0


def versoes(a):
    cliente = conectar()
    lista = sorted(cliente.search_model_versions(f"name = '{MODELO}'"), key=lambda v: int(v.version))
    if not lista:
        print(f"o modelo '{MODELO}' ainda não tem versões.")
        return 0
    try:
        atual = str(cliente.get_model_version_by_alias(MODELO, ALIAS).version)
    except mlflow.exceptions.MlflowException:
        atual = None
    print(f"modelo '{MODELO}': {len(lista)} versão(ões); alias '{ALIAS}' na versão {atual or '(nenhuma)'}")
    for mv in lista:
        run = cliente.get_run(mv.run_id) if mv.run_id else None
        m = run.data.metrics if run else {}
        acerto = m.get("acerto_do_par_validacao", m.get("acerto"))
        faltam = o_que_falta(cliente, mv)
        print(f"versão {mv.version}: run '{run.info.run_name if run else '?'}'"
              f"{'  [candidata]' if str(mv.version) == atual else ''}")
        print(f"   acerto gravado: {pct(acerto) if acerto is not None else '(não gravado)'}; "
              f"falta: {', '.join(faltam) if faltam else 'nada'}")
        if mv.tags.get("motivo_candidata"):
            print(f"   motivo: {mv.tags['motivo_candidata']}")
    return 0


def avaliar(a):
    conectar()
    modelo = mlflow.pyfunc.load_model(a.uri)
    caminho = caminho_dos_dados(a.dados)
    _, validacao, dados = exp.ler(caminho, a.fim_treino, a.fim_validacao)
    nota = modelo.predict(validacao[exp.NUMERICAS + exp.CATEGORIAS]).to_numpy()
    y = (validacao["situacao"] != "no_prazo").astype(int).to_numpy()
    print(f"modelo {a.uri}: run {modelo.metadata.run_id[:8] if modelo.metadata.run_id else '?'}")
    print(f"dados: {a.dados}, validação sha256 {dados['validacao']['sha256'][:12]}, {len(y)} entregas, {int(y.sum())} atrasaram")
    print(f"acerto do par {pct(round(exp.acerto_do_par(nota, y), 6))}; Brier {num(round(float(np.mean((nota - y) ** 2)), 6))}")
    if modelo.metadata.run_id:
        try:
            pasta = mlflow.artifacts.download_artifacts(run_id=modelo.metadata.run_id, artifact_path="notas_validacao.csv")
            gravadas = pd.read_csv(pasta)
            iguais = int((np.round(nota, 6) == gravadas["nota"].to_numpy()).sum()) if len(gravadas) == len(nota) else 0
            print(f"notas iguais às que a run gravou: {iguais} de {len(nota)}")
        except Exception:
            print("a run desse modelo não gravou as notas de março e abril")
    return 0


def comparar(a):
    cliente = conectar()
    runs = []
    for nome in (a.run_a, a.run_b):
        r = run_por_nome(cliente, nome)
        if r is None:
            print(f"não achei a run '{nome}' no MLflow.")
            return 1
        campos = {f"parametro.{k}": v for k, v in r.data.params.items()}
        campos.update({f"metrica.{k}": v for k, v in r.data.metrics.items()})
        campos.update({f"tag.{k}": v for k, v in r.data.tags.items()
                       if not k.startswith("mlflow.") and k not in ("repete_run", "repeticao")})
        for entrada in r.inputs.dataset_inputs:
            campos[f"dados_de_referencia.{entrada.dataset.name}.digest"] = entrada.dataset.digest
        runs.append(campos)
    ra, rb = runs
    diferentes = [k for k in sorted(set(ra) | set(rb)) if ra.get(k) != rb.get(k)]
    iguais = sum(1 for k in set(ra) & set(rb) if ra[k] == rb[k])
    print(f"comparando {a.run_a} e {a.run_b}: {iguais} campos iguais, {len(diferentes)} diferentes")
    if diferentes:
        largura = max(len(k) for k in diferentes)
        for k in diferentes:
            va, vb = (str(x.get(k, "(não tem)")) for x in (ra, rb))
            va, vb = (v[:12] if len(v) == 64 else (v[:24] + "…" if len(v) > 25 else v) for v in (va, vb))
            print(f"{k:<{largura}}  {va:<14}  {vb}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="A lista da manhã no MLflow.")
    sub = p.add_subparsers(dest="comando", required=True)
    s = sub.add_parser("registrar")
    s.add_argument("--nome", required=True)
    s.add_argument("--semente", type=int, required=True)
    s.add_argument("--dados", default=exp.DADOS_PADRAO)
    s.add_argument("--fim-treino", default="2026-02-28")
    s.add_argument("--fim-validacao", default="2026-04-30")
    s.add_argument("--rodadas", type=int, default=30)
    s.add_argument("--profundidade", type=int, default=3)
    s.add_argument("--passo", type=float, default=0.1)
    s.add_argument("--subamostra", type=float, default=0.5)
    s.add_argument("--min-folha", type=int, default=20)
    sub.add_parser("repetir").add_argument("nome")
    sub.add_parser("versao").add_argument("nome")
    c = sub.add_parser("candidata")
    c.add_argument("versao", type=int)
    c.add_argument("--motivo", required=True)
    sub.add_parser("versoes")
    v = sub.add_parser("avaliar")
    v.add_argument("uri")
    v.add_argument("--dados", default=exp.DADOS_PADRAO)
    v.add_argument("--fim-treino", default="2026-02-28")
    v.add_argument("--fim-validacao", default="2026-04-30")
    k = sub.add_parser("comparar")
    k.add_argument("run_a")
    k.add_argument("run_b")
    a = p.parse_args(argv)
    return {"registrar": registrar, "repetir": repetir, "versao": versao, "candidata": candidata,
            "versoes": versoes, "avaliar": avaliar, "comparar": comparar}[a.comando](a)


if __name__ == "__main__":
    sys.exit(main())
