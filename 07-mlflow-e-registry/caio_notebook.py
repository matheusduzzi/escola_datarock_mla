"""As células do notebook do Caio, como ele mandou. Material do exercício da aula «MLflow e registry».

    python3 caio_notebook.py

Rode depois da prática da aula, no mesmo MLflow da pasta dessa aula. O programa faz o que o Caio fez
no notebook dele: treina o boosting com a configuração que ele testou na aula 06, grava no MLflow,
registra uma versão nova do `lista-da-manha` e aponta o alias `candidata` para ela.
"""
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")
import mlflow  # noqa: E402

logging.getLogger("mlflow").setLevel(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lista_mlflow as lm  # noqa: E402  (só para achar o MLflow e o boosting da aula 06)

exp = lm.exp

# célula 1: os dados que estavam no notebook
treino, validacao, _ = exp.ler(lm.AULA_06.parent / "dados" / "entregas" / "entregas.csv",
                               "2026-02-28", "2026-04-30")

# célula 2: treinar com o passo menor
categorias = {c: sorted(treino[c].unique()) for c in exp.CATEGORIAS}
xt, yt, colunas = exp.matriz(treino, categorias)
xv, yv, _ = exp.matriz(validacao, categorias)
p = {"rodadas": 60, "profundidade": 3, "passo": 0.05, "subamostra": 0.5, "min_folha": 20, "semente": 2}
modelo = exp.treinar(xt, yt, p)
modelo["colunas"] = colunas
acerto = exp.acerto_do_par(exp.prever(modelo, xv), yv)
print(f"acerto {acerto:.4f}")

# célula 3: mandar para o MLflow e virar a candidata
lm.conectar()
with tempfile.TemporaryDirectory() as tmp:
    arq = Path(tmp) / "modelo.json"
    arq.write_text(json.dumps(modelo, indent=1) + "\n", encoding="utf-8")
    with mlflow.start_run(run_name="notebook-caio"):
        mlflow.log_params({"passo": 0.05, "rodadas": 60})
        mlflow.log_metric("acerto", round(acerto, 6))
        info = mlflow.pyfunc.log_model(name="modelo", python_model=str(lm.AQUI / "modelo_lista.py"),
                                       artifacts={"modelo": str(arq)},
                                       input_example=validacao[exp.NUMERICAS + exp.CATEGORIAS].head(3),
                                       registered_model_name=lm.MODELO)
cliente = mlflow.MlflowClient()
versao = info.registered_model_version
cliente.set_registered_model_alias(lm.MODELO, lm.ALIAS, versao)
print(f"versão {versao} registrada e marcada como {lm.ALIAS}")
