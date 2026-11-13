"""O modelo da lista da manhã, no formato que o MLflow guarda (models from code).

O MLflow guarda este arquivo junto com o `modelo.json` (as árvores da aula 06). Quem carrega o modelo
pelo MLflow roda este código: ele lê as árvores e dá a nota de cada entrega, de 0 a 1.

Só usa pandas e NumPy. Não depende do `experimento.py`: o modelo registrado funciona sozinho.
"""
import json

import mlflow
import numpy as np
import pandas as pd


def _nota_da_arvore(no, x):
    if "valor" in no:
        return np.full(len(x), no["valor"])
    passa = x[:, no["coluna"]] > no["corte"]
    saida = np.empty(len(x))
    saida[passa] = _nota_da_arvore(no["sim"], x[passa])
    saida[~passa] = _nota_da_arvore(no["nao"], x[~passa])
    return saida


class ListaDaManha(mlflow.pyfunc.PythonModel):
    """Recebe as colunas de uma entrega na hora do despacho e retorna a nota de atraso."""

    def load_context(self, context):
        with open(context.artifacts["modelo"], encoding="utf-8") as f:
            self.modelo = json.load(f)

    def _matriz(self, entregas):
        colunas = []
        for nome in self.modelo["colunas"]:
            if "=" in nome:                       # coluna de categoria, como "transportadora=Rapida"
                coluna, valor = nome.split("=", 1)
                colunas.append((entregas[coluna].astype(str) == valor).astype(float).to_numpy())
            else:
                colunas.append(entregas[nome].astype(float).to_numpy())
        return np.column_stack(colunas)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.Series:
        x = self._matriz(model_input)
        soma = np.full(len(x), self.modelo["base"])
        for arvore in self.modelo["arvores"]:
            soma += self.modelo["passo"] * _nota_da_arvore(arvore, x)
        return pd.Series(1 / (1 + np.exp(-soma)), name="nota")


mlflow.models.set_model(ListaDaManha())
