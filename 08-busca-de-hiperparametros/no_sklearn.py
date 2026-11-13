"""A busca aleatória da aula com o RandomizedSearchCV do scikit-learn, na mesma divisão por data.

    python3 no_sklearn.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV  # noqa: E402

from busca import AQUI, DADOS_PADRAO, ler, matriz  # noqa: E402

treino, validacao = ler(AQUI / DADOS_PADRAO)          # maio e junho já saíram aqui
xt, yt = matriz(treino)
xv, yv = matriz(validacao)
x, y = np.vstack([xt, xv]), np.concatenate([yt, yv])
divisao = [(np.arange(len(yt)), np.arange(len(yt), len(y)))]   # uma divisão só: treina até fevereiro, mede março e abril

busca = RandomizedSearchCV(
    HistGradientBoostingClassifier(early_stopping=False, random_state=0),
    {"learning_rate": loguniform(0.01, 0.3),
     "max_iter": randint(5, 401),
     "max_leaf_nodes": randint(2, 33),
     "min_samples_leaf": randint(5, 401),
     "l2_regularization": loguniform(0.001, 10)},
    n_iter=27, scoring="roc_auc", cv=divisao, refit=False, random_state=0)
busca.fit(x, y)

print(f"{len(busca.cv_results_['params'])} avaliações; a melhor: {100 * busca.best_score_:.2f}%".replace(".", ","))
for nome, valor in sorted(busca.best_params_.items()):
    print(f"   {nome} = {valor:.4g}" if isinstance(valor, float) else f"   {nome} = {valor}")
