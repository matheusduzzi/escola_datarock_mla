# Kit do curso ML avançado

Os arquivos que você usa nas aulas: o diagnóstico, os pedidos e as entregas da Loja Exemplo e os clientes do Sistema Exemplo, do case.

## Como o kit está organizado

| Pasta | O que tem |
|---|---|
| `00-apresentacao/` | O [diagnóstico](00-apresentacao/diagnostico.md), os dados e programas das tarefas, a [folha de respostas](00-apresentacao/minhas-respostas.md), a [folha da ordem de estudo](00-apresentacao/minha-ordem.md) e a `primeira_olhada.py`, que a aula roda nas entregas. |
| `dados/loja/` | 2400 pedidos com o rótulo de cancelamento, os mesmos de ML supervisionado e de ML: preparação e validação. Ver [dados/loja/LEIA-ME.md](dados/loja/LEIA-ME.md). |
| `dados/entregas/` | 12000 entregas despachadas em um ano, com a situação de cada uma: no prazo, atraso curto, atraso longo ou extravio. Ver [dados/entregas/LEIA-ME.md](dados/entregas/LEIA-ME.md). |
| `dados/assinantes/` | Os clientes do Sistema Exemplo em quatro trimestres, com o rótulo de cancelamento, para o case. Ver [dados/assinantes/LEIA-ME.md](dados/assinantes/LEIA-ME.md). |
| `NN-titulo/` | Uma pasta por aula, com o número e o título da aula. Elas entram no kit junto com as aulas. |

Os comandos das aulas rodam de dentro da pasta da aula. Quando uma aula usa uma base compartilhada, ela diz o caminho, por exemplo `../dados/entregas/entregas.csv`.

## Do que você precisa

- Python 3.9 ou mais novo, com o pandas e o NumPy. É o mesmo ambiente de ML supervisionado; se você ainda não preparou o seu, a aula 38. Dependências e ambientes, de Python para dados, mostra como.
- Um editor de código e um terminal.

As bibliotecas de modelos, de experimentos e de explicação entram na primeira aula que usa cada uma. Essa aula traz a versão, o Python mínimo e o comando de instalação.

Na pasta do kit, instale o que o diagnóstico usa:

```bash
python3 -m pip install -r requirements.txt
```

E veja se funcionou:

```bash
cd 00-apresentacao
python3 tarefa3.py
```

A saída tem três linhas. A primeira é `pedidos usados na taxa: 1200`; as outras duas são a tarefa 3 do diagnóstico. Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

Para gerar de novo as bases compartilhadas, rode `python3 gerar.py` dentro de cada pasta de `dados/`. As saídas são:

| Pasta | Saída de `python3 gerar.py` |
|---|---|
| `dados/loja/` | `pedidos=2400 cancelados=467 cancelados_com_entrega=22 entregues_com_dias_zero=0` |
| `dados/entregas/` | `entregas=12000 no_prazo=8273 atraso_curto=2220 atraso_longo=1366 extravio=141` |
| `dados/assinantes/` | `linhas=7499 clientes=2790 cancelaram=1236 contatados=623` |

Os geradores só usam a biblioteca padrão e saem iguais toda vez.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.

## Licença

A licença de uso e redistribuição do kit ainda vai ser definida pela Escola Datarock. Até lá, use os arquivos para estudar o curso. A Loja Exemplo e o Sistema Exemplo são inventados, e todos os dados foram gerados pelos programas do próprio kit.

## Aulas com arquivos neste kit

| Pasta | Aula |
|---|---|
| [00-apresentacao/](00-apresentacao/) | Apresentação |
| [01-arvore-gini-entropia-e-poda/](01-arvore-gini-entropia-e-poda/) | Árvore, Gini, entropia e poda |
| [02-arvore-na-loja-exemplo/](02-arvore-na-loja-exemplo/) | Árvore na Loja Exemplo |
| [03-bagging-boosting-e-stacking/](03-bagging-boosting-e-stacking/) | Bagging, boosting e stacking |
| [04-random-forest/](04-random-forest/) | Random Forest |
| [06-run-parametro-metrica-e-artefato/](06-run-parametro-metrica-e-artefato/) | Run, parâmetro, métrica e artefato |
| [07-mlflow-e-registry/](07-mlflow-e-registry/) | MLflow e registry |
| [09-optuna/](09-optuna/) | Optuna |
| [11-importancias-nativa-e-por-permutacao/](11-importancias-nativa-e-por-permutacao/) | Importâncias nativa e por permutação |
| [12-shap/](12-shap/) | SHAP |
| [13-shap-e-lime/](13-shap-e-lime/) | SHAP e LIME |
| [14-pdp-e-ice/](14-pdp-e-ice/) | PDP e ICE |
| [15-multiclasse/](15-multiclasse/) | Multiclasse |
| [16-multiclasse-na-loja-exemplo/](16-multiclasse-na-loja-exemplo/) | Multiclasse na Loja Exemplo |
