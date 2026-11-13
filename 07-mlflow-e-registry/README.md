# MLflow e registry

A pasta dessa aula no kit do curso ML avançado. Ela usa o boosting e a planilha corrigida da aula 06, que ficam na pasta `06-run-parametro-metrica-e-artefato/`, ao lado desta.

| Arquivo | O que é |
|---|---|
| `lista_mlflow.py` | Grava a run da lista da manhã no MLflow, repete, registra versões, move o alias `candidata`, avalia e compara |
| `modelo_lista.py` | O modelo no formato que o MLflow guarda: lê as árvores e dá a nota de cada entrega |
| `caio_notebook.py` | As células do notebook do Caio, usadas no exercício |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |
| `requirements.txt` | As versões usadas na aula |

## Como rodar

Essa aula precisa do Python 3.10 ou mais novo. O MLflow 3.18.0 não instala no Python 3.9 do diagnóstico do curso. Se o seu `python3` for o 3.9, instale um Python mais novo e use ele nessa pasta.

Eu sugiro um ambiente só para o curso, dentro dessa pasta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

No Windows, troque `python3` por `py` e a segunda linha por `.venv\Scripts\activate`.

Depois, de dentro dessa pasta:

```bash
python3 lista_mlflow.py registrar --nome paula --semente 2
```

Leva de 10 a 20 segundos: o programa treina 11 vezes (a run e as sementes de 0 a 9). A linha do acerto termina em `63,51%; Brier 0,2077`. Os outros comandos da aula, na ordem:

```bash
python3 lista_mlflow.py versao paula
python3 lista_mlflow.py repetir paula
python3 lista_mlflow.py candidata 1 --motivo "run repetida, modelo igual byte a byte, contrato completo"
python3 lista_mlflow.py versoes
python3 lista_mlflow.py avaliar models:/lista-da-manha@candidata
python3 lista_mlflow.py comparar paula paula-repeticao
```

O MLflow fica todo nessa pasta: o banco `mlflow.db` e a pasta `mlartifacts/` aparecem no primeiro comando. Sem servidor e sem conta. Para começar do zero, apague os dois.

O id de cada run (os 8 caracteres depois de `id`) muda toda vez. Os números e os hashes, não.

Se quiser ver as runs no navegador, rode `mlflow ui --backend-store-uri sqlite:///mlflow.db` nessa pasta e abra o endereço que ele mostrar. A aula não depende disso: tudo que ela usa sai dos comandos acima.

Se aparecer `No module named 'mlflow'`, o `pip` instalou em outro Python. Rode `python3 -m pip install -r requirements.txt` com o mesmo `python3` que você usa para rodar o programa.

Se aparecer `não achei .../06-run-parametro-metrica-e-artefato/experimento.py`, a pasta da aula 06 não está ao lado desta. Baixe o kit inteiro.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
