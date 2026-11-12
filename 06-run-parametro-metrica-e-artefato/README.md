# Run, parâmetro, métrica e artefato

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `experimento.py` | O boosting da aula. Grava cada run em `runs/<nome>/`, roda várias sementes e compara duas runs |
| `dados/entregas.csv` | As entregas depois da correção do SAC de 3 de julho de 2026 |
| `dados/atualizar.py` | O programa que monta `dados/entregas.csv` a partir das entregas do curso |
| `dados/LEIA-ME.md` | O que mudou na planilha corrigida |
| `runs/caio/` | A run do Caio, usada no exercício |
| `contrato.md` | O modelo do contrato do experimento, para preencher |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

A planilha de antes da correção é a do curso, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` daquela pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. De dentro dessa pasta:

```bash
python3 experimento.py rodar --nome paula --semente 2
```

Leva menos de dois segundos. A primeira linha da saída é `run 'paula' gravada em runs/paula/`, e a linha do acerto do par termina em `validação 63,51%; Brier na validação 0,2077`. Se o seu NumPy for outro, os hashes dos artefatos e as últimas casas podem mudar: a aula explica por quê.

Os outros comandos da aula:

```bash
python3 experimento.py rodar --nome planilha-antiga --semente 2 --dados ../dados/entregas/entregas.csv
python3 experimento.py comparar runs/paula runs/planilha-antiga
python3 experimento.py sementes
python3 experimento.py sementes --dados ../dados/entregas/entregas.csv
```

O `rodar` pede sempre `--nome` e `--semente`. Se a pasta `runs/<nome>/` já existe, ele para sem gravar nada: escolha outro nome. Para começar de novo, apague a pasta da run que você gravou, nunca a `runs/caio/`.

O `sementes` roda as sementes de 0 a 9 sem gravar runs; `--de` e `--ate` mudam a faixa. Ele leva de cinco a dez segundos.

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

Se você apagar `dados/entregas.csv` sem querer, rode `python3 dados/atualizar.py` nessa pasta. O arquivo sai igual, byte a byte.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
