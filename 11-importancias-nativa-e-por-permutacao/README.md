# Importâncias nativa e por permutação

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `importancias.py` | Todas as contas da aula: a floresta, a importância nativa, a permutação, as colunas que andam juntas, os 5 sorteios lado a lado e os dois meses separados |
| `medidas.py` | A árvore, a floresta, o acerto do par e as duas importâncias, no pandas e no NumPy |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são as entregas da Loja Exemplo, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. De dentro dessa pasta:

```bash
python3 importancias.py
```

Leva cerca de dois minutos. A saída tem seis partes, as mesmas da aula, e sai igual toda vez. A primeira linha é `1. As entregas e a floresta` e a segunda começa com `treino: 8215 entregas`.

Se o CSV estiver em outro lugar, passe o caminho:

```bash
python3 importancias.py --dados caminho/para/entregas.csv
```

## Os pedidos extras

Duas opções acrescentam contas no fim da saída. Elas podem aparecer mais de uma vez no mesmo comando.

| Opção | O que faz |
|---|---|
| `--grupo distancia_km,valor_frete` | embaralha as colunas juntas, com a mesma ordem, nos 5 sorteios; com uma coluna só, mostra a queda dela e a faixa |
| `--tirar valor_frete` | treina a floresta de novo sem a coluna, nos 5 sorteios, e mostra o acerto e as 5 maiores quedas |
| `--so-extras` | pula as seis partes da aula e mostra só a floresta de referência e os pedidos extras |

Os nomes das colunas são os da parte 2 da saída, como `peso_kg` ou `transportadora=Economica`. Com um nome que não existe, o programa para e mostra a lista certa.

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
