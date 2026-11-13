# Seleção de atributos

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `selecionar.py` | Todas as contas da aula: as 16 colunas e as 416, o filtro escolhido em dois lugares, o wrapper para a frente, o L1, o pedido da gerente e a tabela lado a lado |
| `selecao.py` | O filtro, o wrapper, a regressão logística com L2 e com L1, as notas fora da dobra e as medidas, no pandas e no NumPy |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são as entregas da Loja Exemplo, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. De dentro dessa pasta:

```bash
python3 selecionar.py
```

Leva menos de um minuto. A saída tem sete partes, as mesmas da aula, e sai igual toda vez. A primeira linha é `1. As entregas` e a segunda começa com `treino: 8215 entregas`.

Se o CSV estiver em outro lugar, passe o caminho:

```bash
python3 selecionar.py --dados caminho/para/entregas.csv
```

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

O exercício não pede código. Você preenche as tabelas do `minhas-respostas.md` com a aula e a saída do `selecionar.py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
