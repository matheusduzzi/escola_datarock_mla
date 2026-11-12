# SHAP e LIME

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `shap_e_lime.py` | Todas as contas da aula: a floresta, o SHAP com seis fundos, o LIME com cinco sorteios e três larguras, as duas explicações lado a lado e a primeira coluna de cada conta |
| `explicar.py` | O SHAP com as 128 combinações das 7 colunas e o LIME com faixas por quartil, vizinhas e reta, no NumPy |
| `floresta.py` | A floresta das aulas 04 e 11, com as 7 colunas dessa aula |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são as entregas da Loja Exemplo, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. Essa aula não usa as bibliotecas `shap` e `lime`: o programa faz as duas contas, para você ver cada passo. A seção 7.6 da aula mostra o mesmo código nas bibliotecas. De dentro dessa pasta:

```bash
python3 shap_e_lime.py
```

Leva uns 10 segundos. A saída tem cinco partes, as mesmas da aula, e sai igual toda vez. A primeira linha é `1. A floresta e as entregas` e a segunda começa com `treino: 8215 entregas`.

Para fazer as mesmas contas com outras entregas, passe o id. Pode repetir:

```bash
python3 shap_e_lime.py --entrega 9751 --entrega 9531
```

A entrega precisa ser de março ou abril de 2026, com id de 8216 a 10103. As entregas de julho a fevereiro são o treino da floresta, e maio e junho são o teste do curso: o programa recusa as duas. Se o CSV estiver em outro lugar, passe o caminho:

```bash
python3 shap_e_lime.py --dados caminho/para/entregas.csv
```

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

O exercício não pede código. Você lê a saída e preenche o `minhas-respostas.md`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
