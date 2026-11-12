# Árvore, Gini, entropia e poda

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `divisao.py` | Todas as contas da aula: as dez entregas feitas à mão, a primeira e a segunda pergunta da árvore pelo Gini e pela entropia, a árvore sem limite, a poda e a tabela lado a lado |
| `arvore.py` | O Gini, a entropia, a árvore, o caminho de poda, a escolha do alfa pelas dobras e as medidas, no pandas e no NumPy |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são as entregas da Loja Exemplo, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. De dentro dessa pasta:

```bash
python3 divisao.py
```

Leva cerca de um minuto. A maior parte do tempo é a escolha do alfa: a árvore cresce de novo cinco vezes. A saída tem sete partes, as mesmas da aula, e sai igual toda vez. A primeira linha é `1. As entregas` e a segunda começa com `treino: 8215 entregas`.

Se o CSV estiver em outro lugar, passe o caminho:

```bash
python3 divisao.py --dados caminho/para/entregas.csv
```

Se o arquivo não estiver no caminho, o programa para com a mensagem `não achei o arquivo de entregas` e sai com código 2. Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

O exercício não pede código. Os números dele estão no enunciado. Você pode fazer as contas numa planilha, na calculadora ou no Python, e usar as funções `gini` e `entropia` do `arvore.py` para ver se a sua conta bateu:

```python
import arvore
print(arvore.gini(4, 10), arvore.entropia(4, 10))
```

Essa linha imprime o Gini e a entropia das dez entregas da aula: `0.48 0.9709505944546686`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
