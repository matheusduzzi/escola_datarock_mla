# SHAP

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `shap_pequeno.py` | Todas as contas da aula: a tabela de notas, as 8 combinações e as 6 ordens de uma entrega contra uma referência, as três referências e o frete com a distância |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são as entregas da Loja Exemplo, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. Essa aula não usa a biblioteca `shap`: o programa faz a conta inteira, com as 8 combinações das três colunas. De dentro dessa pasta:

```bash
python3 shap_pequeno.py
```

Leva poucos segundos. A saída tem quatro partes, as mesmas da aula, e sai igual toda vez. A primeira linha é `1. A tabela de notas` e a segunda começa com `treino: 8215 entregas`.

Para fazer as mesmas contas com outra entrega ou outra referência, passe o id:

```bash
python3 shap_pequeno.py --entrega 8403
python3 shap_pequeno.py --entrega 8403 --referencia 18
```

A entrega precisa ser de julho de 2025 a abril de 2026, com id até 10103. Maio e junho são o teste do curso, e o programa recusa. Se o CSV estiver em outro lugar, passe o caminho:

```bash
python3 shap_pequeno.py --dados caminho/para/entregas.csv
```

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

O exercício não pede código. Faça a conta da tarefa 1 à mão, com a tabela da parte 1, e só depois rode o programa para ver se bateu.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
