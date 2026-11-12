# Multiclasse

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `multiclasse.py` | Todas as contas da aula: as quatro situações, as três estratégias (multinomial, um contra o resto e um contra um), a matriz de confusão 4 × 4 com as médias micro, macro e ponderada, a regra da nota acima do normal e as médias só de atraso longo e extravio |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são as entregas da Loja Exemplo, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna. A coluna `situacao` tem as quatro situações da aula.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. Essa aula não usa o scikit-learn: o programa treina a regressão logística no NumPy, pelo método de Newton, e faz as contas das médias na mão. De dentro dessa pasta:

```bash
python3 multiclasse.py
```

Leva menos de um segundo. A saída tem cinco partes, as mesmas da aula, e sai igual toda vez. A primeira linha é `1. As quatro situações` e a segunda é `treino: 8215 entregas de julho de 2025 a fevereiro de 2026`.

O programa usa o treino (julho de 2025 a fevereiro de 2026) e a validação (março e abril de 2026). Maio e junho ficam guardados para o teste do curso e não entram em nenhuma conta. Se o CSV estiver em outro lugar, passe o caminho:

```bash
python3 multiclasse.py --dados caminho/para/entregas.csv
```

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

O exercício não pede código. As contas do aviso saem da matriz das partes 3 e 4, com somas e uma divisão. Se quiser, faça no Python, mas faça primeiro no papel.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
