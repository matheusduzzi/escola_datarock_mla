# Multiclasse na Loja Exemplo

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `custos.py` | Todas as contas da aula: treina o mesmo modelo de quatro situações da aula 15 e mede quanto custa cada regra de decisão com uma matriz de custo |
| `custos_logistica.csv` | A matriz de custo da gerente de logística, em reais por entrega |
| `custos_logistica_extravio_500.csv` | A mesma matriz com o extravio sem tratamento a R$ 500 |
| `custos_aviso.csv` | A matriz do aviso do atendimento, em branco, para você preencher no exercício |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são as entregas da Loja Exemplo, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. Essa aula não usa o scikit-learn. De dentro dessa pasta:

```bash
python3 custos.py
```

Leva menos de um segundo e sai igual toda vez. A saída tem quatro partes, as mesmas da aula. A primeira linha é `1. A matriz de custo: custos_logistica.csv`.

Para usar outra matriz:

```bash
python3 custos.py --custos custos_logistica_extravio_500.csv
```

Para escolher a situação com uma matriz e medir o custo com outra, como na seção 6 da aula:

```bash
python3 custos.py --custos custos_logistica_extravio_500.csv --decidir-com custos_logistica.csv
```

O programa usa o treino (julho de 2025 a fevereiro de 2026) e a validação (março e abril de 2026). Maio e junho ficam guardados para o teste do curso e não entram em nenhuma conta. Se o CSV das entregas estiver em outro lugar, passe `--dados caminho/para/entregas.csv`.

## A matriz de custo

Uma linha por situação real e uma coluna por situação prevista, em reais por entrega. A primeira coluna se chama `real`. As 16 células precisam de número; vírgula ou ponto como separador decimal, tanto faz. Se faltar uma célula, uma linha ou uma coluna, o programa para e diz qual.

```
real,no_prazo,atraso_curto,atraso_longo,extravio
no_prazo,0,0,4,12
...
```

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
