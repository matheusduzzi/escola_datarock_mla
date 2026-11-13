# Sucesso do cliente

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `sucesso.py` | Todas as contas da aula: a divisão por data, a ligação e o NPS vazio, os três candidatos, a nota por plano, a guarda, o teste de abril e a lista |
| `caso.py` | As colunas do modelo, a guarda, a floresta, a regressão logística e as medidas, no pandas e no NumPy |
| `testar_caso.py` | Os testes da guarda, da lista e das travas do `sucesso.py` |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são os clientes do Sistema Exemplo, em `../dados/assinantes/assinantes.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. De dentro dessa pasta:

```bash
python3 sucesso.py
```

Leva uns cinco segundos e sai igual toda vez. A saída tem cinco partes, as mesmas da aula. A primeira linha é `1. Os clientes e a divisão por data`. A foto de abril de 2026 fica fechada: as contas só olham as colunas dela, nunca quem cancelou.

Para abrir o teste, uma vez, depois de escolher o modelo:

```bash
python3 sucesso.py --abrir-teste
```

A saída acrescenta a parte 6. Para escrever a lista de abril, a planilha de revisão à mão e a ficha do modelo na pasta `saida/`:

```bash
python3 sucesso.py --lista
```

Se o CSV estiver em outro lugar, passe o caminho com `--dados caminho/para/assinantes.csv`.

## As opções do exercício

| Opção | O que faz |
|---|---|
| `--capacidade 100` | muda o tamanho da lista (padrão 150) |
| `--com-contato` | põe `recebeu_contato_30d` no modelo |
| `--nps-media` | preenche o NPS vazio com a média do treino e tira a coluna `nps_vazio` |
| `--modelo logistica` | troca a floresta pela regressão logística |
| `--so-comparar` | mostra só a parte 3, a comparação na foto de janeiro |
| `--grupo 20` | muda o tamanho do grupo sorteado sem ligação, na parte 6 (padrão 30) |

As opções podem aparecer juntas. O teste e a lista são só da versão escolhida: com `--abrir-teste` ou `--lista`, o programa recusa `--com-contato`, `--nps-media`, `--modelo` e `--so-comparar`, e sai com código 2. Uma capacidade fora da foto ou um grupo do tamanho da lista também saem com código 2 e a mensagem do problema.

## Os testes

```bash
python3 testar_caso.py
```

Cada teste monta uma foto com um defeito (coluna faltando, duas datas, cliente repetido, plano novo, valor fora da faixa) e espera que a guarda pare ou separe a linha. A última linha é `todos os testes passaram`.

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
