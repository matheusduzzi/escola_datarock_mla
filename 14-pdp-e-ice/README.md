# PDP e ICE

A pasta dessa aula no kit do curso ML avançado.

| Arquivo | O que é |
|---|---|
| `pdp_ice.py` | Todas as contas da aula: a floresta, a PDP e o ICE da distância e do peso, onde as entregas existem, o ALE e os 5 sorteios |
| `dependencia.py` | O ICE, a PDP, a contagem de entregas parecidas no treino e o ALE, no NumPy |
| `floresta.py` | A floresta das aulas 04 e 11, com as 8 colunas dessa aula |
| `minhas-respostas.md` | Os espaços do exercício para você preencher |

Os dados são as entregas da Loja Exemplo, em `../dados/entregas/entregas.csv`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente do diagnóstico do curso. Essa aula não usa o scikit-learn: o programa faz as contas, para você ver cada passo. A seção 9.6 da aula mostra o mesmo gráfico no scikit-learn. De dentro dessa pasta:

```bash
python3 pdp_ice.py
```

Leva uns 20 segundos. A saída tem sete partes, as mesmas da aula, e sai igual toda vez. A primeira linha é `1. A floresta e as entregas` e a segunda começa com `treino: 8215 entregas`. A parte 7 treina mais quatro florestas. Para pular essa parte, rode `python3 pdp_ice.py --sem-sorteios`.

Para fazer as contas com outra coluna, passe o nome dela. As colunas aceitas são `distancia_km`, `peso_kg`, `prazo_prometido_dias` e `transportadora`:

```bash
python3 pdp_ice.py --coluna transportadora
```

Se o CSV estiver em outro lugar, passe o caminho:

```bash
python3 pdp_ice.py --dados caminho/para/entregas.csv
```

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

## Os gráficos

Para desenhar os gráficos, o programa usa o matplotlib. Foi testado com o matplotlib 3.9, no Python 3.9. Instale com:

```bash
python3 -m pip install matplotlib
```

E rode com `--graficos`:

```bash
python3 pdp_ice.py --graficos
```

Os gráficos ficam na pasta `graficos/`, um arquivo por coluna: `pdp_ice_distancia_km.png`, `pdp_ice_peso_kg.png` e, com `--coluna transportadora`, `pdp_ice_transportadora.png`. Sem o matplotlib, o programa mostra as tabelas, avisa que falta o matplotlib e termina com erro, sem desenhar nada. As tabelas não dependem dele.

O exercício pede o gráfico da transportadora e as ressalvas escritas no `minhas-respostas.md`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
