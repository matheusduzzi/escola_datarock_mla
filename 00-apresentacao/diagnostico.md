# Diagnóstico: por onde começar em ML avançado

A gerente de logística da Loja Exemplo quer saber, na hora do despacho, quais entregas vão atrasar. O pessoal do Sistema Exemplo, que vende o programa de gestão da loja, quer outra coisa: saber quais clientes vão cancelar a assinatura. Antes de treinar a primeira árvore para qualquer um dos dois, eu refiz cinco contas que o curso usa desde a primeira aula. As cinco tarefas abaixo são essas contas e o código que eu usei.

Elas pedem o que vem dos cursos anteriores: escolher a complexidade de um modelo pela validação, montar a matriz de confusão com custo, calcular um encoding só no treino, escolher entre configurações sem gastar o teste e separar associação de efeito. O diagnóstico não é prova e não tem nota. Ele mostra por onde você começa. Anote na [folha de respostas](minhas-respostas.md) a conta que fez, o comando que rodou e a saída que apareceu. Só depois compare com as respostas.

As tarefas 1, 4 e 5 são de lápis e papel ou calculadora. As tarefas 2 e 3 rodam de dentro da pasta `00-apresentacao/`:

```bash
cd 00-apresentacao
python3 tarefa3.py
```

No Windows, troque `python3` por `py`.

## Tarefa 1. Quanto regularizar

Uma colega ajustou uma regressão Ridge para o valor do frete com dez colunas tiradas da distância (a distância, o quadrado dela, o cubo e assim por diante). Ela treinou com 20 entregas e mediu o erro em outras 400, que não entraram no ajuste. O erro é o RMSE, em reais.

| alpha | RMSE no treino (R$) | RMSE na validação (R$) |
|---|---|---|
| 0,0000001 | 27,60 | 55,00 |
| 0,00001 | 29,40 | 35,90 |
| 0,001 | 30,70 | 31,90 |
| 0,01 | 31,10 | 32,10 |
| 0,1 | 34,40 | 35,50 |
| 1 | 37,90 | 39,90 |

1. Que alpha você escolheria? Por quê?
2. Por que o erro no treino só sobe quando o alpha cresce?
3. Com alpha 0,0000001, o erro na validação é o dobro do erro no treino. Como se chama isso? O que o modelo fez com as 20 entregas?
4. Os alphas 0,001 e 0,01 ficam a R$ 0,20 um do outro na validação. Isso muda a sua escolha?
5. Você escolheu o alpha olhando a validação. O RMSE dela é uma boa estimativa do erro nas entregas do mês que vem? O que falta?

## Tarefa 2. Matriz de confusão e custo dos erros

Em `tarefa2.py`, escreva duas funções, sem loop em Python:

| Função | O que retorna |
|---|---|
| `matriz_confusao(y, previsto)` | um array 2 x 2 do NumPy, `[[vn, fp], [fn, vp]]`; `y` e `previsto` podem chegar como listas de 0 e 1; para com `ValueError` se os tamanhos forem diferentes ou se aparecer outro valor |
| `custo(matriz, custo_fp, custo_fn)` | o custo total dos erros: falsos positivos vezes `custo_fp` mais falsos negativos vezes `custo_fn` |

Depois rode:

```bash
python3 tarefa2.py
python3 testar_tarefa2.py
```

Enquanto as funções não existem, os dois comandos param com `NotImplementedError`. Anote na folha:

1. As duas linhas que `tarefa2.py` mostrou.
2. No exemplo, ligar para um cliente custa R$ 15 e um pedido cancelado sem ligação custa R$ 120. Quanto custa prever "entregue" para todos os dez pedidos? E prever "cancelado" para todos?
3. O modelo do exemplo acerta 7 de 10. Prever "cancelado" para todos acerta 4 de 10 e custa menos. O que isso diz sobre escolher o limiar pelo acerto?
4. Por que o teste exige a matriz 2 x 2 mesmo quando não há nenhum cancelado?

## Tarefa 3. Taxa por vendedor sem olhar o teste

Os vendedores do marketplace da Loja Exemplo cancelam pedidos em taxas diferentes. A ideia da equipe é usar a taxa de cancelamento de cada vendedor como nota do pedido. Abra `tarefa3.py` e leia tudo antes de rodar. Os 900 primeiros pedidos, pela data, são o treino e os 300 últimos, o teste. A AUC mede se pedido cancelado recebe nota maior que pedido entregue; 0,5 é o mesmo que sortear. Rode e anote as três linhas.

1. Que pedidos entraram no cálculo da taxa de cada vendedor?
2. A AUC no teste saiu maior que no treino. Por que isso deveria te deixar desconfiado?
3. Reescreva o programa para calcular a taxa de cada vendedor e a taxa geral só com o treino. Rode e anote as três linhas novas.
4. Na versão nova, a AUC no treino continua bem acima da do teste. Por quê? Pense num vendedor que só tem um pedido no treino.

## Tarefa 4. Quarenta configurações e um teste só

Outra colega testou 40 configurações de um modelo de cancelamento. Ela avaliou todas no mesmo teste, com 400 pedidos de março, dos quais 98 foram cancelados. As AUCs ficaram entre 0,63 e 0,74. Ela escolheu a melhor e escreveu no relatório: "AUC de 0,74". Em abril, a configuração escolhida deu 0,69. As 40, em média, deram 0,68 em abril.

1. Por que a melhor AUC no teste tende a ficar acima do que o modelo faz no mês seguinte?
2. O que ela deveria ter usado para escolher entre as 40? Para que serve o teste, então?
3. Ela pode escrever "AUC de 0,74" no relatório? O que você escreveria no lugar?
4. As 40 AUCs vão de 0,63 a 0,74 no mesmo teste. O que essa faixa diz sobre a diferença entre as configurações?

## Tarefa 5. Quem recebe ligação cancela mais

No Sistema Exemplo, a equipe de sucesso do cliente liga para os clientes que ela acha que vão cancelar. Num trimestre, com 2.000 clientes ativos, ela separou os números por grupo de risco, pela régua que ela mesma usa para decidir para quem ligar:

| Grupo de risco | Recebeu ligação | Clientes | Cancelaram em 90 dias |
|---|---|---|---|
| alto | sim | 360 | 108 |
| alto | não | 140 | 56 |
| baixo | sim | 40 | 4 |
| baixo | não | 1.460 | 146 |

1. Qual a taxa de cancelamento de quem recebeu ligação? E de quem não recebeu?
2. Dentro do grupo de risco alto, qual a diferença entre quem recebeu e quem não recebeu ligação? E dentro do baixo?
3. Por que a comparação geral e a comparação dentro dos grupos dão sinais opostos?
4. Um modelo de cancelamento treinado com a coluna "recebeu ligação" dá peso positivo a ela. A equipe pode concluir que ligar faz o cliente cancelar? Pode usar esse modelo para escolher para quem ligar no próximo trimestre?

## Depois das tarefas

Compare a folha com as respostas na aula e veja a tabela da apresentação do curso. Ela diz que aula estudar para cada tarefa em que você travou.

Se precisar de ajuda, deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
