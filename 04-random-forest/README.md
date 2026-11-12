# 04. Random Forest

Os arquivos da aula «Random Forest» do curso ML avançado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `arvores.py` | A leitura das entregas, a árvore por faixas, a floresta com os dois sorteios, a nota fora da amostra, a regressão logística, o acerto do par, o Brier e o rastro do benchmark. Os outros programas usam ele. |
| `floresta.py` | As contas da aula: as entregas, as referências, a floresta por número de árvores, por colunas por nó e por tamanho da amostra, a nota fora da amostra e o custo. Salva `benchmark.csv` e `benchmark-rastro.json`. |
| `meu_benchmark.py` | Onde você escreve as funções `resumir` e `escolher` do exercício. O resto do programa já está pronto. |
| `testar_meu_benchmark.py` | Os testes das suas duas funções, com três configurações inventadas dentro do próprio teste. |
| `minhas-respostas.md` | Onde você escreve as respostas do exercício. |

As entregas vêm de `../dados/entregas/entregas.csv`, a base compartilhada do kit. Os programas tiram maio e junho logo na leitura: esses dois meses são o teste e ficam fechados.

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, como no README do kit do curso. A floresta é feita no NumPy, sem o scikit-learn, para você ver cada sorteio.

```bash
python3 floresta.py
```

Leva alguns minutos: são 35 florestas. A saída tem sete partes, as mesmas das seções 3 a 8 da aula. As partes 1 a 6 saem iguais às da aula, letra por letra, com os mesmos dados e as mesmas versões. A parte 7 é o tempo da sua máquina e sai diferente. No fim, o programa salva `benchmark.csv` e `benchmark-rastro.json` nessa pasta. Para salvar em outro lugar, use `python3 floresta.py --saida outra/pasta`.

```bash
python3 testar_meu_benchmark.py
python3 meu_benchmark.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 20 testes passaram`. Se um teste falhar, a mensagem diz o que saiu e o que era esperado. O `meu_benchmark.py` leva cerca de um minuto e salva `meu_benchmark.csv` e `meu_benchmark-rastro.json`.

Se o tempo sair muito diferente entre duas rodadas, veja se outro programa estava rodando ao mesmo tempo. A linha "carga da máquina" da parte 7 ajuda a ver isso.

Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`. No Windows, a carga da máquina não aparece.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
