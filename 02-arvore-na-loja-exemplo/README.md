# 02. Árvore na Loja Exemplo

Os arquivos da aula «Árvore na Loja Exemplo» do curso ML avançado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `arvore.py` | A leitura dos pedidos, a árvore pelo Gini, a regressão logística das referências, o acerto do par e o Brier. Os outros programas usam ele. |
| `profundidade.py` | As contas da aula: os pedidos, as referências, a árvore por profundidade, a árvore de profundidade 2, a primeira divisão, a árvore sem limite e a troca dos meses do treino. |
| `meu_diagnostico.py` | Onde você escreve as funções `tabela_generalizacao` e `escolher` do exercício. O resto do programa já está pronto. |
| `testar_meu_diagnostico.py` | Os testes das suas duas funções, com oito pedidos inventados dentro do próprio teste. |
| `minhas-respostas.md` | Onde você escreve as respostas do exercício. |

Os pedidos vêm de `../dados/loja/pedidos.csv`, a base compartilhada do kit. Os programas tiram junho logo na leitura: junho é o teste e fica fechado.

## Como rodar

Você precisa do Python com o pandas e o NumPy, como no README do kit do curso. A árvore é feita no NumPy, sem o scikit-learn, para você ver cada passo.

```bash
python3 profundidade.py
```

A saída tem sete partes numeradas e é a mesma das seções 3 a 7 da aula. Ela sai igual toda vez: não tem sorteio. Para ver a árvore de outra profundidade, rode por exemplo `python3 profundidade.py --desenhar 4`.

```bash
python3 testar_meu_diagnostico.py
python3 meu_diagnostico.py
```

Antes de você escrever as funções, os dois param com `NotImplementedError`. É o esperado. Quando as duas funções estiverem certas, o teste mostra `ok: 13 testes passaram`. Se um teste falhar, a mensagem diz o que saiu e o que era esperado.

Se aparecer `No module named 'pandas'`, o pandas foi instalado em outro Python; o README do kit do curso mostra como resolver. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
