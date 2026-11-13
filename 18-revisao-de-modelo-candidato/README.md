# Revisão de modelo candidato

A pasta dessa aula no kit do curso ML avançado.

| Arquivo ou pasta | O que é |
|---|---|
| `revisao.py` | O roteiro de revisão: roda o pacote de novo e mostra as seis partes da aula |
| `caso.py` | A floresta, a regressão logística, a guarda e as medidas, os mesmos da aula 17, «Sucesso do cliente» |
| `testar_revisao.py` | Os testes do roteiro |
| `candidato-diego/` | O pacote da versão 1 do Diego, que a aula revisa |
| `modelo-atual/` | O modelo da aula 17, no mesmo formato de pacote |
| `candidato-diego-v2/` | O pacote da versão 2 do Diego, do exercício |
| `minhas-respostas.md` | As tabelas do exercício para você preencher |

Os dados são os clientes do Sistema Exemplo, em `../dados/assinantes/`, na pasta `dados/` do kit. O `LEIA-ME.md` dessa pasta diz o que é cada coluna e o que o export completo tem a mais.

## O que um pacote tem

Cada pasta de candidato tem três arquivos, combinados pelo time:

| Arquivo | O que é |
|---|---|
| `candidato.py` | O código que treina e mede. Ele tem as funções `colunas`, `dividir`, `treinar` e `nota`, a lista `CONFIGS` com as configurações testadas e `DADOS`, o arquivo que ele lê |
| `ficha.json` | O que o autor escreveu sobre o modelo: dados e `sha256`, colunas, divisão, configuração escolhida, resultado e ambiente |
| `registro.csv` | Uma linha por rodada: a configuração, onde ela foi medida, o acerto do par, quantos dos 150 primeiros cancelaram e qual foi escolhida |

## Como rodar

Você precisa do Python 3.9 ou mais novo, com o pandas e o NumPy, o mesmo ambiente das aulas anteriores. De dentro dessa pasta:

```bash
python3 revisao.py candidato-diego
```

Leva uns dez segundos e sai igual toda vez. A primeira linha é `Revisão do pacote candidato-diego, de Diego: ...`. O roteiro roda o `candidato.py` do pacote de novo numa pasta temporária e não mexe nos arquivos entregues. Ele só olha o rótulo de abril quando o próprio pacote já abriu o teste uma vez, depois da escolha.

Para os outros pacotes:

```bash
python3 revisao.py modelo-atual
python3 revisao.py candidato-diego-v2
```

Um pacote sem algum dos três arquivos, com um campo faltando na ficha ou com uma configuração escolhida que o `candidato.py` não tem faz o roteiro parar com código 2 e a mensagem do problema.

## Os testes

```bash
python3 testar_revisao.py
```

Cada teste monta um caso com um defeito conhecido (coluna que conta fotos de depois, coluna que copia o rótulo, categoria que só existe em abril, rodada alterada no registro, teste aberto duas vezes) e espera que o roteiro aponte o defeito. A linha `revisao.py: não achei a pasta do pacote: nao-existe` aparece no meio, de propósito. A última linha é `todos os testes passaram`.

Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas numpy` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
