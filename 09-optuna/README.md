# Optuna

A pasta dessa aula no kit do curso ML avançado. Ela usa duas pastas ao lado desta: a da aula 08, `08-busca-de-hiperparametros/`, de onde vem a avaliação do boosting (`busca.py`), e a da aula 06, `06-run-parametro-metrica-e-artefato/`, com a planilha corrigida em `dados/entregas.csv`.

| Arquivo | O que é |
|---|---|
| `estudo.py` | Roda o plano da aula 08 no Optuna: cria, continua, repete e compara estudos, roda várias sementes e exporta o estudo com a decisão |
| `teste_final.py` | Confere o estudo exportado e abre maio e junho uma vez, com recibo |
| `espaco.json` | O espaço de busca, o mesmo arquivo da aula 08 |
| `minhas-respostas.md` | As respostas do exercício |
| `minha-decisao.md` | A decisão do exercício, para você preencher |
| `requirements.txt` | As versões usadas na aula |

## Como rodar

Essa aula precisa do Optuna e do scikit-learn. As versões do `requirements.txt` pedem Python 3.12 ou mais novo. Se o seu `python3` for o 3.9 do diagnóstico, instale um Python mais novo e use ele nessa pasta. Se você já montou o ambiente da aula «Busca de hiperparâmetros», pode usar o mesmo e só instalar o que falta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

No Windows, troque `python3` por `py` e a segunda linha por `.venv\Scripts\activate`.

Depois, de dentro dessa pasta:

```bash
python3 estudo.py rodar --nome plano --semente 0
```

A última linha da saída é `melhor: 64,09%, 0,48 ponto acima da configuração de hoje`. Os comandos da aula, na ordem:

```bash
python3 estudo.py repetir --semente 0 --vezes 2
python3 estudo.py repetir --semente 0 --vezes 2 --jobs 4
python3 estudo.py rodar --nome interrompido --semente 0 --parar-em 20
python3 estudo.py rodar --nome interrompido --semente 0
python3 estudo.py comparar plano interrompido
python3 estudo.py rodar --nome esquecido --semente 0 --parar-em 20
python3 estudo.py rodar --nome esquecido --semente 0 --esquecer-sampler
python3 estudo.py comparar plano esquecido
python3 estudo.py sementes --de 0 --ate 4
python3 estudo.py exportar --nome plano
python3 estudo.py sementes --de 0 --ate 0 --embaralhar-teste
python3 teste_final.py estudos/plano-estudo.json
```

Cada estudo de 40 tentativas leva uns 15 segundos de processador. O `sementes --de 0 --ate 4` roda cinco estudos e leva perto de um minuto e meio.

Cada estudo fica em `estudos/<nome>.db`, um arquivo SQLite, com o sampler guardado em `estudos/<nome>-sampler.pkl`. Rodar o mesmo `--nome` de novo continua o estudo, com o mesmo plano. Para outro plano, use outro nome. O `exportar` grava `estudos/<nome>-tentativas.csv` e `estudos/<nome>-estudo.json`. O `teste_final.py` grava `estudos/teste-final.json` e não abre o teste de novo enquanto ele existir.

Abra só os arquivos `.pkl` que você mesmo gravou: abrir um `.pkl` roda código.

O boosting usa uma thread só, como na aula 08, e os segundos de processador mudam um pouco de uma rodada para outra. Os acertos e as tentativas não mudam, com uma tentativa por vez. Com `--jobs` acima de 1, as tentativas mudam a cada rodada: a aula mostra por quê. Eu só rodei numa máquina (macOS, Apple M4). Em outra, com as mesmas versões, os números devem sair iguais ou muito perto; se saírem diferentes, compare as versões, como na aula 06. Com outra versão do Optuna, o mesmo plano pode sugerir outras tentativas.

Se aparecer `No module named 'optuna'`, o `pip` instalou em outro Python. Rode `python3 -m pip install -r requirements.txt` com o mesmo `python3` que você usa para rodar o programa.

Se aparecer `não achei 08-busca-de-hiperparametros/busca.py` ou `não achei ...entregas.csv`, falta uma das pastas ao lado desta. Baixe o kit inteiro.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
