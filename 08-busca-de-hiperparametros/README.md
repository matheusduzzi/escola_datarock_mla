# Busca de hiperparâmetros

A pasta dessa aula no kit do curso ML avançado. Ela usa a planilha corrigida da aula 06, que fica em `06-run-parametro-metrica-e-artefato/dados/entregas.csv`, na pasta ao lado desta.

| Arquivo | O que é |
|---|---|
| `busca.py` | Mede a configuração de hoje, o custo de um espaço e roda a grade, a busca aleatória e a otimização bayesiana. Também compara março com abril e mede o ruído da validação |
| `espaco.json` | O espaço de busca da aula, com a faixa e a escala de cada hiperparâmetro |
| `grade_caio.json` | A grade de 432 combinações que o Caio propôs |
| `poucas_rodadas.json` | Uma grade de 3 pontos no canto de poucas rodadas, que a grade do Caio deixou de fora |
| `grade_27.json` | Uma grade de 27 combinações, do mesmo tamanho do orçamento da aula |
| `grade_caio_mensal.json` | A grade que o Caio propôs para a busca de todo mês, usada no exercício |
| `no_sklearn.py` | A mesma busca aleatória com o `RandomizedSearchCV` do scikit-learn |
| `meu-plano.md` | O plano de busca do exercício, para você preencher |
| `minhas-respostas.md` | As respostas do exercício |
| `requirements.txt` | As versões usadas na aula |

## Como rodar

Essa aula precisa do scikit-learn. As versões do `requirements.txt` pedem Python 3.12 ou mais novo. Se o seu `python3` for o 3.9 do diagnóstico, instale um Python mais novo e use ele nessa pasta. Se você já montou o ambiente da aula «MLflow e registry», pode usar o mesmo e só instalar o que falta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

No Windows, troque `python3` por `py` e a segunda linha por `.venv\Scripts\activate`.

Depois, de dentro dessa pasta:

```bash
python3 busca.py atual
```

A segunda linha da saída é `acerto do par em março e abril: 63,61% (março 65,34%, abril 61,87%)`. Os comandos da aula, na ordem:

```bash
python3 busca.py custo --espaco espaco.json --amostras 10 --orcamento 27
python3 busca.py grade --espaco grade_caio.json --nome grade-caio
python3 busca.py efeitos buscas/grade-caio.csv
python3 busca.py grade --espaco poucas_rodadas.json --nome poucas-rodadas
python3 busca.py grade --espaco grade_27.json --nome grade-27
python3 busca.py aleatoria --espaco espaco.json --orcamento 27 --semente 0
python3 busca.py bayesiana --espaco espaco.json --orcamento 27 --semente 0
python3 busca.py repetir --metodo aleatoria --espaco espaco.json --orcamento 60
python3 busca.py repetir --metodo bayesiana --espaco espaco.json --orcamento 60
python3 busca.py meses buscas/grade-caio.csv
python3 busca.py ruido buscas/grade-caio.csv
python3 busca.py bayesiana --espaco espaco.json --orcamento 40 --semente 0 --nome plano
python3 busca.py meses buscas/plano.csv
python3 no_sklearn.py
```

A grade do Caio leva de dois a quatro minutos. Cada `repetir` roda dez buscas de 60 avaliações e leva alguns minutos. Os outros comandos levam segundos.

Cada busca grava uma linha por avaliação em `buscas/<nome>.csv`, com a configuração, o acerto em março e abril juntos, em março, em abril e os segundos. Uma busca gravada não é regravada: para rodar de novo, use outro `--nome` ou apague o arquivo.

O `busca.py` usa uma thread só, para o tempo de cada avaliação ser comparável entre as buscas. Os segundos são de processador e mudam um pouco de uma rodada para outra e de uma máquina para outra. Os acertos não mudam de uma rodada para outra. Eu só rodei numa máquina (macOS, Apple M4). Em outra, com as mesmas versões, eles devem sair iguais ou muito perto; se saírem diferentes, compare as versões, como na aula 06. Se quiser usar mais núcleos, rode com `OMP_NUM_THREADS=4 python3 busca.py ...`.

Se aparecer `No module named 'sklearn'`, o `pip` instalou em outro Python. Rode `python3 -m pip install -r requirements.txt` com o mesmo `python3` que você usa para rodar o programa.

Se aparecer `não achei ...entregas.csv`, a pasta da aula 06 não está ao lado desta. Baixe o kit inteiro.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
