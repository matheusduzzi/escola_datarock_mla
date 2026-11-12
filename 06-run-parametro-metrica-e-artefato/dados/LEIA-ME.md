# Entregas depois da correção do SAC

`entregas.csv` tem as mesmas 12.000 entregas e as mesmas colunas da planilha do curso, `../../dados/entregas/entregas.csv`. O `LEIA-ME.md` daquela pasta diz o que é cada coluna.

Em 3 de julho de 2026, o SAC fechou as reclamações de março e abril. Ele achou 48 entregas que constavam como `no_prazo` e não tinham chegado no prazo. A situação delas mudou:

| Situação nova | Entregas |
|---|---|
| `atraso_longo` | 32 |
| `extravio` | 16 |

Nenhuma outra linha mudou. O treino da aula, de julho de 2025 a fevereiro de 2026, é igual nas duas planilhas, byte a byte.

## Como montar de novo

Da pasta da aula:

```bash
python3 dados/atualizar.py
```

A saída é `entregas=12000 corrigidas_pelo_sac=48`. O programa lê a planilha do curso e escreve `dados/entregas.csv`, igual toda vez. Ele só usa a biblioteca padrão do Python.

Não edite o CSV à mão: o hash muda com um espaço a mais, e as runs da aula deixam de bater.
