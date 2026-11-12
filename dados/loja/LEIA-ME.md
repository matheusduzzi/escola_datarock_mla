# Pedidos da Loja Exemplo

A Loja Exemplo não existe. Os pedidos são números sorteados por `gerar.py`, que só usa a biblioteca padrão do Python. Nenhum dado vem de cliente, pessoa ou empresa real.

## Arquivos

| Arquivo | Linhas | Colunas | Uma linha é |
|---|---|---|---|
| `pedidos.csv` | 2400 | `id_pedido`, `data_pedido`, `categoria`, `forma_pagamento`, `valor`, `prazo_prometido_dias`, `cliente_novo`, `pedidos_anteriores`, `cancelado` | um pedido de 2026-01-01 a 2026-06-30 |
| `pedidos_export_completo.csv` | 2400 | as mesmas, mais `canal`, `dias_ate_entrega` e `contatos_sac` | o mesmo pedido, com o que outros sistemas acrescentam ao export |

`cancelado` é o rótulo: 1 quando o pedido foi cancelado, 0 quando foi entregue.

## Como gerar de novo

```bash
python3 gerar.py
```

A saída é `pedidos=2400 cancelados=467 cancelados_com_entrega=22 entregues_com_dias_zero=0`. Os arquivos saem iguais toda vez. Nas aulas, você prepara e avalia primeiro e só depois abre o gerador para comparar.

Não edite os CSVs à mão. Se uma aula precisar de outro caso, ela traz os arquivos dela na própria pasta.
