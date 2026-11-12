# Clientes do Sistema Exemplo

O Sistema Exemplo não existe. Ele vende um programa de gestão para pequenas lojas por assinatura mensal, e a equipe de sucesso do cliente tenta evitar que as lojas cancelem. Os números são sorteados por `gerar.py`, que só usa a biblioteca padrão do Python. Nenhum dado vem de cliente, pessoa ou empresa real.

## Arquivos

A cada trimestre, a equipe tira uma foto de quem está ativo. Uma linha é um cliente ativo numa data de referência: 2025-07-01, 2025-10-01, 2026-01-01 e 2026-04-01. O mesmo cliente aparece em várias datas enquanto não cancela.

| Arquivo | Linhas | Colunas |
|---|---|---|
| `assinantes.csv` | 7499 | as 14 colunas abaixo |
| `assinantes_export_completo.csv` | 7499 | as mesmas, mais `data_cancelamento` e `motivo_cancelamento`, que o financeiro acrescenta ao export |

| Coluna | O que é |
|---|---|
| `id_cliente` | número do cliente; repete entre as datas |
| `data_referencia` | dia da foto |
| `plano` | `basico`, `pro` ou `rede` |
| `segmento` | ramo da loja: `moda`, `mercado`, `farmacia`, `pet` ou `papelaria` |
| `meses_de_casa` | meses desde a contratação |
| `usuarios_contratados` | quantas pessoas da loja usam o sistema |
| `logins_30d` | entradas no sistema nos 30 dias antes da foto |
| `variacao_logins_pct` | variação dos logins em relação à foto anterior, em % |
| `chamados_30d` | chamados ao suporte nos 30 dias antes da foto |
| `nps_ultimo` | nota de 0 a 10 da última pesquisa; vazio quando o cliente não respondeu |
| `atraso_pagamento_dias` | dias de atraso da última mensalidade |
| `desconto_ativo` | 1 quando o cliente tem desconto na mensalidade |
| `recebeu_contato_30d` | 1 quando a equipe de sucesso do cliente ligou para ele nos 30 dias antes da foto |
| `cancelou_90d` | o rótulo: 1 quando o cliente cancelou nos 90 dias depois da foto |

## Como gerar de novo

```bash
python3 gerar.py
```

A saída é `linhas=7499 clientes=2790 cancelaram=1236 contatados=623`. Os arquivos saem iguais toda vez. Nas aulas, você treina e avalia primeiro e só depois abre o gerador para comparar.

Não edite os CSVs à mão. Se uma aula precisar de outro caso, ela traz os arquivos dela na própria pasta.
