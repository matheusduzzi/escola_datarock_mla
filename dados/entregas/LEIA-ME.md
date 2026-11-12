# Entregas da Loja Exemplo

A Loja Exemplo não existe. As entregas são números sorteados por `gerar.py`, que só usa a biblioteca padrão do Python. Nenhum dado vem de cliente, pessoa ou empresa real.

## Arquivo

`entregas.csv` tem 12000 linhas. Uma linha é um pedido despachado entre 2025-07-01 e 2026-06-30, em ordem de data de despacho. As colunas são o que a loja sabe na hora do despacho, mais o rótulo.

| Coluna | O que é |
|---|---|
| `id_entrega` | número da entrega, na ordem do despacho |
| `data_despacho` | dia em que o pedido saiu do estoque |
| `dia_semana_despacho` | 0 é segunda-feira e 6 é domingo |
| `regiao` | região do Brasil do endereço de entrega |
| `distancia_km` | distância do estoque até o endereço, em km |
| `transportadora` | `Rapida`, `Economica` ou `Propria` (a frota da loja) |
| `peso_kg` | peso do pacote, em kg |
| `itens` | quantidade de itens no pacote |
| `valor_frete` | frete cobrado, em reais |
| `prazo_prometido_dias` | prazo de entrega que o cliente viu na compra |
| `vespera_feriado` | 1 quando o despacho foi até dois dias antes de um feriado nacional |
| `cod_rota` | código da rota de coleta, de 1 a 400 |
| `valor_pedido` | valor dos produtos, em reais |
| `situacao` | o rótulo: `no_prazo`, `atraso_curto` (até 3 dias), `atraso_longo` (4 dias ou mais) ou `extravio` |

Quando uma aula pergunta só se a entrega atrasou, ela junta `atraso_curto`, `atraso_longo` e `extravio` numa classe e diz isso no código.

## Como gerar de novo

```bash
python3 gerar.py
```

A saída é `entregas=12000 no_prazo=8273 atraso_curto=2220 atraso_longo=1366 extravio=141`. O arquivo sai igual toda vez. Nas aulas, você treina e avalia primeiro e só depois abre o gerador para comparar.

Não edite o CSV à mão. Se uma aula precisar de outro caso, ela traz os arquivos dela na própria pasta.
