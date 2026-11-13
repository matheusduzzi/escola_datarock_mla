"""As contas da aula «Sucesso do cliente».

    python3 sucesso.py                 # partes 1 a 5: base, ligação e NPS, candidatos, nota por plano, guarda
    python3 sucesso.py --abrir-teste   # acrescenta a parte 6: a foto de abril, aberta uma vez
    python3 sucesso.py --lista         # escreve a lista de abril e a ficha do modelo em saida/

Opções para o exercício (só mudam a parte 3, na foto de janeiro):
    --capacidade 100      tamanho da lista (padrão 150)
    --com-contato         põe recebeu_contato_30d no modelo
    --nps-media           preenche o NPS vazio com a média do treino, sem a coluna nps_vazio
    --modelo logistica    troca a floresta pela regressão logística
    --so-comparar         mostra só a parte 3
    --grupo 20            tamanho do grupo sorteado sem ligação, na parte 6 (padrão 30)
"""
import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import caso as c

AQUI = Path(__file__).resolve().parent
DADOS = AQUI.parent / "dados" / "assinantes" / "assinantes.csv"
SORTEIOS = range(5)
FLORESTA = {"arvores": 100, "colunas": 4, "min_folha": 20}


def floresta(preparo, treino, sorteio):
    x = preparo.matriz(treino)
    cortes = c.faixas(x)
    modelo = c.treinar_floresta(c.em_faixas(x, cortes), treino["cancelou_90d"].to_numpy(), sorteio=sorteio, **FLORESTA)
    return lambda foto: c.nota_floresta(modelo, c.em_faixas(preparo.matriz(foto), cortes))


def logistica(preparo, treino):
    modelo = c.treinar_logistica(preparo.matriz(treino), treino["cancelou_90d"].to_numpy())
    return lambda foto: c.nota_logistica(modelo, preparo.matriz(foto))


def taxa(y):
    return c.pct(float(np.mean(y)))


# ---------------------------------------------------------------- as partes

def parte1(base, treino, validacao):
    print("1. Os clientes e a divisão por data")
    for data, foto in base.groupby("data_referencia"):
        papel = {"2025-07-01": "treino", "2025-10-01": "treino", c.VALIDACAO: "validação", c.TESTE: "teste, fechado"}[data]
        if data == c.TESTE:
            print(f"   {data}: {len(foto)} clientes, {papel}")
        else:
            print(f"   {data}: {len(foto)} clientes, {int(foto['cancelou_90d'].sum())} cancelaram em 90 dias "
                  f"({taxa(foto['cancelou_90d'])}), {papel}")
    repetidos = validacao["id_cliente"].isin(treino["id_cliente"]).sum()
    print(f"   treino: {len(treino)} linhas de {treino['id_cliente'].nunique()} clientes; "
          f"{repetidos} dos {len(validacao)} clientes de janeiro já estavam no treino")
    print()


def parte2(treino):
    print("2. A ligação e o NPS vazio, no treino")
    y = treino["cancelou_90d"].to_numpy()
    ligou = treino["recebeu_contato_30d"].to_numpy() == 1
    print(f"   receberam ligação nos 30 dias antes: {ligou.sum()} de {len(treino)}")
    print(f"   cancelaram: com ligação {taxa(y[ligou])}, sem ligação {taxa(y[~ligou])}")
    sinais = c.sinais_da_equipe(treino)
    print("   sinais da equipe   clientes  com ligação  cancelaram com ligação  sem ligação")
    diferencas, pesos = [], []
    for rotulo, m in (("0", sinais == 0), ("1", sinais == 1), ("2 ou mais", sinais >= 2)):
        com, sem = y[m & ligou], y[m & ~ligou]
        print(f"   {rotulo:<17} {m.sum():>9}  {len(com):>11}  {taxa(com):>23}  {taxa(sem):>11}")
        if rotulo != "0":
            diferencas.append(sem.mean() - com.mean())
            pesos.append(len(com))
    dif = float(np.average(diferencas, weights=pesos))
    print(f"   com 1 sinal ou mais, a diferença média (sem - com ligação) é {c.num(100 * dif)} pontos")
    junto = c.Preparo(treino, com_contato=True)
    modelo = c.treinar_logistica(junto.matriz(treino), y)
    j = junto.nomes.index("recebeu_contato_30d")
    so = c.treinar_logistica(treino[["recebeu_contato_30d"]].to_numpy(dtype=float), y)
    print(f"   regressão logística, odds de cancelar de quem recebeu ligação: sozinha x{c.num(np.exp(so['coef'][1] / so['desvio'][0]), 2)}; "
          f"com as outras colunas x{c.num(np.exp(modelo['coef'][j + 1] / modelo['desvio'][j]), 2)}")
    vazio = treino["nps_ultimo"].isna().to_numpy()
    print(f"   NPS vazio: {vazio.sum()} clientes ({taxa(vazio)}), cancelaram {taxa(y[vazio])}; "
          f"com NPS respondido, {taxa(y[~vazio])}")
    print(f"   mediana do NPS respondido no treino: {c.num(treino['nps_ultimo'].median(), 0)}; "
          f"média: {c.num(treino['nps_ultimo'].mean(), 2)}")
    print()
    return dif


def parte3(treino, validacao, args):
    k = args.capacidade
    y = validacao["cancelou_90d"].to_numpy()
    preparo = c.Preparo(treino, com_contato=args.com_contato, nps_media=args.nps_media)
    mudancas = [n for n, v in (("com recebeu_contato_30d", args.com_contato), ("NPS vazio pela média", args.nps_media),
                               ("regressão logística", args.modelo == "logistica")) if v]
    print(f"3. Os candidatos em janeiro de 2026: quantos dos {k} primeiros cancelaram")
    print(f"   {len(validacao)} clientes, {int(y.sum())} cancelaram ({taxa(y)}); "
          f"{k} sorteados ao acaso acertariam uns {c.num(k * y.mean(), 0)}")
    print(f"   colunas: {len(preparo.nomes)}" + (f"; mudanças: {', '.join(mudancas)}" if mudancas else ""))
    print("   candidato                      acerto do par   dos primeiros   na lista")
    regra = c.sinais_da_equipe(validacao)
    print(f"   {'regra da equipe (sinais)':<30} {c.pct(c.acerto_do_par(regra, y)):>13}   "
          f"{c.num(c.quantos_na_lista(regra, y, k), 1):>13}   {k}")
    if args.modelo == "logistica":
        nota = logistica(preparo, treino)(validacao)
        print(f"   {'regressão logística':<30} {c.pct(c.acerto_do_par(nota, y)):>13}   "
              f"{c.num(c.quantos_na_lista(nota, y, k), 0):>13}   {k}")
        return None
    notas = []
    for s in SORTEIOS:
        nota = floresta(preparo, treino, s)(validacao)
        notas.append(nota)
        print(f"   {'floresta, sorteio ' + str(s):<30} {c.pct(c.acerto_do_par(nota, y)):>13}   "
              f"{c.num(c.quantos_na_lista(nota, y, k), 0):>13}   {k}")
    acertos = [c.quantos_na_lista(n, y, k) for n in notas]
    pares = [c.acerto_do_par(n, y) for n in notas]
    print(f"   floresta, 5 sorteios: acerto do par de {c.pct(min(pares))} a {c.pct(max(pares))}; "
          f"dos {k} primeiros, de {c.num(min(acertos), 0)} a {c.num(max(acertos), 0)} (média {c.num(np.mean(acertos))})")
    if not args.so_comparar:
        nota = logistica(c.Preparo(treino), treino)(validacao)
        print(f"   {'regressão logística':<30} {c.pct(c.acerto_do_par(nota, y)):>13}   "
              f"{c.num(c.quantos_na_lista(nota, y, k), 0):>13}   {k}")
    print()
    return notas[0]


def parte4(validacao, nota, k):
    print("4. A nota e o que aconteceu em janeiro, floresta do sorteio 0")
    y = validacao["cancelou_90d"].to_numpy()
    print("   plano     clientes   nota média   cancelaram")
    for plano in ["basico", "pro"]:
        m = (validacao["plano"] == plano).to_numpy()
        print(f"   {plano:<9} {m.sum():>8}   {c.pct(nota[m].mean()):>10}   {taxa(y[m]):>10}")
    print(f"   {'todos':<9} {len(y):>8}   {c.pct(nota.mean()):>10}   {taxa(y):>10}")
    topo = c.lista(nota, k)
    print(f"   os {k} primeiros: nota média {c.pct(nota[topo].mean())}, "
          f"o que daria {c.num(nota[topo].sum(), 0)} cancelamentos; cancelaram {int(y[topo].sum())}")
    print()


def parte5(treino, validacao, base):
    print("5. A guarda: o que o treino não cobre")
    for rotulo, preparo, foto in (("janeiro, contra o treino", c.Preparo(treino), validacao),
                                  ("abril, contra treino e janeiro",
                                   c.Preparo(pd.concat([treino, validacao], ignore_index=True)), foto_de_abril(base))):
        fora, motivos, avisos = c.guarda(preparo, foto)
        print(f"   {rotulo}: {len(foto)} clientes, {int(fora.sum())} separados para revisão à mão")
        contagem = {}
        for m in motivos:
            for motivo in m:
                chave = motivo.split(" (")[0]
                contagem[chave] = contagem.get(chave, 0) + 1
        for chave, n in sorted(contagem.items(), key=lambda t: -t[1]):
            print(f"      {chave}: {n}")
        for aviso in avisos:
            print(f"      aviso: {aviso}")
    print("   abril entra sem o rótulo: a guarda só olha o que se sabe no dia da foto")
    abril = foto_de_abril(base)
    rede = (abril["plano"] == "rede").to_numpy()
    preparo = c.Preparo(treino)
    print(f"   como os candidatos treinados em 2025 dão nota aos {rede.sum()} clientes do plano rede de abril, sem rótulo:")
    for nome, dar_nota in (("regressão logística", logistica(preparo, treino)), ("floresta, sorteio 0", floresta(preparo, treino, 0))):
        nota = dar_nota(abril)
        topo = c.lista(nota, 150)
        print(f"      {nome}: nota média {c.pct(nota[rede].mean())} (o resto da foto, {c.pct(nota[~rede].mean())}); "
              f"{int(rede[topo].sum())} deles entre os 150 primeiros")
    print()


def foto_de_abril(base):
    """A foto de abril sem as respostas: é o que existe no dia de dar a nota."""
    return base[base["data_referencia"] == c.TESTE].drop(columns=["cancelou_90d"]).reset_index(drop=True)


def versao_entregue(base):
    treino, validacao, _ = c.dividir(base)
    passado = pd.concat([treino, validacao], ignore_index=True)
    preparo = c.Preparo(passado)
    return passado, preparo, floresta(preparo, passado, 0)


def parte6(base, k, dif, controle=30):
    print("6. O teste: a foto de abril de 2026, aberta uma vez")
    passado, preparo, dar_nota = versao_entregue(base)
    _, _, teste = c.dividir(base, abrir_teste=True)
    y = teste["cancelou_90d"].to_numpy()
    print(f"   versão entregue: a floresta do sorteio 0, treinada de novo nas três fotos ({len(passado)} linhas)")
    print(f"   abril: {len(teste)} clientes, {int(y.sum())} cancelaram ({taxa(y)}); "
          f"{k} ao acaso acertariam uns {c.num(k * y.mean(), 0)}")
    nota = dar_nota(teste)
    fora, _, _ = c.guarda(preparo, teste.drop(columns=["cancelou_90d"]))
    topo = c.lista(nota, k, fora)
    regra = c.sinais_da_equipe(teste)
    dentro = ~fora
    print(f"   acerto do par: floresta {c.pct(c.acerto_do_par(nota, y))}; regra da equipe {c.pct(c.acerto_do_par(regra, y))}")
    print(f"   lista de {k} sem os separados pela guarda: {int(y[topo].sum())} cancelaram; "
          f"regra da equipe na mesma foto, sem os separados: {c.num(c.quantos_na_lista(regra[dentro], y[dentro], k), 1)}")
    print(f"   separados pela guarda: {int(fora.sum())}, cancelaram {int(y[fora].sum())} ({taxa(y[fora])}); "
          f"a floresta dava a eles nota média {c.pct(nota[fora].mean())}")
    print("   plano     clientes   nota média   cancelaram")
    for plano in ["basico", "pro", "rede"]:
        m = (teste["plano"] == plano).to_numpy()
        print(f"   {plano:<9} {m.sum():>8}   {c.pct(nota[m].mean()):>10}   {taxa(y[m]):>10}")
    print(f"   os {k} da lista: nota média {c.pct(nota[topo].mean())}; "
          f"{int(teste['recebeu_contato_30d'].to_numpy()[topo].sum())} já tinham recebido ligação nos 30 dias antes")
    rng = np.random.default_rng(17)
    pares = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y))
        pares.append(c.acerto_do_par(nota[i], y[i]))
    baixo, alto = np.quantile(pares, [0.05, 0.95])
    print(f"   sorteando os clientes de abril de novo, 500 vezes: acerto do par de {c.pct(baixo)} a {c.pct(alto)} em 90% das vezes")
    p = float(y[topo].mean())
    print(f"   a lista acertou {c.pct(p)}; com esse acerto, a quantidade que cancela muda de um trimestre para outro só por sorte:")
    for n in (k, controle):
        print(f"      em {n} clientes: espera {c.num(n * p, 0)}; em 95% dos trimestres, {binomial_baixo(n, p)} ou mais")
    dp1 = (p * (1 - p) / controle + p * (1 - p) / (k - controle)) ** 0.5
    dp4 = (p * (1 - p) / (4 * controle) + p * (1 - p) / (4 * (k - controle))) ** 0.5
    print(f"   grupo de {controle} sorteados da lista sem ligação, contra os {k - controle} com ligação: "
          f"o erro típico da diferença entre os dois é de {c.num(100 * dp1)} pontos em um trimestre "
          f"e {c.num(100 * dp4)} somando quatro")
    print(f"   custo do grupo, pela diferença do treino (parte 2): {controle} x {c.num(100 * dif)} pontos = "
          f"uns {c.num(controle * dif)} cancelamentos a mais por trimestre")
    print()


def binomial_baixo(n, p, chance=0.05):
    """O maior número m tal que a chance de ver m ou mais cancelamentos em n é pelo menos 95%."""
    acumulado = 0.0
    for m in range(n + 1):
        acumulado += math.comb(n, m) * p ** m * (1 - p) ** (n - m)
        if acumulado > chance:
            return m
    return n


def escrever_lista(base, k, caminho_dados):
    passado, preparo, dar_nota = versao_entregue(base)
    foto = foto_de_abril(base)
    fora, motivos, avisos = c.guarda(preparo, foto)
    nota = dar_nota(foto)
    topo = c.lista(nota, k, fora)
    saida = AQUI / "saida"
    saida.mkdir(exist_ok=True)
    tabela = foto.iloc[topo][["id_cliente", "plano", "segmento", "recebeu_contato_30d"]].copy()
    tabela.insert(0, "posicao", range(1, k + 1))
    tabela["nota"] = np.round(nota[topo], 3)
    tabela["sinais_da_equipe"] = c.sinais_da_equipe(foto.iloc[topo])
    tabela.to_csv(saida / f"lista-{c.TESTE}.csv", index=False)
    revisar = pd.DataFrame({"id_cliente": foto["id_cliente"][fora].to_numpy(),
                            "plano": foto["plano"][fora].to_numpy(),
                            "motivo": ["; ".join(m) for m, f in zip(motivos, fora) if f]})
    revisar.to_csv(saida / f"revisar-a-mao-{c.TESTE}.csv", index=False)
    dados_sha = hashlib.sha256(Path(caminho_dados).read_bytes()).hexdigest()
    ficha = [
        "Ficha do modelo de cancelamento do Sistema Exemplo",
        f"dados: assinantes.csv, sha256 {dados_sha}",
        f"treino da versão entregue: fotos {', '.join(c.TREINO + [c.VALIDACAO])} ({len(passado)} linhas)",
        "escolha: floresta contra regra da equipe e regressão logística, na foto de 2026-01-01, treino nas de 2025",
        f"modelo: floresta de {FLORESTA['arvores']} árvores, {FLORESTA['colunas']} colunas por nó, "
        f"folha mínima {FLORESTA['min_folha']}, sorteio 0",
        f"colunas ({len(preparo.nomes)}): {', '.join(preparo.nomes)}",
        f"fora do modelo: recebeu_contato_30d (decisão da equipe, não do cliente)",
        f"NPS vazio: coluna nps_vazio, e o NPS preenchido com a mediana do treino ({c.num(preparo.nps_preenche, 0)})",
        f"lista: {k} clientes de maior nota na foto de {c.TESTE}, sem os {int(fora.sum())} separados pela guarda",
        f"a nota ordena; ela não é a chance de cancelar de 2026 nem o efeito de uma ligação",
    ] + [f"aviso da guarda: {a}" for a in avisos]
    (saida / "ficha-do-modelo.txt").write_text("\n".join(ficha) + "\n", encoding="utf-8")
    print(f"Lista de {c.TESTE}")
    print(f"   saida/lista-{c.TESTE}.csv: {k} clientes; {int(tabela['recebeu_contato_30d'].sum())} já receberam "
          f"ligação nos 30 dias antes; {int((tabela['sinais_da_equipe'] == 0).sum())} sem nenhum sinal da equipe")
    sem = foto.iloc[topo][tabela["sinais_da_equipe"].to_numpy() == 0]
    print(f"   dos {len(sem)} sem nenhum sinal: {int(sem['nps_ultimo'].isna().sum())} com NPS vazio, "
          f"{int((sem['plano'] == 'basico').sum())} no plano básico, {int((sem['meses_de_casa'] <= 12).sum())} com até 12 meses de casa; "
          f"logins nos 30 dias, mediana {c.num(sem['logins_30d'].median(), 0)} (a foto inteira: {c.num(foto['logins_30d'].median(), 0)})")
    print(f"   saida/revisar-a-mao-{c.TESTE}.csv: {len(revisar)} clientes; planos: "
          + ", ".join(f"{p} {n}" for p, n in revisar["plano"].value_counts().items()))
    print("   saida/ficha-do-modelo.txt: os dados, as colunas, os parâmetros e o que a nota não diz")
    print(f"   as 3 primeiras: " + "; ".join(f"cliente {r.id_cliente} ({r.plano}, nota {c.num(r.nota, 3)})"
                                          for r in tabela.head(3).itertuples()))


def main(argv=None):
    p = argparse.ArgumentParser(description="As contas da aula Sucesso do cliente.")
    p.add_argument("--dados", default=str(DADOS), help="caminho do assinantes.csv")
    p.add_argument("--capacidade", type=int, default=150, help="tamanho da lista (padrão 150)")
    p.add_argument("--com-contato", action="store_true", help="põe recebeu_contato_30d no modelo")
    p.add_argument("--nps-media", action="store_true", help="NPS vazio pela média, sem a coluna nps_vazio")
    p.add_argument("--modelo", choices=["floresta", "logistica"], default="floresta")
    p.add_argument("--so-comparar", action="store_true", help="mostra só a parte 3")
    p.add_argument("--grupo", type=int, default=30, help="tamanho do grupo sorteado sem ligação, na parte 6 (padrão 30)")
    p.add_argument("--abrir-teste", action="store_true", help="acrescenta a parte 6, com a foto de abril")
    p.add_argument("--lista", action="store_true", help="escreve a lista de abril e a ficha em saida/")
    args = p.parse_args(argv)
    variante = args.com_contato or args.nps_media or args.modelo != "floresta"
    if (args.abrir_teste or args.lista) and (variante or args.so_comparar):
        p.error("o teste e a lista são só da versão escolhida: tire --com-contato, --nps-media, --modelo e --so-comparar")
    if not Path(args.dados).is_file():
        p.error(f"não achei o arquivo {args.dados}; passe o caminho com --dados")
    base = c.ler(args.dados)
    treino, validacao, _ = c.dividir(base)
    if not 1 <= args.capacidade <= len(validacao):
        p.error(f"--capacidade precisa ficar entre 1 e {len(validacao)}: {args.capacidade}")
    if not 1 <= args.grupo < args.capacidade:
        p.error(f"--grupo precisa ficar entre 1 e {args.capacidade - 1}: {args.grupo}")
    if args.lista:
        escrever_lista(base, args.capacidade, args.dados)
        return 0
    if args.so_comparar:
        parte3(treino, validacao, args)
        return 0
    parte1(base, treino, validacao)
    dif = parte2(treino)
    nota = parte3(treino, validacao, args)
    if nota is not None:
        parte4(validacao, nota, args.capacidade)
    parte5(treino, validacao, base)
    if args.abrir_teste:
        parte6(base, args.capacidade, dif, args.grupo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
