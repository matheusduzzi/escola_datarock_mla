"""O roteiro de revisão de um modelo candidato, da aula «Revisão de modelo candidato».

    python3 revisao.py candidato-diego        # o pacote da versão 1 do Diego
    python3 revisao.py modelo-atual           # o modelo da aula 17, no mesmo formato
    python3 revisao.py candidato-diego-v2     # o exercício

Um pacote é uma pasta com candidato.py, ficha.json e registro.csv (ver o README.md). O roteiro
tem seis partes: reprodução, colunas no dia da foto, divisão e teste, o candidato medido como vai
ser usado em janeiro, o teste e os riscos que ficam. Ele roda o candidato.py de novo numa pasta
temporária e nunca lê o rótulo de abril de um pacote que não abriu o teste.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import caso as c

AQUI = Path(__file__).resolve().parent
PASTA_DADOS = AQUI.parent / "dados" / "assinantes"
ALVO = "cancelou_90d"
DESFECHO = [ALVO, "data_cancelamento", "motivo_cancelamento"]
LISTA = 150
CAMPOS = {"autor", "dados", "modelo", "config_escolhida", "colunas", "divisao", "resultado", "ambiente"}
Z90 = 1.645


class ErroDoPacote(ValueError):
    """O pacote não tem o que a revisão precisa: arquivo, campo da ficha ou função do candidato."""


# ---------------------------------------------------------------- o pacote

def abrir_pacote(pasta):
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise ErroDoPacote(f"não achei a pasta do pacote: {pasta.name}")
    faltam = [a for a in ("candidato.py", "ficha.json", "registro.csv") if not (pasta / a).is_file()]
    if faltam:
        raise ErroDoPacote(f"faltam arquivos no pacote {pasta.name}: {', '.join(faltam)}")
    ficha = json.loads((pasta / "ficha.json").read_text(encoding="utf-8"))
    faltam = sorted(CAMPOS - set(ficha))
    if faltam:
        raise ErroDoPacote(f"faltam campos na ficha.json: {', '.join(faltam)}")
    for campo, dentro in (("dados", ("arquivo", "sha256")), ("resultado", ("acerto_do_par", "dos_150_primeiros"))):
        falta = [d for d in dentro if d not in ficha[campo]]
        if falta:
            raise ErroDoPacote(f"faltam campos em ficha.json/{campo}: {', '.join(falta)}")
    registro = pd.read_csv(pasta / "registro.csv")
    falta = [col for col in ("rodada", "avaliado_em", "acerto_do_par", "dos_150_primeiros", "escolhido")
             if col not in registro.columns]
    if falta:
        raise ErroDoPacote(f"faltam colunas no registro.csv: {', '.join(falta)}")
    spec = importlib.util.spec_from_file_location(f"candidato_{pasta.name.replace('-', '_')}", pasta / "candidato.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    falta = [f for f in ("DADOS", "CONFIGS", "colunas", "dividir", "treinar", "nota") if not hasattr(modulo, f)]
    if falta:
        raise ErroDoPacote(f"o candidato.py não tem: {', '.join(falta)}")
    if ficha["config_escolhida"] not in modulo.CONFIGS:
        raise ErroDoPacote("a config_escolhida da ficha não está entre as CONFIGS do candidato.py")
    return {"pasta": pasta, "ficha": ficha, "registro": registro, "modulo": modulo}


def sha(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def arquivo_do_sha(valor, pasta=PASTA_DADOS):
    """O nome do arquivo de dados do kit que tem esse SHA-256, ou None."""
    for arq in sorted(Path(pasta).glob("*.csv")):
        if sha(arq) == valor:
            return arq.name
    return None


def registros_iguais(a, b):
    """Quantas rodadas do registro b são iguais às do registro a, linha a linha."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return 0
    return int(sum(a.iloc[i].astype(str).tolist() == b.iloc[i].astype(str).tolist() for i in range(len(a))))


# ---------------------------------------------------------------- o dia da foto

def como_no_dia(base, data):
    """A base como ela existe no dia `data`: só as fotos até esse dia, e a foto do dia sem o desfecho."""
    b = base[base["data_referencia"] <= data].copy()
    hoje = (b["data_referencia"] == data).to_numpy()
    for col in DESFECHO:
        if col in b.columns:
            b[col] = b[col].astype(object)
            b.loc[hoje, col] = np.nan
    return b


def colunas_que_mudam(modulo, base, data):
    """Compara as colunas das linhas de `data` calculadas com a base inteira e com a base do dia.

    Retorna (mudam, so_com_depois): quantas linhas mudam em cada coluna e as colunas que só
    existem quando a base tem fotos posteriores a `data`. O treino é o mesmo nas duas contas:
    as fotos de antes de `data`.
    """
    dia = como_no_dia(base, data)
    inteira = modulo.colunas(base, (base["data_referencia"] < data).to_numpy())
    so_dia = modulo.colunas(dia, (dia["data_referencia"] < data).to_numpy())
    linhas = (base["data_referencia"] == data).to_numpy()
    a = inteira[linhas].reset_index(drop=True)
    b = so_dia[(dia["data_referencia"] == data).to_numpy()].reset_index(drop=True)
    mudam = {}
    for col in a.columns:
        if col in b.columns:
            n = int((~np.isclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float))).sum())
            if n:
                mudam[col] = n
    return mudam, sorted(set(a.columns) - set(b.columns))


def coluna_igual(x, serie):
    """As colunas de x que repetem a série, linha a linha."""
    alvo = np.asarray(serie, dtype=float)
    return [col for col in x.columns if np.array_equal(x[col].to_numpy(dtype=float), alvo)]


# ---------------------------------------------------------------- o teste

def uso_do_teste(registro):
    """Como o pacote usou o teste: quantas rodadas foram medidas nele e se a escolha veio antes."""
    no_teste = registro["avaliado_em"].astype(str).str.startswith("teste").to_numpy()
    escolhido = registro["escolhido"].to_numpy() == 1
    n = int(no_teste.sum())
    if n == 0:
        return {"rodadas_no_teste": 0, "aberto_uma_vez": False, "texto": "nenhuma rodada no teste"}
    ultima = len(registro) - 1
    escolha_antes = bool((escolhido & ~no_teste).any())
    uma_vez = n == 1 and bool(no_teste[ultima]) and escolha_antes
    if uma_vez:
        texto = "o teste foi aberto uma vez, na última rodada, depois da escolha"
    elif escolha_antes:
        texto = f"o teste foi aberto {n} vezes"
    else:
        texto = f"a escolha foi feita no próprio teste: {n} rodadas medidas nele"
    return {"rodadas_no_teste": n, "aberto_uma_vez": uma_vez, "texto": texto}


def erro_tipico_da_diferenca(acertos_a, acertos_b, k):
    """O erro típico, em clientes, da diferença entre duas listas de k clientes."""
    pa, pb = acertos_a / k, acertos_b / k
    return k * (pa * (1 - pa) / k + pb * (1 - pb) / k) ** 0.5


# ---------------------------------------------------------------- as partes

def treinar_e_dar_nota(modulo, x, y, treino, avaliar, config):
    modelo = modulo.treinar(x[treino].to_numpy(dtype=float), y[treino], config)
    return modulo.nota(modelo, x[avaliar].to_numpy(dtype=float))


def parte1(pacote):
    ficha, modulo = pacote["ficha"], pacote["modulo"]
    print("1. Reprodução")
    no_kit = arquivo_do_sha(ficha["dados"]["sha256"])
    print(f"   a ficha diz: dados {ficha['dados']['arquivo']}, sha256 {ficha['dados']['sha256'][:12]}…")
    print(f"   o sha256 da ficha é de: {no_kit or 'nenhum arquivo do kit'}"
          + ("" if no_kit == ficha["dados"]["arquivo"] else "   <- não é o arquivo que a ficha diz"))
    print(f"   o candidato.py lê: {Path(modulo.DADOS).name}")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "-B", "candidato.py", "--saida", tmp], cwd=pacote["pasta"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise ErroDoPacote(f"o candidato.py saiu com código {r.returncode}: {r.stderr.strip()[-300:]}")
        novo = pd.read_csv(Path(tmp) / "registro.csv")
    iguais = registros_iguais(pacote["registro"], novo)
    print(f"   rodei o candidato.py de novo: {iguais} de {len(pacote['registro'])} rodadas iguais ao registro.csv entregue")
    reg = pacote["registro"]
    final = reg[reg["escolhido"] == 1].iloc[-1]
    res = ficha["resultado"]
    bate = (abs(float(final["acerto_do_par"]) - res["acerto_do_par"]) < 0.0005
            and abs(float(final["dos_150_primeiros"]) - res["dos_150_primeiros"]) < 0.05)
    print(f"   a ficha promete: acerto do par {c.pct(res['acerto_do_par'])}, dos 150 primeiros {c.num(res['dos_150_primeiros'], 0)}; "
          f"no registro, rodada {int(final['rodada'])}: {c.pct(final['acerto_do_par'])} e {c.num(final['dos_150_primeiros'])}"
          + ("" if bate else "   <- não bate"))
    amb = ficha["ambiente"]
    aqui = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    diferentes = [k for k in aqui if str(amb.get(k)) != aqui[k]]
    print(f"   ambiente da ficha: Python {amb.get('python')}, pandas {amb.get('pandas')}, NumPy {amb.get('numpy')}; "
          + ("o mesmo daqui" if not diferentes else f"aqui: {', '.join(f'{k} {aqui[k]}' for k in diferentes)}"))
    print()
    return {"iguais": iguais, "bate": bate, "arquivo_certo": no_kit == ficha["dados"]["arquivo"]}


def parte2(pacote, base):
    modulo = pacote["modulo"]
    print("2. As colunas no dia da foto")
    treino_jan = base["data_referencia"].isin(c.TREINO).to_numpy()
    x = modulo.colunas(como_no_dia(base, c.VALIDACAO), treino_jan[base["data_referencia"].to_numpy() <= c.VALIDACAO])
    print(f"   colunas do modelo ({len(pacote['ficha']['colunas'])}): {', '.join(pacote['ficha']['colunas'])}")
    mudam, so_depois = colunas_que_mudam(modulo, base, c.VALIDACAO)
    n = int((base["data_referencia"] == c.VALIDACAO).sum())
    print(f"   na foto de janeiro, refazendo as colunas só com o que existia em {c.VALIDACAO}:")
    if not mudam and not so_depois:
        print("      nenhuma coluna muda")
    for col, k in sorted(mudam.items(), key=lambda t: -t[1]):
        print(f"      {col} muda em {k} das {n} linhas")
    for col in so_depois:
        print(f"      {col} só existe com fotos de depois de janeiro")
    dia = como_no_dia(base, c.VALIDACAO)
    decisao = coluna_igual(x, dia["recebeu_contato_30d"])
    vazio = coluna_igual(x, dia["nps_ultimo"].isna())
    print(f"   coluna de decisão da equipe no modelo: {', '.join(decisao) if decisao else 'nenhuma'}")
    print(f"   coluna que marca o NPS vazio: {', '.join(vazio) if vazio else 'nenhuma; o vazio vira um número como outro qualquer'}")
    print()
    return {"mudam": mudam, "so_depois": so_depois, "decisao": decisao, "vazio": vazio}


def parte3(pacote, base):
    print("3. A divisão e o teste")
    treino, avaliado = pacote["modulo"].dividir(base)
    def datas(m):
        return ", ".join(f"{d}: {n}" for d, n in base.loc[m, "data_referencia"].value_counts().sort_index().items())
    print(f"   treino: {int(treino.sum())} linhas ({datas(treino)})")
    print(f"   avaliação do pacote: {int(avaliado.sum())} linhas ({datas(avaliado)})")
    ids_av = set(base.loc[avaliado, "id_cliente"])
    nas_duas = len(ids_av & set(base.loc[treino, "id_cliente"]))
    print(f"   clientes nas duas pontas: {nas_duas} dos {len(ids_av)} clientes da avaliação")
    depois = base.loc[avaliado, "data_referencia"].min() > base.loc[treino, "data_referencia"].max()
    print(f"   a avaliação vem toda depois do treino: {'sim' if depois else 'não'}")
    reg = pacote["registro"]
    for lugar, grupo in reg.groupby("avaliado_em", sort=False):
        print(f"   rodadas medidas em {lugar}: {len(grupo)}, acerto do par de {c.pct(grupo['acerto_do_par'].min())} "
              f"a {c.pct(grupo['acerto_do_par'].max())}")
    uso = uso_do_teste(reg)
    print(f"   {uso['texto']}")
    print()
    return {"depois": bool(depois), "uso": uso}


def parte4(pacote, base):
    modulo, config = pacote["modulo"], pacote["ficha"]["config_escolhida"]
    print("4. O candidato medido como vai ser usado: treino em julho e outubro de 2025, nota em janeiro de 2026")
    y = base[ALVO].to_numpy()
    treino = base["data_referencia"].isin(c.TREINO).to_numpy()
    jan = (base["data_referencia"] == c.VALIDACAO).to_numpy()
    yj = y[jan]
    regra = c.sinais_da_equipe(base[jan])
    print(f"   {int(jan.sum())} clientes, {int(yj.sum())} cancelaram; 150 ao acaso acertariam uns {c.num(LISTA * yj.mean(), 0)}")
    print(f"   régua, a regra da equipe: "
          f"acerto do par {c.pct(c.acerto_do_par(regra, yj))}, dos 150 primeiros {c.num(c.quantos_na_lista(regra, yj, LISTA))}")
    x = modulo.colunas(base, treino)
    n1 = treinar_e_dar_nota(modulo, x, y, treino, jan, config)
    print(f"   a versão escolhida, com as colunas feitas na base inteira: acerto do par {c.pct(c.acerto_do_par(n1, yj))}, "
          f"dos 150 primeiros {c.num(c.quantos_na_lista(n1, yj, LISTA))}")
    dia = como_no_dia(base, c.VALIDACAO)
    tr_dia = dia["data_referencia"].isin(c.TREINO).to_numpy()
    jan_dia = (dia["data_referencia"] == c.VALIDACAO).to_numpy()
    xd = modulo.colunas(dia, tr_dia)
    yd = dia[ALVO].to_numpy(dtype=float)
    yd[jan_dia] = yj   # o rótulo de janeiro só entra para medir, depois da nota
    notas = []
    for cfg in modulo.CONFIGS:
        notas.append((cfg, treinar_e_dar_nota(modulo, xd, yd, tr_dia, jan_dia, cfg)))
    n2 = dict((json.dumps(cfg, sort_keys=True), n) for cfg, n in notas)[json.dumps(config, sort_keys=True)]
    print(f"   a versão escolhida, com as colunas feitas só com o que existia em {c.VALIDACAO}: "
          f"acerto do par {c.pct(c.acerto_do_par(n2, yj))}, dos 150 primeiros {c.num(c.quantos_na_lista(n2, yj, LISTA))}")
    pares = [c.acerto_do_par(n, yj) for _, n in notas]
    listas = [c.quantos_na_lista(n, yj, LISTA) for _, n in notas]
    melhor = max(range(len(notas)), key=lambda i: (listas[i], pares[i]))
    print(f"   as {len(notas)} configurações do pacote, do mesmo jeito: acerto do par de {c.pct(min(pares))} a {c.pct(max(pares))}; "
          f"dos 150 primeiros, de {c.num(min(listas))} a {c.num(max(listas))}; a maior lista é a da configuração "
          f"{json.dumps(notas[melhor][0], ensure_ascii=False)}")
    print("   plano     clientes   nota média   cancelaram")
    for plano in ["basico", "pro"]:
        m = (base.loc[jan, "plano"] == plano).to_numpy()
        print(f"   {plano:<9} {m.sum():>8}   {c.pct(n2[m].mean()):>10}   {c.pct(yj[m].mean()):>10}")
    print()
    return {"lista_entregue": c.quantos_na_lista(n1, yj, LISTA), "lista_no_dia": c.quantos_na_lista(n2, yj, LISTA),
            "regua": c.quantos_na_lista(regra, yj, LISTA)}


def parte5(pacote, base, uso):
    print("5. O teste de abril")
    if not uso["aberto_uma_vez"]:
        print("   o pacote não tem um teste aberto uma vez depois da escolha; a revisão não abre abril")
        print()
        return None
    reg = pacote["registro"]
    final = reg.iloc[-1]
    y = base[ALVO].to_numpy()
    abril = (base["data_referencia"] == c.TESTE).to_numpy()
    passado = base["data_referencia"].isin(c.TREINO + [c.VALIDACAO]).to_numpy()
    foto = base[abril].drop(columns=[ALVO]).reset_index(drop=True)
    fora, _, _ = c.guarda(c.Preparo(base[passado]), foto)
    dentro = ~fora if "sem os separados pela guarda" in str(final["avaliado_em"]) else np.ones(len(foto), dtype=bool)
    ya = y[abril][dentro]
    regra = c.sinais_da_equipe(foto[dentro])
    r = c.quantos_na_lista(regra, ya, LISTA)
    acertos = float(final["dos_150_primeiros"])
    et = erro_tipico_da_diferenca(acertos, r, LISTA)
    dif = acertos - r
    baixo, alto = dif - Z90 * et, dif + Z90 * et
    print(f"   o pacote abriu abril na rodada {int(final['rodada'])}: {c.num(acertos, 0)} dos 150 primeiros cancelaram, "
          f"acerto do par {c.pct(final['acerto_do_par'])}")
    print(f"   régua nos mesmos clientes: {c.num(r)} dos 150 primeiros, acerto do par {c.pct(c.acerto_do_par(regra, ya))}")
    print(f"   diferença para a régua: {c.num(dif)} clientes; erro típico {c.num(et)}; "
          f"faixa de 90%: de {c.num(baixo)} a {c.num(alto)}")
    print(f"   a faixa inteira fica acima de zero: {'sim' if baixo > 0 else 'não'}")
    print()
    return {"diferenca": dif, "baixo": baixo, "alto": alto}


def parte6(pacote, base):
    modulo, config = pacote["modulo"], pacote["ficha"]["config_escolhida"]
    print("6. Os riscos que ficam, na foto de abril sem o rótulo")
    dia = como_no_dia(base, c.TESTE)
    passado = dia["data_referencia"].isin(c.TREINO + [c.VALIDACAO]).to_numpy()
    abril = (dia["data_referencia"] == c.TESTE).to_numpy()
    x = modulo.colunas(dia, passado)
    nota = treinar_e_dar_nota(modulo, x, dia[ALVO].to_numpy(dtype=float), passado, abril, config)
    foto = dia[abril].drop(columns=DESFECHO, errors="ignore").reset_index(drop=True)
    fora, motivos, _ = c.guarda(c.Preparo(dia[passado]), foto)
    topo = c.lista(nota, LISTA)
    rede = (foto["plano"] == "rede").to_numpy()
    print(f"   a guarda separa {int(fora.sum())} dos {len(foto)} clientes de abril; {int(rede.sum())} são do plano rede")
    print(f"   sem a guarda, o candidato põe {int(fora[topo].sum())} deles entre os 150 primeiros "
          f"({int(rede[topo].sum())} do plano rede)")
    print()
    return {"fora_na_lista": int(fora[topo].sum())}


def main(argv=None):
    p = argparse.ArgumentParser(description="Roteiro de revisão de um modelo candidato.")
    p.add_argument("pacote", help="pasta do pacote, por exemplo candidato-diego")
    args = p.parse_args(argv)
    try:
        pacote = abrir_pacote(AQUI / args.pacote)
        base = c.ler(pacote["modulo"].DADOS)
        if ALVO not in base.columns:
            raise ErroDoPacote("os dados do candidato não têm o rótulo cancelou_90d")
        print(f"Revisão do pacote {args.pacote}, de {pacote['ficha']['autor']}: {pacote['ficha']['modelo']}")
        print()
        parte1(pacote)
        parte2(pacote, base)
        tres = parte3(pacote, base)
        parte4(pacote, base)
        parte5(pacote, base, tres["uso"])
        parte6(pacote, base)
    except ErroDoPacote as erro:
        print(f"revisao.py: {erro}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
