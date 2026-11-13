"""Os testes do roteiro de revisão. Cada teste monta um caso com um defeito conhecido e espera que
o roteiro aponte o defeito, ou pare com erro, e que o caso sem defeito passe.

    python3 testar_revisao.py
"""
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

import caso as c
import revisao as r

AQUI = Path(__file__).resolve().parent
falhas = []


def teste(nome, cond):
    print(("ok     " if cond else "FALHOU ") + nome)
    if not cond:
        falhas.append(nome)


def erro(func, *args):
    try:
        func(*args)
    except r.ErroDoPacote as e:
        return str(e)
    return None


def registro(lugares, escolhido):
    return pd.DataFrame({"rodada": range(1, len(lugares) + 1), "avaliado_em": lugares,
                         "acerto_do_par": 0.7, "dos_150_primeiros": 60.0, "escolhido": escolhido})


def modulo(colunas):
    return types.SimpleNamespace(colunas=colunas)


def main():
    base = c.ler(r.PASTA_DADOS / "assinantes.csv")
    completo = c.ler(r.PASTA_DADOS / "assinantes_export_completo.csv")

    # o pacote
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        teste("pasta que não existe para com erro", "não achei a pasta" in (erro(r.abrir_pacote, tmp / "nada") or ""))
        p = tmp / "pacote"
        shutil.copytree(AQUI / "modelo-atual", p)
        (p / "ficha.json").unlink()
        teste("pacote sem ficha.json para com erro", "ficha.json" in (erro(r.abrir_pacote, p) or ""))
        shutil.copy2(AQUI / "modelo-atual" / "ficha.json", p / "ficha.json")
        ficha = json.loads((p / "ficha.json").read_text(encoding="utf-8"))
        sem = {k: v for k, v in ficha.items() if k != "resultado"}
        (p / "ficha.json").write_text(json.dumps(sem), encoding="utf-8")
        teste("ficha sem o resultado para com erro", "resultado" in (erro(r.abrir_pacote, p) or ""))
        outra = dict(ficha, config_escolhida={"modelo": "floresta", "sorteio": 9})
        (p / "ficha.json").write_text(json.dumps(outra), encoding="utf-8")
        teste("configuração escolhida que o candidato não tem para com erro",
              "config_escolhida" in (erro(r.abrir_pacote, p) or ""))
        (p / "ficha.json").write_text(json.dumps(ficha), encoding="utf-8")
        reg = pd.read_csv(p / "registro.csv")
        reg.drop(columns=["escolhido"]).to_csv(p / "registro.csv", index=False)
        teste("registro sem a coluna escolhido para com erro", "escolhido" in (erro(r.abrir_pacote, p) or ""))
        reg.to_csv(p / "registro.csv", index=False)
        teste("o mesmo pacote, consertado, abre", erro(r.abrir_pacote, p) is None)
        teste("revisao.py sai com código 2 para um pacote que não existe", r.main(["nao-existe"]) == 2)

    # a reprodução
    teste("o sha256 do export completo aponta o export completo",
          r.arquivo_do_sha(r.sha(r.PASTA_DADOS / "assinantes_export_completo.csv")) == "assinantes_export_completo.csv")
    teste("um sha256 que não é de arquivo do kit não aponta nada", r.arquivo_do_sha("0" * 64) is None)
    a = pd.read_csv(AQUI / "candidato-diego" / "registro.csv")
    b = a.copy()
    teste("registro igual: todas as rodadas iguais", r.registros_iguais(a, b) == len(a))
    b.loc[2, "acerto_do_par"] = 0.95
    teste("uma rodada alterada: uma a menos", r.registros_iguais(a, b) == len(a) - 1)
    teste("registro com outras colunas: nenhuma igual", r.registros_iguais(a, b.drop(columns=["sorteio"])) == 0)

    # o dia da foto
    dia = r.como_no_dia(completo, c.VALIDACAO)
    jan = dia["data_referencia"] == c.VALIDACAO
    teste("a base do dia de janeiro não tem a foto de abril", c.TESTE not in set(dia["data_referencia"]))
    teste("a foto de janeiro fica sem o rótulo, a data e o motivo do cancelamento",
          dia.loc[jan, r.DESFECHO].isna().all().all())
    out = dia["data_referencia"] == "2025-10-01"
    teste("a foto de outubro continua com o rótulo", dia.loc[out, "cancelou_90d"].notna().all())

    def com_fotos(b, treino):
        return pd.DataFrame({"fotos": b.groupby("id_cliente")["id_cliente"].transform("size").astype(float)})

    def com_rotulo(b, treino):
        return pd.DataFrame({"copia": b["cancelou_90d"].astype(float)})

    def limpa(b, treino):
        preparo = c.Preparo(b[treino])
        return pd.DataFrame(preparo.matriz(b), columns=preparo.nomes, index=b.index)

    def com_categoria(b, treino):
        return pd.get_dummies(b[["plano"]], dtype=float)

    mudam, depois = r.colunas_que_mudam(modulo(com_fotos), base, c.VALIDACAO)
    teste("coluna que conta fotos de depois é apontada", mudam.get("fotos", 0) > 0 and not depois)
    mudam, _ = r.colunas_que_mudam(modulo(com_rotulo), base, c.VALIDACAO)
    teste("coluna que copia o rótulo do dia é apontada em todas as linhas de janeiro",
          mudam.get("copia") == int((base["data_referencia"] == c.VALIDACAO).sum()))
    mudam, depois = r.colunas_que_mudam(modulo(limpa), base, c.VALIDACAO)
    teste("as colunas do caso.py não mudam", mudam == {} and depois == [])
    _, depois = r.colunas_que_mudam(modulo(com_categoria), base, c.VALIDACAO)
    teste("categoria que só aparece em abril é apontada", depois == ["plano_rede"])
    x = pd.DataFrame({"a": base["logins_30d"].astype(float), "ligou": base["recebeu_contato_30d"].astype(float)})
    teste("a coluna da ligação é achada pelo conteúdo, com outro nome", r.coluna_igual(x, base["recebeu_contato_30d"]) == ["ligou"])

    # o teste
    diego = r.uso_do_teste(registro(["teste (20%)"] * 6, [0, 0, 0, 0, 0, 1]))
    teste("seis rodadas no teste, escolha entre elas: escolha no teste",
          not diego["aberto_uma_vez"] and diego["rodadas_no_teste"] == 6 and "próprio teste" in diego["texto"])
    certo = r.uso_do_teste(registro(["validação"] * 3 + ["teste (abril)"], [0, 1, 0, 1]))
    teste("três na validação e uma no teste, no fim: aberto uma vez", certo["aberto_uma_vez"])
    duas = r.uso_do_teste(registro(["validação"] * 3 + ["teste (abril)"] * 2, [0, 1, 0, 0, 1]))
    teste("teste aberto duas vezes depois da escolha: não vale", not duas["aberto_uma_vez"] and "2 vezes" in duas["texto"])
    meio = r.uso_do_teste(registro(["validação", "teste (abril)", "validação"], [1, 0, 1]))
    teste("teste aberto no meio das rodadas: não vale", not meio["aberto_uma_vez"])
    nenhum = r.uso_do_teste(registro(["validação"] * 2, [1, 0]))
    teste("sem rodada no teste: zero", nenhum["rodadas_no_teste"] == 0 and not nenhum["aberto_uma_vez"])
    teste("erro típico da diferença entre 69 e 47,8 em 150: 8,4",
          abs(r.erro_tipico_da_diferenca(69, 47.8, 150) - 8.4) < 0.05)

    print("todos os testes passaram" if not falhas else f"{len(falhas)} teste(s) falharam")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
