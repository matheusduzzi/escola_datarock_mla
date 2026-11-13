"""Testa a guarda, a lista e as travas do sucesso.py.

    python3 testar_caso.py

Cada teste monta uma foto com um defeito e espera que a guarda pare ou separe a linha. Depois,
a mesma foto corrigida precisa passar. Sai com código 0 quando todos passam.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

import caso as c

AQUI = Path(__file__).resolve().parent
DADOS = AQUI.parent / "dados" / "assinantes" / "assinantes.csv"
BASE = c.ler(DADOS)
TREINO, VALIDACAO, TESTE = c.dividir(BASE)
PREPARO = c.Preparo(TREINO)
falhas = []


def foto():
    return VALIDACAO.drop(columns=["cancelou_90d"]).head(50).copy()


def espera_erro(f, trecho):
    try:
        f()
    except (c.ErroDaFoto, ValueError) as erro:
        return trecho in str(erro)
    return False


def teste(nome, ok):
    print(("ok    " if ok else "FALHA ") + nome)
    if not ok:
        falhas.append(nome)


def rodar(*args):
    r = subprocess.run([sys.executable, "-B", str(AQUI / "sucesso.py"), *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    boa = foto()
    fora, _, avisos = c.guarda(PREPARO, boa)
    teste("foto de janeiro sem defeito passa, sem linha separada nem aviso", not fora.any() and not avisos)

    teste("falta coluna: para", espera_erro(lambda: c.guarda(PREPARO, boa.drop(columns=["logins_30d"])), "faltam colunas"))
    duas = boa.copy()
    duas.loc[duas.index[0], "data_referencia"] = "2025-10-01"
    teste("duas datas na mesma foto: para", espera_erro(lambda: c.guarda(PREPARO, duas), "uma data de referência só"))
    repetido = boa.copy()
    repetido.loc[repetido.index[1], "id_cliente"] = repetido["id_cliente"].iloc[0]
    teste("cliente repetido: para", espera_erro(lambda: c.guarda(PREPARO, repetido), "cliente repetido"))
    vazio = boa.copy()
    vazio.loc[vazio.index[2], "chamados_30d"] = np.nan
    teste("coluna numérica vazia: para", espera_erro(lambda: c.guarda(PREPARO, vazio), "valor vazio"))

    nova = boa.copy()
    nova.loc[nova.index[3], "plano"] = "rede"
    fora, motivos, _ = c.guarda(PREPARO, nova)
    teste("plano que o treino não tem: só essa linha vai para a revisão",
          fora.sum() == 1 and fora[3] and "plano novo: rede" in motivos[3][0])
    alto = boa.copy()
    alto.loc[alto.index[4], "usuarios_contratados"] = 12
    fora, motivos, _ = c.guarda(PREPARO, alto)
    teste("usuários acima da faixa do treino: separa", fora.sum() == 1 and fora[4])
    menor, maior = PREPARO.faixa["meses_de_casa"]
    perto = boa.copy()
    perto.loc[perto.index[5], "meses_de_casa"] = maior + 2
    fora, _, _ = c.guarda(PREPARO, perto)
    teste(f"meses de casa 2 acima do maior do treino ({maior:g}): fica na lista", not fora.any())
    sem_nps = boa.copy()
    sem_nps["nps_ultimo"] = np.nan
    _, _, avisos = c.guarda(PREPARO, sem_nps)
    teste("NPS vazio em toda a foto: aviso", len(avisos) == 1 and "NPS vazio" in avisos[0])
    corrigida = nova.copy()
    corrigida.loc[corrigida.index[3], "plano"] = boa["plano"].iloc[3]
    fora, _, avisos = c.guarda(PREPARO, corrigida)
    teste("a mesma foto com o plano corrigido volta a passar", not fora.any() and not avisos)

    nota = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    y = np.array([1, 1, 0, 0, 1])
    teste("lista de 2 com empate na fronteira: 1 + 1/3", abs(c.quantos_na_lista(nota, y, 2) - 4 / 3) < 1e-12)
    teste("lista maior que a foto: erro", espera_erro(lambda: c.quantos_na_lista(nota, y, 6), "de 1 a 5"))
    teste("lista de zero: erro", espera_erro(lambda: c.quantos_na_lista(nota, y, 0), "de 1 a 5"))
    marcado = np.array([True, False, False, False, False])
    teste("linha separada pela guarda não entra na lista", list(c.lista(nota, 2, marcado)) == [1, 2])
    teste("lista maior que as linhas que sobram: erro",
          espera_erro(lambda: c.lista(nota, 5, marcado), "de 1 a 4"))

    teste("sem --abrir-teste, a foto de abril não sai da divisão", TESTE is None)
    codigo, saida = rodar("--abrir-teste", "--com-contato")
    teste("teste aberto com uma variante: recusa com código 2", codigo == 2 and "só da versão escolhida" in saida)
    codigo, saida = rodar("--lista", "--modelo", "logistica")
    teste("lista com outra versão: recusa com código 2", codigo == 2 and "só da versão escolhida" in saida)
    codigo, saida = rodar("--capacidade", "0")
    teste("capacidade 0: recusa com código 2", codigo == 2 and "--capacidade precisa ficar" in saida)
    codigo, saida = rodar("--capacidade", "100", "--grupo", "100")
    teste("grupo do tamanho da lista: recusa com código 2", codigo == 2 and "--grupo precisa ficar" in saida)
    codigo, saida = rodar("--dados", "nao-existe.csv")
    teste("arquivo que não existe: recusa com código 2", codigo == 2 and "não achei o arquivo" in saida)
    codigo, saida = rodar("--so-comparar", "--capacidade", "100")
    teste("a comparação roda e não abre abril", codigo == 0 and "2026-04-01" not in saida and "abril" not in saida)

    print(f"{len(falhas)} falha(s)" if falhas else "todos os testes passaram")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
