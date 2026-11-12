"""Testes das duas funções do meu_diagnostico.py.

    python3 testar_meu_diagnostico.py

Os oito pedidos de treino e os quatro de validação são inventados aqui, pequenos o bastante para
fazer a árvore à mão. Treino: o valor vai de 1 a 8 e cancelaram os de valor 3, 5, 6, 7 e 8.
"""
import sys

import numpy as np

import meu_diagnostico as m

X_TREINO = np.arange(1, 9, dtype=float).reshape(-1, 1)
Y_TREINO = np.array([0, 0, 1, 0, 1, 1, 1, 1])
X_VAL = np.array([[1.0], [2.0], [7.0], [8.0]])
Y_VAL = np.array([0, 1, 0, 1])


def perto(a, b):
    return abs(a - b) < 1e-9


def linha(min_folha):
    return m.tabela_generalizacao(X_TREINO, Y_TREINO, X_VAL, Y_VAL, [min_folha])[0]


def espera_erro(funcao, *args):
    try:
        funcao(*args)
    except ValueError:
        return
    raise AssertionError("era para parar com ValueError, e não parou")


# ---- tabela_generalizacao

def testar_folha_de_um_decora():
    r = linha(1)
    # folhas puras: [1,2] não cancelaram, [3] sim, [4] não, [5 a 8] sim
    assert r["min_folha"] == 1, f"min_folha: {r['min_folha']}"
    assert r["folhas"] == 4, f"folhas: {r['folhas']}, esperado 4"
    assert perto(r["treino"], 1.0), f"treino: {r['treino']}, esperado 1.0"
    # validação: valor 1 e 2 recebem nota 0; 7 e 8 recebem 1; um par ganho, um perdido, dois empates
    assert perto(r["validacao"], 0.5), f"validação: {r['validacao']}, esperado 0.5"
    assert perto(r["queda"], 0.5), f"queda: {r['queda']}, esperado 0.5"


def testar_folha_de_quatro():
    r = linha(4)
    # só cabe a divisão no meio: [1 a 4] com 1 de 4 cancelado (nota 0,25) e [5 a 8] com 4 de 4
    assert r["folhas"] == 2, f"folhas: {r['folhas']}, esperado 2"
    # 15 pares no treino: 12 ganhos e 3 empates (meio cada) = 13,5 / 15
    assert perto(r["treino"], 0.9), f"treino: {r['treino']}, esperado 0.9"
    assert perto(r["validacao"], 0.5), f"validação: {r['validacao']}, esperado 0.5"
    assert perto(r["queda"], 0.4), f"queda: {r['queda']}, esperado 0.4"


def testar_folha_maior_que_metade_nao_divide():
    r = linha(5)
    # com 8 pedidos, nenhuma divisão deixa 5 de cada lado: uma folha só, a mesma nota para todos
    assert r["folhas"] == 1, f"folhas: {r['folhas']}, esperado 1"
    assert perto(r["treino"], 0.5) and perto(r["validacao"], 0.5), f"treino {r['treino']}, validação {r['validacao']}"
    assert perto(r["queda"], 0.0), f"queda: {r['queda']}, esperado 0"


def testar_ordem_da_lista():
    t = m.tabela_generalizacao(X_TREINO, Y_TREINO, X_VAL, Y_VAL, [5, 1, 4])
    assert [r["min_folha"] for r in t] == [5, 1, 4], f"ordem: {[r['min_folha'] for r in t]}"
    assert [r["folhas"] for r in t] == [1, 4, 2], f"folhas: {[r['folhas'] for r in t]}"


def testar_min_folhas_vazia():
    espera_erro(m.tabela_generalizacao, X_TREINO, Y_TREINO, X_VAL, Y_VAL, [])


def testar_min_folha_zero():
    espera_erro(m.tabela_generalizacao, X_TREINO, Y_TREINO, X_VAL, Y_VAL, [5, 0])


def testar_min_folha_fracionaria():
    espera_erro(m.tabela_generalizacao, X_TREINO, Y_TREINO, X_VAL, Y_VAL, [2.5])


def testar_tamanhos_treino():
    espera_erro(m.tabela_generalizacao, X_TREINO, Y_TREINO[:-1], X_VAL, Y_VAL, [1])


def testar_tamanhos_validacao():
    espera_erro(m.tabela_generalizacao, X_TREINO, Y_TREINO, X_VAL[:-1], Y_VAL, [1])


# ---- escolher

def _t(*linhas):
    return [{"min_folha": mf, "folhas": f, "treino": tr, "validacao": va, "queda": tr - va}
            for mf, f, tr, va in linhas]


def testar_escolhe_pela_validacao():
    tabela = _t((1, 50, 1.00, 0.60), (5, 30, 0.95, 0.62), (10, 20, 0.85, 0.70), (20, 10, 0.80, 0.68))
    r = m.escolher(tabela)
    assert r["min_folha"] == 10, f"escolheu min_folha {r['min_folha']}; o maior acerto na validação é o de 10"


def testar_empate_fica_a_mais_simples():
    tabela = _t((1, 50, 1.00, 0.60), (5, 20, 0.90, 0.70), (10, 10, 0.85, 0.70), (20, 5, 0.80, 0.65))
    r = m.escolher(tabela)
    assert r["min_folha"] == 10, f"escolheu min_folha {r['min_folha']}; no empate fica a de menos folhas (10)"


def testar_tabela_vazia():
    espera_erro(m.escolher, [])


def testar_acerto_fora_de_0_a_1():
    espera_erro(m.escolher, _t((1, 50, 1.00, 1.20)))
    espera_erro(m.escolher, _t((1, 50, -0.10, 0.60)))


def main():
    testes = [(nome, f) for nome, f in globals().items() if nome.startswith("testar_")]
    falhas = 0
    for nome, teste in testes:
        try:
            teste()
        except NotImplementedError as erro:
            print(f"NotImplementedError: {erro}")
            sys.exit(1)
        except Exception as erro:
            falhas += 1
            print(f"FALHOU {nome}: {type(erro).__name__}: {erro}")
    if falhas:
        print(f"{falhas} de {len(testes)} testes falharam")
        sys.exit(1)
    print(f"ok: {len(testes)} testes passaram")


if __name__ == "__main__":
    main()
