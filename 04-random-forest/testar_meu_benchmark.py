"""Testes das duas funções do meu_benchmark.py.

    python3 testar_meu_benchmark.py

As linhas são inventadas aqui, pequenas o bastante para fazer as contas à mão:

    config A: sorteios 0, 1, 2; validação 0,60, 0,62 e 0,64; fora da amostra 0,58, 0,60 e 0,62;
              segundos 1, 1,5 e 5 (mediana 1,5; a média seria 2,5)
    config B: sorteios 0 e 1; validação 0,63 e 0,63; fora 0,61 e 0,61; segundos 5 e 7 (mediana 6)
    config C: sorteios 0 e 1; validação 0,61 e 0,615; fora 0,60 e 0,60; segundos 0,5 e 0,7 (mediana 0,6)
"""
import sys

import meu_benchmark as m


def perto(a, b):
    return abs(a - b) < 1e-9


def linha(config, sorteio, validacao, fora, segundos):
    return {"config": config, "sorteio": sorteio, "validacao": validacao, "fora_da_amostra": fora,
            "segundos": segundos}


def linhas_abc():
    return [linha("A", 0, 0.60, 0.58, 1.0), linha("B", 0, 0.63, 0.61, 5.0), linha("A", 1, 0.62, 0.60, 1.5),
            linha("C", 0, 0.61, 0.60, 0.5), linha("B", 1, 0.63, 0.61, 7.0), linha("A", 2, 0.64, 0.62, 5.0),
            linha("C", 1, 0.615, 0.60, 0.7)]


def por_config(resumo):
    return {r["config"]: r for r in resumo}


def espera_erro(funcao, *args):
    try:
        funcao(*args)
    except ValueError:
        return
    raise AssertionError("era para parar com ValueError, e não parou")


# ---- resumir

def testar_resumo_de_a():
    r = por_config(m.resumir(linhas_abc()))["A"]
    assert r["sorteios"] == 3, f"sorteios: {r['sorteios']}, esperado 3"
    assert perto(r["media"], 0.62), f"média: {r['media']}, esperado 0.62"
    assert perto(r["menor"], 0.60) and perto(r["maior"], 0.64), f"menor e maior: {r['menor']}, {r['maior']}"
    assert perto(r["diferenca"], 0.04), f"diferença: {r['diferenca']}, esperado 0.04 (maior menos menor)"
    assert perto(r["fora_da_amostra"], 0.60), f"fora da amostra: {r['fora_da_amostra']}, esperado 0.60"


def testar_tempo_pela_mediana():
    r = por_config(m.resumir(linhas_abc()))
    assert perto(r["A"]["segundos"], 1.5), f"segundos de A: {r['A']['segundos']}, esperado 1.5 (a mediana; a média dá 2.5)"
    assert perto(r["B"]["segundos"], 6.0), f"segundos de B: {r['B']['segundos']}, esperado 6.0"


def testar_ordem_da_primeira_aparicao():
    ordem = [r["config"] for r in m.resumir(linhas_abc())]
    assert ordem == ["A", "B", "C"], f"ordem: {ordem}, esperado ['A', 'B', 'C']"
    ordem = [r["config"] for r in m.resumir(linhas_abc()[1:] + linhas_abc()[:1])]
    assert ordem == ["B", "A", "C"], f"ordem com B primeiro: {ordem}, esperado ['B', 'A', 'C']"


def testar_linhas_vazias():
    espera_erro(m.resumir, [])


def testar_sorteio_repetido():
    linhas = linhas_abc() + [linha("C", 1, 0.62, 0.60, 0.6)]
    espera_erro(m.resumir, linhas)


def testar_um_sorteio_so():
    linhas = [l for l in linhas_abc() if not (l["config"] == "C" and l["sorteio"] == 1)]
    espera_erro(m.resumir, linhas)


def testar_linha_sem_sorteio():
    linhas = linhas_abc()
    del linhas[3]["sorteio"]
    espera_erro(m.resumir, linhas)


def testar_linha_sem_tempo():
    linhas = linhas_abc()
    del linhas[0]["segundos"]
    espera_erro(m.resumir, linhas)


def testar_acerto_acima_de_1():
    linhas = linhas_abc()
    linhas[0]["validacao"] = 62.0          # em pontos, e não de 0 a 1
    espera_erro(m.resumir, linhas)


def testar_fora_abaixo_de_0():
    linhas = linhas_abc()
    linhas[2]["fora_da_amostra"] = -0.1
    espera_erro(m.resumir, linhas)


def testar_tempo_zero():
    linhas = linhas_abc()
    linhas[1]["segundos"] = 0.0
    espera_erro(m.resumir, linhas)


# ---- escolher

def resumo_abc():
    return [{"config": "A", "media": 0.62, "diferenca": 0.04, "segundos": 1.5},
            {"config": "B", "media": 0.63, "diferenca": 0.0, "segundos": 6.0},
            {"config": "C", "media": 0.6125, "diferenca": 0.005, "segundos": 0.6}]


def testar_sem_limite_apertado_fica_a_melhor():
    e = m.escolher(resumo_abc(), 10.0)
    # cabem as três; B tem a maior média e diferença 0: só B empata com ela mesma
    assert e["config"] == "B", f"escolhida: {e['config']}, esperado B"


def testar_limite_tira_a_melhor():
    e = m.escolher(resumo_abc()[:2], 3.0)
    # B leva 6 s e não cabe; sobra A
    assert e["config"] == "A", f"escolhida: {e['config']}, esperado A (B não cabe em 3 s)"


def testar_empate_fica_a_mais_rapida():
    e = m.escolher(resumo_abc(), 3.0)
    # cabem A e C; a melhor é A (0,62, diferença 0,04); C (0,6125) fica a menos de 0,04 dela
    assert e["config"] == "C", f"escolhida: {e['config']}, esperado C (empata com A e é mais rápida)"


def testar_fora_do_empate_fica_a_melhor():
    resumo = resumo_abc()
    resumo[2]["media"] = 0.57          # agora C fica 0,05 abaixo de A, mais que a diferença de A
    e = m.escolher(resumo, 3.0)
    assert e["config"] == "A", f"escolhida: {e['config']}, esperado A"


def testar_empate_usa_a_diferenca_da_melhor():
    resumo = resumo_abc()
    resumo[0]["diferenca"] = 0.005     # A varia pouco: C, 0,0075 abaixo, já não empata
    resumo[2]["diferenca"] = 0.05      # a diferença de C não entra na regra
    e = m.escolher(resumo, 3.0)
    assert e["config"] == "A", f"escolhida: {e['config']}, esperado A"


def testar_limite_exato_cabe():
    e = m.escolher(resumo_abc(), 1.5)
    # A leva exatamente 1,5 s e cabe; empata com C, que é mais rápida
    assert e["config"] == "C", f"escolhida: {e['config']}, esperado C"
    e = m.escolher(resumo_abc()[:2], 1.5)
    assert e["config"] == "A", f"escolhida: {e['config']}, esperado A (1,5 s cabe em 1,5 s)"


def testar_nenhuma_cabe():
    espera_erro(m.escolher, resumo_abc(), 0.5)


def testar_resumo_vazio():
    espera_erro(m.escolher, [], 10.0)


def testar_limite_zero():
    espera_erro(m.escolher, resumo_abc(), 0.0)


def main():
    testes = [(nome, f) for nome, f in globals().items() if nome.startswith("testar_")]
    falhas = 0
    for nome, f in testes:
        try:
            f()
        except NotImplementedError:
            raise
        except Exception as erro:          # noqa: BLE001
            falhas += 1
            print(f"FALHOU {nome}: {erro}")
    if falhas:
        print(f"{falhas} de {len(testes)} testes falharam")
        return 1
    print(f"ok: {len(testes)} testes passaram")
    return 0


if __name__ == "__main__":
    sys.exit(main())
