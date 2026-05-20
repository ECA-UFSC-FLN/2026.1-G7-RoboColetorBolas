"""
BallCollectionPlanner.py — Rota Ótima de Recolha de Bolas UFSC/FEUP
=====================================================================
Dado o conjunto de posições de bolas detetadas em todo o campo e a posição
atual do robô, calcula a sequência de waypoints que minimiza a distância
total percorrida para recolher todas as bolas.

Algoritmo (duas fases):
  1. Nearest-Neighbor greedy — constrói a rota partindo do robô, sempre
     indo para a bola mais próxima ainda não visitada.  O(n²), muito rápido.
  2. 2-opt local search — melhora iterativamente a rota tentando inverter
     segmentos.  Para n < 80 bolas converge em < 20 ms.

Uso típico em GraphProcessor.py:

    from _PLANNING.ball_collection_planner import BallCollectionPlanner

    planner = BallCollectionPlanner()

    # Quando o threshold é atingido:
    waypoints = planner.planear(
        pos_robo=(cx_metros, cy_metros),
        bolas=lista_de_(x,y)_em_metros,
    )
    # waypoints: list[(x,y)] na ordem de visita.
    # Depois iteramos: planner.avancar_waypoint() quando o robô chegar
    # a cada ponto.

    ponto_atual = planner.waypoint_atual()
    planner.avancar_waypoint()
    concluido  = planner.concluido()
"""

import math
import os
import sys
import time
from typing import List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from _COMMON.logging_utils import log as _log

MOD = "BALL_PLANNER"

Ponto = Tuple[float, float]


# ─────────────────────────────────────────────
#  FUNÇÕES GEOMÉTRICAS INTERNAS
# ─────────────────────────────────────────────
def _dist(a: Ponto, b: Ponto) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nn_route(inicio: Ponto, pontos: List[Ponto]) -> List[Ponto]:
    """Nearest-neighbor greedy: começa no `inicio`, vai sempre ao mais próximo."""
    if not pontos:
        return []
    restantes = list(pontos)
    rota: List[Ponto] = []
    atual = inicio
    while restantes:
        idx = min(range(len(restantes)), key=lambda i: _dist(atual, restantes[i]))
        ponto = restantes.pop(idx)
        rota.append(ponto)
        atual = ponto
    return rota


def _custo_total(inicio: Ponto, rota: List[Ponto]) -> float:
    if not rota:
        return 0.0
    total = _dist(inicio, rota[0])
    for i in range(len(rota) - 1):
        total += _dist(rota[i], rota[i + 1])
    return total


def _dois_opt(inicio: Ponto, rota: List[Ponto], max_passes: int = 100) -> List[Ponto]:
    """
    2-opt local search: inverte segmentos [i..j] se isso reduzir o custo.
    Repete até não melhorar ou atingir max_passes.
    """
    melhor = list(rota)
    custo  = _custo_total(inicio, melhor)
    n      = len(melhor)

    for _ in range(max_passes):
        melhorou = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                nova  = melhor[:i] + melhor[i:j + 1][::-1] + melhor[j + 1:]
                c_nova = _custo_total(inicio, nova)
                if c_nova < custo - 1e-9:
                    melhor   = nova
                    custo    = c_nova
                    melhorou = True
                    break          # reinicia o loop externo mais cedo
            if melhorou:
                break
        if not melhorou:
            break

    return melhor


def _deduplicar(bolas: List[Ponto], raio_m: float = 0.05) -> List[Ponto]:
    """Remove bolas duplicadas dentro de `raio_m` metros."""
    unicas: List[Ponto] = []
    for b in bolas:
        if all(_dist(b, u) > raio_m for u in unicas):
            unicas.append(b)
    return unicas


# ─────────────────────────────────────────────
#  CLASSE PRINCIPAL
# ─────────────────────────────────────────────
class BallCollectionPlanner:
    """
    Gere o estado da trajetória global:
      - calcula a rota quando chamado planear()
      - mantém o índice do waypoint atual
      - sinaliza quando a rota está concluída
    """

    def __init__(self):
        self._waypoints: List[Ponto] = []
        self._idx:       int         = 0
        self._inicio:    Optional[Ponto] = None
        self._custo_m:   float       = 0.0
        self._ativa:     bool        = False

    # ── API pública ────────────────────────────────────────────────

    def planear(
        self,
        pos_robo: Ponto,
        bolas:    List[Ponto],
        otimizar: bool = True,
    ) -> List[Ponto]:
        """
        Recebe a posição do robô (metros) e a lista de bolas (metros).
        Calcula e guarda internamente a rota.  Devolve a lista de waypoints.

        Parâmetros:
            pos_robo  — (x, y) do centro do robô em metros
            bolas     — lista de (x, y) detetadas pela visão (já em metros)
            otimizar  — se True aplica 2-opt (recomendado; < 20 ms para n ≤ 80)
        """
        t0 = time.perf_counter()

        bolas_u = _deduplicar(bolas, raio_m=0.05)
        if not bolas_u:
            _log(MOD, "AVISO", "planear() chamado com lista de bolas vazia.")
            self._waypoints = []
            self._idx       = 0
            self._ativa     = False
            return []

        rota = _nn_route(pos_robo, bolas_u)
        custo_nn = _custo_total(pos_robo, rota)

        if otimizar and len(rota) > 2:
            rota = _dois_opt(pos_robo, rota)

        custo_final = _custo_total(pos_robo, rota)
        dt_ms = (time.perf_counter() - t0) * 1000

        _log(MOD, "EVENTO",
             f"Rota calculada: {len(rota)} waypoints | "
             f"dist={custo_final:.2f}m (NN={custo_nn:.2f}m, "
             f"melhoria={100*(custo_nn-custo_final)/max(custo_nn,1e-9):.1f}%) | "
             f"tempo={dt_ms:.1f}ms")

        self._waypoints = rota
        self._idx       = 0
        self._inicio    = pos_robo
        self._custo_m   = custo_final
        self._ativa     = True
        return list(rota)

    def waypoint_atual(self) -> Optional[Ponto]:
        """Devolve o waypoint atual (x, y) ou None se concluído/inativo."""
        if not self._ativa or self._idx >= len(self._waypoints):
            return None
        return self._waypoints[self._idx]

    def avancar_waypoint(self) -> bool:
        """
        Avança para o próximo waypoint.
        Devolve True se ainda há waypoints, False se a rota ficou concluída.
        """
        if not self._ativa:
            return False
        self._idx += 1
        restantes = len(self._waypoints) - self._idx
        if restantes <= 0:
            self._ativa = False
            _log(MOD, "EVENTO", "Rota global concluída — todos os waypoints visitados.")
            return False
        _log(MOD, "DEBUG",
             f"Waypoint {self._idx}/{len(self._waypoints)} "
             f"→ ({self._waypoints[self._idx][0]:.3f}m, "
             f"{self._waypoints[self._idx][1]:.3f}m) | "
             f"restam {restantes}")
        return True

    def concluido(self) -> bool:
        """True se a rota foi iniciada e todos os waypoints foram visitados."""
        return self._ativa is False and len(self._waypoints) > 0 and self._idx >= len(self._waypoints)

    def cancelar(self):
        """Cancela a rota em curso (ex: novas bolas detetadas, reset)."""
        _log(MOD, "AVISO", "Rota global cancelada.")
        self._waypoints = []
        self._idx       = 0
        self._ativa     = False

    # ── Propriedades de estado ─────────────────────────────────────

    @property
    def ativa(self) -> bool:
        return self._ativa

    @property
    def n_waypoints(self) -> int:
        return len(self._waypoints)

    @property
    def idx_atual(self) -> int:
        return self._idx

    @property
    def waypoints_restantes(self) -> int:
        return max(0, len(self._waypoints) - self._idx)

    @property
    def custo_total_m(self) -> float:
        return self._custo_m

    def resumo(self) -> dict:
        """Dicionário de estado para incluir no broadcast do GraphProcessor."""
        return {
            "planner_ativo":       self._ativa,
            "waypoint_idx":        self._idx,
            "waypoints_total":     len(self._waypoints),
            "waypoints_restantes": self.waypoints_restantes,
            "custo_total_m":       round(self._custo_m, 3),
        }


# ─────────────────────────────────────────────
#  DEMO / TESTE RÁPIDO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)

    bolas_teste = [(random.uniform(0, 10), random.uniform(0, 6)) for _ in range(20)]
    robo_pos    = (0.5, 3.0)

    p = BallCollectionPlanner()
    rota = p.planear(robo_pos, bolas_teste, otimizar=True)

    print(f"\nRota ({p.n_waypoints} pontos, custo total = {p.custo_total_m:.2f} m):")
    for i, (x, y) in enumerate(rota):
        print(f"  {i+1:2d}. ({x:.3f}, {y:.3f})")

    print("\nSimulação de avanço:")
    while not p.concluido():
        wp = p.waypoint_atual()
        print(f"  → robô chegou a {wp}")
        p.avancar_waypoint()
    print("Rota concluída!")




