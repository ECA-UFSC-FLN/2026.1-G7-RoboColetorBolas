"""
GraphProcessor.py — Acumulação de Bolas e Geração de Trajetória UFSC/FEUP
==========================================================================
Recebe posições retificadas (metros) do retificador via socket, deduplica
bolas espacialmente, acumula contagens em faixas horizontais espaçadas pela
largura configurada do robô, que se
ajustam ao polígono da quadra de atuação (definido pelos pontos de
calibração) e dispara uma trajetória quando uma faixa concentra peso
suficiente.

A conclusão do varrimento é detetada por VISÃO: o GraphProcessor
monitoriza a posição do robô (via ArUco no retificador) através de uma
máquina de estados de duas fases — aguarda_inicio → em_varrimento →
concluído. Não há sinal externo.

Fluxo:
  1. Lê calibração + pontos → constrói polígono e faixas
  2. Liga-se ao retificador (porta 6020) e pede JSONs em loop
  3. Para cada JSON: deduplica bolas (raio espacial), recalcula pesos
  4. Quando uma faixa cumpre o critério de disparo:
       - gera JSON + imagem da trajetória de referência (3 pontos)
       - entra em fase aguarda_inicio
       - quando o robô chega ao ponto inicial → fase em_varrimento
       - quando o robô chega ao ponto final → zera faixa e remove bolas
  5. Visualização ao vivo em matplotlib

Portas:
  6013  health-check
  6020  cliente do retificador (pedidos de JSON)

Critério de disparo:
  peso_faixa / total_acumulado >= THRESHOLD_PCT  E  peso_faixa >= K_MIN

Critério de chegada (via visão, ArUco):
  distância(ArUco frontal, ponto_alvo) <= DIST_CHEGADA_M
"""

import json
import sys
import time
import socket
import threading
import argparse
from collections import deque
from queue import Queue, Empty, Full
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from datetime import datetime
from multiprocessing.connection import Client, Listener

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
BASE_PATH = Path(__file__).resolve().parents[1]
PASTA_RES         = BASE_PATH / "resultados"
PASTA_CALIB       = PASTA_RES / "calibracao"
PASTA_POSICOES    = PASTA_RES / "posicoes"
PASTA_TOPDOWN     = PASTA_RES / "imagens_topdown"
PASTA_TRAJETORIAS = PASTA_RES / "trajetorias"

CALIB_FILE        = PASTA_CALIB / "homografia_calibracao.json"

# Portas
PORTA_HEALTH       = 6013
PORTA_RET_GRAFO    = 6020          # ligamos ao retificador para puxar JSONs
PORTA_BROADCAST    = 6021          # empurramos estado para o RobotController
AUTHKEY_GRAFO      = b"grafo_ufsc"
AUTHKEY_BROADCAST  = b"controlador_ufsc"

# ─────────────────────────────────────────────
#  PARÂMETROS COMPORTAMENTAIS
# ─────────────────────────────────────────────
# Os valores reais são lidos de resultados/configuracao/parametros.json no
# arranque (em main()). As constantes abaixo são DEFAULTS de fallback,
# usados apenas se o ficheiro não puder ser carregado.
import _CONFIG.system_parameters as _params
from _PLANNING.ball_collection_planner import BallCollectionPlanner as GlobalPlanner
_CFG: dict = {}                      # preenchido em main()

LARGURA_ROBO_M    = 0.40
RAIO_DEDUP_M      = 0.08
THRESHOLD_PCT     = 0.25
K_MIN             = 4
N_OBS_MIN_ESTAVEL = 3                # mín. de observações para bola contar
TEMPO_MIN_ESTAVEL_S = 1.0            # confirma estabilidade independentemente do FPS
VELOCIDADE_MAX_PARADA_M_S = 0.08     # acima disto a bola ainda está em movimento
RAIO_ESTACIONARIA_BOLA_M = 0.03      # permanência espacial para confirmar bola parada
TEMPO_EXPIRAR_BOLA_S = 1.5           # legado: bolas já não expiram antes do disparo
INTERVALO_VIS     = 0.10             # refresh da janela matplotlib (s) — fixo
TIMEOUT_RET       = 1.5              # timeout em cada pedido de JSON — fixo

# ─────────────────────────────────────────────
#  TOLERÂNCIAS DE CHEGADA AO PONTO  (afinar em testes!)
# ─────────────────────────────────────────────
# Quão perto (em metros) tem de estar o ArUco frontal para considerarmos
# que "chegou" a um ponto. Avaliar em campo: depende do tamanho da quadra,
# da precisão da homografia e da estabilidade da deteção do ArUco.
TOLERANCIA_DISTANCIA_AO_PONTO = 0.20   # metros

# Quão alinhado tem de estar o vetor (traseira→frente) com o vetor de
# varrimento (inicial→final) para o robô avançar para a fase de varrimento.
# Só é verificado na chegada ao PONTO INICIAL — na chegada ao FINAL não
# importa a orientação.
TOLERANCIA_ANGULO_GRAUS       = 15.0

# Tempo máximo de varrimento; se for excedido, liberta a faixa por
# segurança (provável perda de ArUco ou bloqueio mecânico).
TIMEOUT_VARRIMENTO_S          = 90.0
RAIO_RECOLHA_BOLA_M           = 0.18

# ── Modo de operação (lido de parametros.json em main()) ──────
# "FAIXAS" — comportamento original: disparo por zona
# "GLOBAL" — threshold no campo todo + trajetória TSP
MODO_OPERACAO  = "FAIXAS"
K_MIN_GLOBAL   = 10

ALFABETO          = "abcdefghijklmnopqrstuvwxyz"


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
from _COMMON.logging_utils import log as _log

MOD = "GRAFO"

def log(nivel: str, msg: str):
    """Atalho local: encapsula bolas_log.log com o módulo fixo."""
    _log(MOD, nivel, msg)


# ─────────────────────────────────────────────
#  HEALTH SERVER
# ─────────────────────────────────────────────
def iniciar_health_server(porta: int = PORTA_HEALTH):
    def _serve():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("localhost", porta))
            srv.listen(5)
            while True:
                try:
                    conn, _ = srv.accept()
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            log("AVISO", f"Health-server falhou na porta {porta}: {e}")
        finally:
            srv.close()

    threading.Thread(target=_serve, daemon=True).start()
    log("DEBUG", f"Health-check ativo na porta {porta}")


# ─────────────────────────────────────────────
#  GEOMETRIA DA QUADRA E FAIXAS
# ─────────────────────────────────────────────
def carregar_pontos_calibracao() -> tuple[list[tuple[float, float]], dict]:
    """
    Lê o homografia_calibracao.json (para ppm/x_min/y_min) e o
    pontos_<timestamp>.json mais recente (para os pontos reais em metros).
    Devolve (pontos_metros, calib_dict).
    """
    if not CALIB_FILE.exists():
        raise FileNotFoundError(f"Calibração não encontrada: {CALIB_FILE}")

    with open(CALIB_FILE) as f:
        calib = json.load(f)

    fichs = sorted(PASTA_CALIB.glob("pontos_*.json"))
    if not fichs:
        raise FileNotFoundError(
            "Nenhum ficheiro pontos_*.json encontrado em resultados/calibracao/"
        )
    fich_pontos = fichs[-1]   # mais recente
    log("DEBUG", f"Pontos de calibração: {fich_pontos.name}")

    with open(fich_pontos) as f:
        dados = json.load(f)

    pts = [(p["real_x_m"], p["real_y_m"]) for p in dados["pontos"]]
    return pts, calib


def construir_poligono_quadra(pontos: list[tuple[float, float]]) -> Polygon:
    """
    Constrói o polígono da quadra de atuação a partir dos pontos de
    calibração usando o convex hull — fica fechado, convexo e bem formado
    independentemente da ordem em que foram marcados.
    """
    if len(pontos) < 3:
        raise ValueError("Mínimo 3 pontos para definir um polígono.")

    # convex hull via shapely (simples e robusto)
    p = Polygon(pontos).convex_hull
    if not isinstance(p, Polygon):
        raise ValueError("Pontos de calibração não formam um polígono válido.")

    log("DEBUG", f"Polígono da quadra construído ({len(p.exterior.coords) - 1} vértices, "
              f"área={p.area:.2f}m²)")
    return p


class FaixaQuadra:
    """
    Faixa horizontal entre y_min e y_max, recortada pelo polígono da quadra.

    - peso: nº de bolas conhecidas atualmente atribuídas a esta faixa
    - em_execucao: True enquanto o robô está a apanhar nesta faixa
    """
    def __init__(self, idx: int, y_min: float, y_max: float, poligono_quadra: Polygon):
        self.id        = idx
        self.label     = ALFABETO[idx + 1] if idx + 1 < len(ALFABETO) else f"f{idx}"
        self.y_min     = y_min
        self.y_max     = y_max
        self.y_centro  = (y_min + y_max) / 2.0

        # Geometria efetiva: interseção da banda com o polígono
        bbox = poligono_quadra.bounds
        x_lo, x_hi = bbox[0] - 1.0, bbox[2] + 1.0
        banda = Polygon([(x_lo, y_min), (x_hi, y_min), (x_hi, y_max), (x_lo, y_max)])
        self.poligono = poligono_quadra.intersection(banda)

        # Posições inicial (mais à esquerda) e final (mais à direita) à altura y_centro
        linha = LineString([(x_lo, self.y_centro), (x_hi, self.y_centro)])
        seg = poligono_quadra.intersection(linha)
        if seg.is_empty:
            # fallback: usa bounds
            self.x_inicial = bbox[0]
            self.x_final   = bbox[2]
        else:
            xs = [c[0] for c in seg.coords] if hasattr(seg, "coords") else []
            if not xs:
                # MultiLineString
                xs = [c[0] for ls in seg.geoms for c in ls.coords]
            self.x_inicial = min(xs)
            self.x_final   = max(xs)

        self.pos_inicial = (self.x_inicial, self.y_centro)
        self.pos_final   = (self.x_final,   self.y_centro)

        self.peso        = 0
        self.em_execucao = False

    def contem(self, x: float, y: float) -> bool:
        return self.poligono.contains(Point(x, y))

    def __repr__(self):
        return (f"Faixa[{self.id}/{self.label}] y∈[{self.y_min:.2f},{self.y_max:.2f}] "
                f"peso={self.peso}{' ★' if self.em_execucao else ''}")


def construir_faixas(poligono: Polygon, largura_robo_m: float) -> list[FaixaQuadra]:
    """Constrói bandas horizontais consecutivas com a largura do robô.

    A última banda é recortada no limite superior da quadra quando a dimensão
    vertical não é um múltiplo exato da largura configurada.
    """
    if largura_robo_m <= 0:
        raise ValueError("A largura do robô deve ser maior que zero.")
    y_min, y_max = poligono.bounds[1], poligono.bounds[3]
    faixas = []
    y0 = y_min
    i = 0
    while y0 < y_max - 1e-9:
        y1 = min(y0 + largura_robo_m, y_max)
        faixas.append(FaixaQuadra(i, y0, y1, poligono))
        y0 = y1
        i += 1
    log("HUMANO", f"Quadra dividida em {len(faixas)} faixas horizontais "
                  f"pela largura do robô ({largura_robo_m*100:.1f}cm).")
    for f in faixas:
        log("DEBUG", f"  {f}  inicio=({f.x_inicial:.2f},{f.y_centro:.2f}) "
                    f"final=({f.x_final:.2f},{f.y_centro:.2f})")
    return faixas


# ─────────────────────────────────────────────
#  ESTADO PARTILHADO (bolas conhecidas + faixas)
# ─────────────────────────────────────────────
class BolaConhecida:
    __slots__ = (
        "x", "y", "n_obs", "primeira_visao", "ultima_visao", "faixa_id",
        "velocidade_m_s", "estacionaria_desde", "confirmada",
        "ancora_x", "ancora_y", "obs_estacionarias",
    )
    def __init__(self, x, y, faixa_id):
        agora = time.time()
        self.x = x
        self.y = y
        self.n_obs = 1
        self.primeira_visao = agora
        self.ultima_visao = agora
        self.faixa_id = faixa_id
        self.velocidade_m_s = float("inf")
        self.estacionaria_desde = agora
        self.ancora_x = x
        self.ancora_y = y
        self.obs_estacionarias = 1
        self.confirmada = False

    def atualizar(self, x: float, y: float, agora: float):
        if self.confirmada:
            # Depois de confirmada, preserva exatamente o ponto onde a bola
            # esteve parada, mesmo que mais tarde seja atingida ou movida.
            self.n_obs += 1
            self.ultima_visao = agora
            return

        dt = max(agora - self.ultima_visao, 1e-3)
        dist = ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5
        velocidade_inst = dist / dt
        self.velocidade_m_s = (
            velocidade_inst if self.velocidade_m_s == float("inf")
            else 0.65 * self.velocidade_m_s + 0.35 * velocidade_inst
        )

        # Em movimento damos mais peso à medição nova para evitar que a média
        # antiga "puxe" a bola para trás no gráfico ao vivo.
        alpha = 0.75 if self.velocidade_m_s > VELOCIDADE_MAX_PARADA_M_S else 0.35
        self.x = (1.0 - alpha) * self.x + alpha * x
        self.y = (1.0 - alpha) * self.y + alpha * y
        self.n_obs += 1
        self.ultima_visao = agora

        dist_ancora = ((x - self.ancora_x) ** 2 + (y - self.ancora_y) ** 2) ** 0.5
        if dist_ancora > RAIO_ESTACIONARIA_BOLA_M:
            # A candidata deslocou-se: começa uma nova janela de repouso.
            self.ancora_x = x
            self.ancora_y = y
            self.estacionaria_desde = agora
            self.obs_estacionarias = 1
        else:
            self.obs_estacionarias += 1

    def estavel(self, agora: float | None = None) -> bool:
        if self.confirmada:
            return True

        tempo_parada = self.ultima_visao - self.estacionaria_desde
        if (
            self.obs_estacionarias >= N_OBS_MIN_ESTAVEL
            and tempo_parada >= TEMPO_MIN_ESTAVEL_S
        ):
            self.confirmada = True
            log("DEBUG",
                f"Bola confirmada e persistente em ({self.x:.3f},{self.y:.3f})m "
                f"apos {self.obs_estacionarias} observacoes/{tempo_parada:.2f}s parada.")
        return self.confirmada


class EstadoGrafo:
    """Estado partilhado entre threads. Tudo atrás de um lock."""
    def __init__(self, faixas: list[FaixaQuadra], poligono: Polygon, calib: dict):
        self.lock      = threading.RLock()
        self.faixas    = faixas
        self.poligono  = poligono
        self.calib     = calib
        self.bolas     = []                 # list[BolaConhecida]
        self.robo      = {
            "frontal": None,
            "traseiro": None,
            "orientacao_graus": None,
            "qualidade_localizacao": {},
        }
        self.ultimo_indice_processado = -1
        self.ultima_latencia_retificador_ms = None
        self.disparo_ativo = None           # FaixaQuadra atualmente em execução
        self.fase_varrimento = None         # None | "aguarda_inicio" | "em_varrimento"
        self.t_disparo_iniciado = 0.0       # timestamp do início do disparo (timeout)
        self.ultimo_indice_disparo = 0      # numerador para os ficheiros gerados
        self.contador_disparos = 0
        self.aguarda_confirmacao_novo_conjunto = False

        # ── Modo GLOBAL ────────────────────────────────────────────
        # planner_global: instância do GlobalPlanner que gere os waypoints TSP
        # fase_global:    máquina de estados própria, independente da de faixas
        #   "aguardar"  — a contar bolas; YOLO ativo
        #   "calcular"  — threshold atingido; a computar rota (1 frame)
        #   "executar"  — a seguir waypoints; YOLO pausado
        self.planner_global = GlobalPlanner()
        self.fase_global    = "aguardar"    # só usado quando MODO_OPERACAO == "GLOBAL"
        self.t_global_iniciado = 0.0        # para timeout de segurança

    def total_peso(self) -> int:
        return sum(f.peso for f in self.faixas)

    def total_bolas_estaveis(self) -> int:
        """Total de bolas que já passaram o filtro de estabilidade (n_obs >= mínimo).
        Usado pelo modo GLOBAL para verificar o threshold global de disparo."""
        agora = time.time()
        return sum(1 for bc in self.bolas if bc.estavel(agora))


def solicitar_confirmacao_novo_conjunto(estado: EstadoGrafo, origem: str):
    """Pausa novos disparos até o operador confirmar no terminal.

    A leitura corre numa thread para não bloquear visão, broadcast ou a ordem
    de paragem enviada ao robô quando a trajetória termina.
    """
    with estado.lock:
        if estado.aguarda_confirmacao_novo_conjunto:
            return
        estado.aguarda_confirmacao_novo_conjunto = True

    def _aguardar():
        log("HUMANO",
            f"{origem} concluído. Retira/descarrega as bolas do robô e prime "
            "ENTER para autorizar o próximo conjunto.")
        try:
            input("\n  [CONFIRMAÇÃO] Prime ENTER para procurar o próximo conjunto... ")
        except EOFError:
            log("ERRO", "Terminal sem entrada disponível; novos conjuntos continuam bloqueados.")
            return

        with estado.lock:
            estado.aguarda_confirmacao_novo_conjunto = False
            if estado.fase_global == "aguardar_confirmacao":
                estado.fase_global = "aguardar"
            if estado.fase_varrimento == "aguardar_confirmacao":
                estado.fase_varrimento = None
        log("EVENTO", "Operador confirmou: procura do próximo conjunto autorizada.")

    threading.Thread(target=_aguardar, daemon=True,
                     name="confirmacao-novo-conjunto").start()


# ─────────────────────────────────────────────
#  DEDUPLICAÇÃO ESPACIAL
# ─────────────────────────────────────────────
def _faixa_de(y: float, faixas: list[FaixaQuadra]) -> int:
    """Devolve o id da faixa que contém este y, ou -1 se fora."""
    for f in faixas:
        if f.y_min <= y < f.y_max:
            return f.id
    # ponto exatamente no topo da última faixa
    if faixas and abs(y - faixas[-1].y_max) < 1e-9:
        return faixas[-1].id
    return -1


def deduplicar_e_atualizar(estado: EstadoGrafo, bolas_novas: list[dict]):
    """
    Para cada bola detetada no frame atual:
      - procura a bola conhecida mais próxima dentro de RAIO_DEDUP_M
      - se existe, atualiza posição com média acumulativa
      - senão, regista como nova
    Bolas fora do polígono são ignoradas.

    Antes do disparo, uma bola detetada dentro da quadra fica persistente:
    falhas temporárias do YOLO não removem a bola do estado. A remoção
    acontece só por recolha/proximidade do robô durante a execução.
    """
    with estado.lock:
        agora = time.time()

        for b in bolas_novas:
            x, y = float(b["x"]), float(b["y"])

            if not estado.poligono.contains(Point(x, y)):
                continue   # fora da quadra de atuação

            # encontrar bola conhecida mais próxima
            melhor = None
            melhor_d = float("inf")
            for bc in estado.bolas:
                d = ((bc.x - x) ** 2 + (bc.y - y) ** 2) ** 0.5
                if d < melhor_d:
                    melhor_d = d
                    melhor = bc

            if melhor is not None and melhor_d <= RAIO_DEDUP_M:
                melhor.atualizar(x, y, agora)
                # faixa pode ter mudado se a bola se moveu para a fronteira
                fid = _faixa_de(melhor.y, estado.faixas)
                if fid != melhor.faixa_id:
                    melhor.faixa_id = fid
            else:
                fid = _faixa_de(y, estado.faixas)
                if fid >= 0:
                    estado.bolas.append(BolaConhecida(x, y, fid))


def recalcular_pesos(estado: EstadoGrafo):
    """Recalcula o peso de cada faixa a partir das bolas conhecidas."""
    with estado.lock:
        agora = time.time()
        for f in estado.faixas:
            f.peso = 0
        for bc in estado.bolas:
            if 0 <= bc.faixa_id < len(estado.faixas):
                # faixas em execução não acumulam peso (já estão a ser limpas)
                f = estado.faixas[bc.faixa_id]
                if not f.em_execucao and bc.estavel(agora):
                    # Só bolas confirmadas por tempo + velocidade contam para
                    # o peso. Isto evita que 30 FPS transforme 3 frames em
                    # apenas 0.1 s de "estabilidade".
                    f.peso += 1


def verificar_disparo(estado: EstadoGrafo) -> FaixaQuadra | None:
    """
    Devolve a faixa que cumpre o critério de disparo, se alguma. Em empate,
    devolve a de maior peso.
    """
    with estado.lock:
        if estado.disparo_ativo is not None:
            return None
        total = estado.total_peso()
        if total == 0:
            return None
        candidatas = [f for f in estado.faixas
                      if not f.em_execucao
                      and f.peso >= K_MIN
                      and (f.peso / total) >= THRESHOLD_PCT]
        if not candidatas:
            return None
        return max(candidatas, key=lambda f: f.peso)


def consumar_disparo(estado: EstadoGrafo, faixa: FaixaQuadra, motivo: str = "concluído"):
    """Marca a faixa como livre sem apagar bolas que a visão ainda não confirmou como recolhidas."""
    with estado.lock:
        faixa.peso = 0
        faixa.em_execucao = False
        estado.disparo_ativo = None
        pedir_confirmacao = motivo == "concluído"
        estado.fase_varrimento = (
            "aguardar_confirmacao" if pedir_confirmacao else None
        )
        log("HUMANO",
            f"Faixa {faixa.label} {motivo}. Bolas mantidas no estado salvo as "
            f"recolhidas por proximidade ao robô.")
    if pedir_confirmacao:
        solicitar_confirmacao_novo_conjunto(estado, f"Conjunto da faixa {faixa.label}")


# ─────────────────────────────────────────────
#  DETECÇÃO DE CHEGADA DO ROBÔ (VISÃO / ArUco)
# ─────────────────────────────────────────────
def _posicao_robo(robo: dict) -> tuple[float, float] | None:
    """
    Devolve a posição usada pelo servidor para o robô: ArUco frontal.
    A orientação continua a usar frontal-traseiro em _vetor_robo().
    """
    qualidade = robo.get("qualidade_localizacao") or {}
    if str(qualidade.get("fonte", "ARUCO")).upper() == "COR" and not qualidade.get(
        "valida_controle", False
    ):
        return None
    f = robo.get("frontal")
    if not f:
        return None
    return (f["x"], f["y"])


def _vetor_robo(robo: dict) -> tuple[float, float] | None:
    """
    Vetor de orientação do robô = (frontal − traseiro), NÃO normalizado.
    None se faltar algum marcador.
    """
    qualidade = robo.get("qualidade_localizacao") or {}
    if str(qualidade.get("fonte", "ARUCO")).upper() == "COR" and not qualidade.get(
        "valida_controle", False
    ):
        return None
    f = robo.get("frontal")
    t = robo.get("traseiro")
    if not f or not t:
        return None
    return (f["x"] - t["x"], f["y"] - t["y"])


def _alinhamento_graus(v_alvo: tuple[float, float],
                       v_robo: tuple[float, float]) -> float | None:
    """
    Ângulo entre os vetores em graus (0 = perfeitamente alinhado).
    Independente de convenção de eixos — usa apenas produto escalar.
    None se algum vetor for nulo.
    """
    import math
    nv = (v_alvo[0]**2 + v_alvo[1]**2) ** 0.5
    nr = (v_robo[0]**2 + v_robo[1]**2) ** 0.5
    if nv < 1e-9 or nr < 1e-9:
        return None
    cos = (v_alvo[0]*v_robo[0] + v_alvo[1]*v_robo[1]) / (nv * nr)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _distancia(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def remover_bolas_recolhidas_por_robo(estado: EstadoGrafo) -> int:
    """
    Durante a execução a lista de bolas fica congelada face ao YOLO. A única
    remoção permitida é por evidência geométrica: ArUco frontal passou perto.
    """
    with estado.lock:
        pos_robo = _posicao_robo(estado.robo)
        if pos_robo is None or not estado.bolas:
            return 0

        mantidas = []
        removidas = []
        for bc in estado.bolas:
            if _distancia(pos_robo, (bc.x, bc.y)) <= RAIO_RECOLHA_BOLA_M:
                removidas.append(bc)
            else:
                mantidas.append(bc)

        if not removidas:
            return 0

        estado.bolas = mantidas
        for f in estado.faixas:
            f.peso = 0
        agora = time.time()
        for bc in estado.bolas:
            if 0 <= bc.faixa_id < len(estado.faixas) and bc.estavel(agora):
                estado.faixas[bc.faixa_id].peso += 1

        log("EVENTO",
            f"{len(removidas)} bola(s) removida(s) por proximidade ao robô "
            f"(raio={RAIO_RECOLHA_BOLA_M*100:.0f}cm).")
        return len(removidas)


def verificar_progresso_varrimento(estado: EstadoGrafo):
    """
    Chamada a cada novo pacote enquanto há disparo ativo. Avança a máquina
    de estados:
        aguarda_inicio  →  em_varrimento  (perto do ponto inicial E alinhado)
        em_varrimento   →  consumado      (perto do ponto final)
    Também aplica o timeout de segurança.
    """
    with estado.lock:
        faixa = estado.disparo_ativo
        if faixa is None:
            return

        # Timeout de segurança (ex: ArUco perdido)
        if (time.time() - estado.t_disparo_iniciado) > TIMEOUT_VARRIMENTO_S:
            log("AVISO", f"Timeout de varrimento ({TIMEOUT_VARRIMENTO_S:.0f}s) "
                         f"na faixa {faixa.label} — a libertar por segurança.")
            consumar_disparo(estado, faixa, motivo="libertada (timeout)")
            return

        pos_robo = _posicao_robo(estado.robo)
        if pos_robo is None:
            # Sem detecção do robô neste frame — só esperamos próximo frame
            return

        remover_bolas_recolhidas_por_robo(estado)

        if estado.fase_varrimento == "aguarda_inicio":
            d = _distancia(pos_robo, faixa.pos_inicial)
            if d > TOLERANCIA_DISTANCIA_AO_PONTO:
                return  # ainda não chegou

            # Está perto do ponto inicial — agora verifica orientação
            v_alvo = (faixa.pos_final[0] - faixa.pos_inicial[0],
                      faixa.pos_final[1] - faixa.pos_inicial[1])
            v_robo = _vetor_robo(estado.robo)
            ang = _alinhamento_graus(v_alvo, v_robo) if v_robo else None

            if ang is None:
                # Sem orientação — espera próximo frame
                return
            if ang > TOLERANCIA_ANGULO_GRAUS:
                # Chegou ao ponto mas ainda não está apontado para a direita.
                # Log apenas a primeira vez ou periodicamente, para não spammar.
                if not getattr(estado, "_avisou_alinhamento", False):
                    log("HUMANO", f"Robô junto ao início (d={d*100:.1f}cm) "
                                f"mas desalinhado ({ang:.1f}°>{TOLERANCIA_ANGULO_GRAUS:.0f}°). "
                                f"A aguardar orientação correta...")
                    estado._avisou_alinhamento = True
                return

            estado.fase_varrimento = "em_varrimento"
            estado._avisou_alinhamento = False
            log("EVENTO", f"Robô no ponto inicial da faixa {faixa.label} "
                           f"(d={d*100:.1f}cm, ângulo={ang:.1f}°) — VARRIMENTO INICIADO.")

        elif estado.fase_varrimento == "em_varrimento":
            d = _distancia(pos_robo, faixa.pos_final)
            if d <= TOLERANCIA_DISTANCIA_AO_PONTO:
                log("EVENTO", f"Robô no ponto final da faixa {faixa.label} "
                               f"(d={d*100:.1f}cm).")
                consumar_disparo(estado, faixa)


# ─────────────────────────────────────────────
#  GERAÇÃO DA TRAJETÓRIA (JSON + IMAGEM)
# ─────────────────────────────────────────────
def _ultimo_topdown_disponivel() -> Path | None:
    if not PASTA_TOPDOWN.exists():
        return None
    fichs = sorted(PASTA_TOPDOWN.glob("frame_*.jpg"))
    return fichs[-1] if fichs else None


def _m_para_px(x_m: float, y_m: float, calib: dict) -> tuple[int, int]:
    ppm   = calib["ppm"]
    x_min = calib.get("x_min", 0.0)
    y_min = calib.get("y_min", 0.0)
    return int(round((x_m - x_min) * ppm)), int(round((y_m - y_min) * ppm))


def gerar_artefactos_disparo(estado: EstadoGrafo, faixa: FaixaQuadra):
    """
    Sobre a última imagem top-down, desenha:
      - polígono da quadra
      - faixas (a que disparou destacada)
      - bolas conhecidas
      - vetor do robô atual (traseiro → frontal)
      - vetor de referência da trajetória (inicial → final da faixa)
    Guarda PNG + JSON em resultados/trajetorias/.
    """
    PASTA_TRAJETORIAS.mkdir(parents=True, exist_ok=True)

    with estado.lock:
        agora = time.time()
        estado.contador_disparos += 1
        idx = estado.contador_disparos
        calib = estado.calib
        bolas_snap = [(bc.x, bc.y,
                       bc.faixa_id == faixa.id,
                       bc.estavel(agora))
                      for bc in estado.bolas]
        robo = dict(estado.robo)
        poligono = estado.poligono
        faixas_snap = list(estado.faixas)

    # ── Carregar última top-down ─────────────────────────────
    fich_td = _ultimo_topdown_disponivel()
    if fich_td is not None:
        img = cv2.imread(str(fich_td))
    else:
        img = None

    # Se não houver top-down, criamos canvas vazio do tamanho certo
    out_w, out_h = calib.get("output_size_px", [800, 600])
    if img is None or img.shape[1] != out_w or img.shape[0] != out_h:
        img = np.full((out_h, out_w, 3), 30, dtype=np.uint8)
        log("AVISO", "Sem top-down compatível — a desenhar sobre canvas vazio.")

    # Escurecer ligeiramente para o overlay sobressair
    img = cv2.addWeighted(img, 0.65, np.zeros_like(img), 0.0, 0)

    # ── Polígono da quadra ───────────────────────────────────
    pts_pol = np.array(
        [_m_para_px(x, y, calib) for x, y in poligono.exterior.coords],
        dtype=np.int32
    )
    cv2.polylines(img, [pts_pol], True, (0, 220, 220), 2)

    # ── Faixas ───────────────────────────────────────────────
    for f in faixas_snap:
        if f.poligono.is_empty:
            continue
        coords = list(f.poligono.exterior.coords) if f.poligono.geom_type == "Polygon" \
                 else [c for g in f.poligono.geoms for c in g.exterior.coords]
        pts = np.array([_m_para_px(x, y, calib) for x, y in coords], dtype=np.int32)
        if f.id == faixa.id:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 200))
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        else:
            cv2.polylines(img, [pts], True, (80, 80, 80), 1)

        # rótulo
        cx_m = (f.x_inicial + f.x_final) / 2
        cx, cy = _m_para_px(cx_m, f.y_centro, calib)
        cv2.putText(img, f"{f.label} (peso={f.peso})", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255) if f.id == faixa.id else (180, 180, 180), 1)

    # ── Bolas conhecidas ─────────────────────────────────────
    # Estáveis (n_obs >= mínimo): círculo amarelo cheio, contam para o peso.
    # Instáveis (ainda em validação): círculo cinzento mais pequeno, não contam.
    for x, y, na_faixa, estavel in bolas_snap:
        px, py = _m_para_px(x, y, calib)
        if estavel:
            cor = (0, 255, 255) if na_faixa else (0, 215, 255)
            raio_in, raio_out = 7, 8
        else:
            cor = (120, 120, 120)
            raio_in, raio_out = 4, 5
        cv2.circle(img, (px, py), raio_in, cor, -1)
        cv2.circle(img, (px, py), raio_out, (0, 0, 0), 1)

    # ── Trajetória de referência: robô → inicial → final ─────
    # Segmento 1: posição atual do robô (ArUco frontal) → ponto inicial da faixa
    # Segmento 2: ponto inicial → ponto final (varrimento da faixa)
    px_ini, py_ini = _m_para_px(*faixa.pos_inicial, calib)
    px_fim, py_fim = _m_para_px(*faixa.pos_final,   calib)

    if robo.get("frontal"):
        rfx = robo["frontal"]["x"];  rfy = robo["frontal"]["y"]
        px_robo, py_robo = _m_para_px(rfx, rfy, calib)
        # Segmento de aproximação (cor diferente para distinguir do varrimento)
        cv2.arrowedLine(img, (px_robo, py_robo), (px_ini, py_ini),
                        (0, 200, 255), 2, tipLength=0.05)   # ciano
        cv2.putText(img, "APROXIMACAO", (px_robo + 8, py_robo - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    # Segmento de varrimento (verde, mais grosso — é o que importa)
    cv2.arrowedLine(img, (px_ini, py_ini), (px_fim, py_fim),
                    (0, 255, 0), 3, tipLength=0.04)
    cv2.circle(img, (px_ini, py_ini), 6, (0, 255, 0), -1)
    cv2.circle(img, (px_ini, py_ini), 8, (255, 255, 255), 1)
    cv2.putText(img, "VARRIMENTO", (px_ini + 8, py_ini - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Indicador de orientação alvo no ponto inicial — uma seta pequena
    # apontando "para a direita" (mesma direção do varrimento, mas curta)
    seta_len_px = 60
    norm = ((px_fim - px_ini)**2 + (py_fim - py_ini)**2) ** 0.5
    if norm > 1e-3:
        ux = (px_fim - px_ini) / norm
        uy = (py_fim - py_ini) / norm
        x_off = px_ini + int(ux * seta_len_px)
        y_off = py_ini + int(uy * seta_len_px)
        # offset perpendicular para não sobrepor a seta principal
        cv2.arrowedLine(img,
                        (px_ini - int(uy * 25), py_ini + int(ux * 25)),
                        (x_off - int(uy * 25), y_off + int(ux * 25)),
                        (255, 255, 0), 2, tipLength=0.3)
        cv2.putText(img, "orientacao alvo",
                    (px_ini - int(uy * 25) + 5, py_ini + int(ux * 25) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # ── Vetor de orientação atual do robô (traseiro → frontal) ──
    if robo.get("frontal") and robo.get("traseiro"):
        rfx = robo["frontal"]["x"];   rfy = robo["frontal"]["y"]
        rtx = robo["traseiro"]["x"];  rty = robo["traseiro"]["y"]
        ptx, pty = _m_para_px(rtx, rty, calib)
        pfx, pfy = _m_para_px(rfx, rfy, calib)
        cv2.arrowedLine(img, (ptx, pty), (pfx, pfy), (255, 100, 0), 3, tipLength=0.15)
        cv2.putText(img, "ROBO", (pfx + 8, pfy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

    # ── Cabeçalho informativo ────────────────────────────────
    header = f"DISPARO #{idx} | Faixa {faixa.label} | peso={faixa.peso} de {estado.total_peso() or '?'} | {datetime.now().strftime('%H:%M:%S')}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, header, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)

    # ── Guardar ──────────────────────────────────────────────
    fich_png = PASTA_TRAJETORIAS / f"trajetoria_{idx:04d}.png"
    ok, buf = cv2.imencode(".png", img)
    if ok:
        fich_png.write_bytes(buf.tobytes())
        log("DEBUG", f"Imagem da trajetória: {fich_png.name}")

    # ── JSON da trajetória (DOIS SEGMENTOS SEPARADOS) ────────
    pos_robo_atual = None
    if robo.get("frontal"):
        pos_robo_atual = {
            "x": round(robo["frontal"]["x"], 4),
            "y": round(robo["frontal"]["y"], 4),
            "referencia": "aruco_frontal",
        }
    pt_inicial = {"x": round(faixa.pos_inicial[0], 4),
                  "y": round(faixa.pos_inicial[1], 4)}
    pt_final   = {"x": round(faixa.pos_final[0],   4),
                  "y": round(faixa.pos_final[1],   4)}

    # Vetor de orientação alvo (unitário) — apontado para a direita,
    # ao longo do varrimento (ponto inicial → ponto final)
    v_alvo = (faixa.pos_final[0] - faixa.pos_inicial[0],
              faixa.pos_final[1] - faixa.pos_inicial[1])
    norma  = (v_alvo[0]**2 + v_alvo[1]**2) ** 0.5
    v_dir  = (round(v_alvo[0] / norma, 4),
              round(v_alvo[1] / norma, 4)) if norma > 1e-9 else (1.0, 0.0)

    saida = {
        "indice":            idx,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "faixa_id":          faixa.id,
        "faixa_label":       faixa.label,
        "y_min":             round(faixa.y_min, 4),
        "y_max":             round(faixa.y_max, 4),
        "tolerancia_distancia_ao_ponto_m": TOLERANCIA_DISTANCIA_AO_PONTO,
        "tolerancia_angulo_graus":         TOLERANCIA_ANGULO_GRAUS,
        "vetor_orientacao_alvo":           {"x": v_dir[0], "y": v_dir[1]},
        "segmentos": [
            {
                "ordem":      1,
                "tipo":       "aproximacao",
                "descricao":  "Posição atual do robô (ArUco frontal) → ponto inicial da faixa. "
                              "Ao chegar, o robô deve estar APONTADO PARA A DIREITA "
                              "(alinhado com o vetor de orientação alvo).",
                "origem":     pos_robo_atual,
                "destino":    pt_inicial,
                "exige_orientacao_alvo": True,
            },
            {
                "ordem":      2,
                "tipo":       "varrimento",
                "descricao":  "Ponto inicial → ponto final da faixa. "
                              "Movimento em linha reta, recolhendo bolas.",
                "origem":     pt_inicial,
                "destino":    pt_final,
                "exige_orientacao_alvo": False,
            },
        ],
        "peso_faixa":        faixa.peso,
        "peso_total":        estado.total_peso(),
        "n_bolas_estimadas": faixa.peso,
        "robo": {
            "frontal":          robo.get("frontal"),
            "traseiro":         robo.get("traseiro"),
            "orientacao_graus": robo.get("orientacao_graus"),
        },
    }
    fich_json = PASTA_TRAJETORIAS / f"trajetoria_{idx:04d}.json"
    with open(fich_json, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=4, ensure_ascii=False)
    log("DEBUG", f"JSON da trajetória: {fich_json.name}")

    return idx


# ─────────────────────────────────────────────
#  MODO GLOBAL — THRESHOLD E PROGRESSO DE WAYPOINTS
# ─────────────────────────────────────────────

def verificar_disparo_global(estado: EstadoGrafo) -> bool:
    """
    Verifica se o total de bolas estáveis atingiu K_MIN_GLOBAL.
    Só actua quando fase_global == "aguardar".
    Devolve True se o disparo for iniciado (transição para "calcular").
    """
    with estado.lock:
        if estado.fase_global != "aguardar":
            return False
        n = estado.total_bolas_estaveis()
        if n < K_MIN_GLOBAL:
            return False
        estado.fase_global = "calcular"
        log("EVENTO",
            f"★ DISPARO GLOBAL! {n} bolas estáveis ≥ {K_MIN_GLOBAL} — "
            f"a calcular trajetória TSP...")
        return True


def _bolas_estaveis_metros(estado: EstadoGrafo) -> list[tuple[float, float]]:
    """Devolve lista de (x, y) em metros das bolas estáveis."""
    agora = time.time()
    return [(bc.x, bc.y) for bc in estado.bolas if bc.estavel(agora)]


def calcular_rota_global(estado: EstadoGrafo):
    """
    Fase "calcular": pede a posição do robô e lança o GlobalPlanner.
    Se não há ArUco neste frame, mantém fase "calcular" e tenta no
    próximo frame (sem bloquear).
    Transita para "executar" quando a rota estiver pronta.
    """
    with estado.lock:
        robo = dict(estado.robo)
        bolas = _bolas_estaveis_metros(estado)

    frontal  = robo.get("frontal")
    if not frontal:
        log("AVISO", "Modo GLOBAL: aguardando ArUco para calcular rota...")
        return

    cx = frontal["x"]
    cy = frontal["y"]

    if not bolas:
        log("AVISO", "Modo GLOBAL: sem bolas estáveis para planear rota — reset.")
        with estado.lock:
            estado.fase_global = "aguardar"
        return

    estado.planner_global.planear((cx, cy), bolas, otimizar=True)

    with estado.lock:
        estado.fase_global        = "executar"
        estado.t_global_iniciado  = time.time()
    log("HUMANO",
        f"Modo GLOBAL: rota com {estado.planner_global.n_waypoints} waypoints "
        f"({estado.planner_global.custo_total_m:.2f}m). YOLO pausado.")


def verificar_progresso_global(estado: EstadoGrafo):
    """
    Chamada a cada frame enquanto fase_global == "executar".
    Verifica se o robô chegou ao waypoint atual e avança se sim.
    Quando todos os waypoints forem visitados, volta a "aguardar".
    Também aplica timeout de segurança igual ao de faixas.
    """
    with estado.lock:
        if estado.fase_global != "executar":
            return

        # Timeout de segurança
        if (time.time() - estado.t_global_iniciado) > TIMEOUT_VARRIMENTO_S:
            log("AVISO",
                f"Timeout GLOBAL ({TIMEOUT_VARRIMENTO_S:.0f}s) — "
                f"a libertar por segurança e resetar.")
            estado.planner_global.cancelar()
            estado.fase_global = "aguardar"
            return

        wp = estado.planner_global.waypoint_atual()
        if wp is None:
            # rota já concluída
            _concluir_global(estado)
            return

        pos_robo = _posicao_robo(estado.robo)
        if pos_robo is None:
            return  # sem ArUco neste frame

        remover_bolas_recolhidas_por_robo(estado)

        dist = _distancia(pos_robo, wp)
        if dist <= TOLERANCIA_DISTANCIA_AO_PONTO:
            tem_mais = estado.planner_global.avancar_waypoint()
            if not tem_mais:
                _concluir_global(estado)


def _concluir_global(estado: EstadoGrafo):
    """Limpa a rota GLOBAL e aguarda autorização para um novo conjunto."""
    for f in estado.faixas:
        f.peso = 0
    recalcular_pesos(estado)
    n_restantes = len(estado.bolas)
    estado.planner_global.cancelar()
    estado.fase_global = "aguardar_confirmacao"
    log("EVENTO",
        f"Rota GLOBAL concluída — {n_restantes} bola(s) ainda no estado. "
        f"A aguardar confirmação do operador antes de retomar o YOLO.")
    solicitar_confirmacao_novo_conjunto(estado, "Conjunto GLOBAL")


# ─────────────────────────────────────────────
#  BROADCASTER DE ESTADO (push para o RobotController)
# ─────────────────────────────────────────────
# Servidor que aceita ligações persistentes e empurra o estado actual
# (posição/orientação do robô + fase de varrimento + alvo de destino)
# sempre que chega um novo pacote do retificador.
_broadcast_clientes: list = []
_broadcast_lock = threading.Lock()


def loop_servidor_broadcast(parar: threading.Event):
    """
    Aceita ligações de clientes que querem receber o estado em tempo real.
    Clientes conhecidos:
      - RobotController (calcula comandos motores em função do estado)
      - VisionProcessing (desliga YOLO durante disparo para acelerar ArUco)
    Múltiplos clientes simultâneos suportados.
    """
    try:
        listener = Listener(("localhost", PORTA_BROADCAST), authkey=AUTHKEY_BROADCAST)
    except OSError as e:
        log("ERRO", f"Não foi possível abrir porta {PORTA_BROADCAST}: {e}")
        return

    log("DEBUG", f"Broadcaster de estado ativo na porta {PORTA_BROADCAST}")

    while not parar.is_set():
        try:
            conn = listener.accept()
        except Exception:
            continue
        with _broadcast_lock:
            _broadcast_clientes.append(conn)
        log("DEBUG", f"Cliente ligado ao broadcaster ({len(_broadcast_clientes)} ativo(s))")


def broadcast_estado(estado: "EstadoGrafo"):
    """
    Empurra um snapshot do estado actual a todos os clientes ligados.
    Clientes que tenham desligado são removidos silenciosamente.

    Campos comuns a ambos os modos:
      timestamp, robo, modo_operacao, fase, faixa_label, alvo_destino

    Campos adicionais no modo GLOBAL:
      waypoint_idx, waypoints_total, waypoints_restantes
    """
    with _broadcast_lock:
        if not _broadcast_clientes:
            return

    with estado.lock:
        robo         = dict(estado.robo)
        modo_op      = MODO_OPERACAO

        if modo_op == "GLOBAL":
            fase_global  = estado.fase_global
            # VisionProcessing pausa YOLO durante movimento e confirmação.
            if fase_global == "executar":
                fase_bc = "global_executar"
            elif fase_global == "aguardar_confirmacao":
                fase_bc = "aguardar_confirmacao"
            else:
                fase_bc = None
            faixa_label  = None
            wp           = estado.planner_global.waypoint_atual()
            alvo_destino = {"x": wp[0], "y": wp[1]} if wp else None
            waypoints_completos = [
                {"x": float(x), "y": float(y)}
                for x, y in estado.planner_global._waypoints
            ]
            extra = {
                "waypoint_idx":        estado.planner_global.idx_atual,
                "waypoints_total":     estado.planner_global.n_waypoints,
                "waypoints_restantes": estado.planner_global.waypoints_restantes,
                "waypoints":           waypoints_completos,
                "trajetoria_id":       (
                    f"global:{int(estado.t_global_iniciado * 1000)}"
                    if fase_global == "executar" else None
                ),
            }
        else:
            # Modo FAIXAS — igual ao original
            fase_bc      = estado.fase_varrimento
            faixa_label  = None
            alvo_destino = None
            if estado.disparo_ativo is not None:
                faixa_label = estado.disparo_ativo.label
                if fase_bc == "aguarda_inicio":
                    alvo_destino = {"x": estado.disparo_ativo.pos_inicial[0],
                                    "y": estado.disparo_ativo.pos_inicial[1]}
                elif fase_bc == "em_varrimento":
                    alvo_destino = {"x": estado.disparo_ativo.pos_final[0],
                                    "y": estado.disparo_ativo.pos_final[1]}
            extra = {}

    pacote = {
        "timestamp":     time.time(),
        "robo":          robo,
        "modo_operacao": modo_op,
        "fase":          fase_bc,
        "faixa_label":   faixa_label,
        "alvo_destino":  alvo_destino,
        "indice_visao":  estado.ultimo_indice_processado,
        "latencia_retificador_ms": estado.ultima_latencia_retificador_ms,
        **extra,
    }

    with _broadcast_lock:
        clientes_validos = []
        for c in _broadcast_clientes:
            try:
                c.send(pacote)
                clientes_validos.append(c)
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                try: c.close()
                except Exception: pass
        if len(clientes_validos) != len(_broadcast_clientes):
            log("AVISO", f"Cliente(s) do broadcaster desligaram-se "
                         f"({len(_broadcast_clientes) - len(clientes_validos)})")
        _broadcast_clientes[:] = clientes_validos


# ─────────────────────────────────────────────
#  CLIENTE DO RETIFICADOR (puxa JSONs)
# ─────────────────────────────────────────────
def _enfileirar_pacote_recente(fila_jsons: Queue, pacote: dict):
    if pacote.get("tipo") == "aruco":
        with fila_jsons.mutex:
            antigos = len(fila_jsons.queue)
            fila_jsons.queue = deque(
                p for p in fila_jsons.queue
                if not (isinstance(p, dict) and p.get("tipo") == "aruco")
            )
            removidos = antigos - len(fila_jsons.queue)
            fila_jsons.unfinished_tasks = max(
                0, fila_jsons.unfinished_tasks - removidos
            )
            fila_jsons.not_full.notify_all()

    try:
        fila_jsons.put_nowait(pacote)
    except Full:
        try:
            fila_jsons.get_nowait()
            fila_jsons.task_done()
        except Empty:
            pass
        try:
            fila_jsons.put_nowait(pacote)
        except Full:
            pass


def loop_cliente_retificador(estado: EstadoGrafo, fila_jsons: Queue, parar: threading.Event):
    """
    Loop persistente: liga ao retificador (porta 6020) e pede o próximo
    JSON disponível. O retificador bloqueia até haver um. Quando recebe,
    mete na fila local para o loop principal processar.
    """
    log("DEBUG", f"Cliente do retificador a tentar ligar à porta {PORTA_RET_GRAFO}...")

    backoff = 0.5
    while not parar.is_set():
        try:
            with Client(("localhost", PORTA_RET_GRAFO), authkey=AUTHKEY_GRAFO) as conn:
                log("HUMANO", "GraphProcessor ligado ao pipeline.")
                backoff = 0.5
                while not parar.is_set():
                    try:
                        conn.send({"acao": "pedir_proximo"})
                        # bloqueia até chegar resposta
                        if conn.poll(timeout=TIMEOUT_RET):
                            pacote = conn.recv()
                            if pacote and isinstance(pacote, dict):
                                _enfileirar_pacote_recente(fila_jsons, pacote)
                        else:
                            # nada novo; pequena pausa e tenta de novo
                            time.sleep(0.1)
                    except (EOFError, ConnectionResetError):
                        log("AVISO", "Ligação ao retificador caiu. Reabrindo...")
                        break
        except (ConnectionRefusedError, OSError):
            if not parar.is_set():
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 5.0)
        except Exception as e:
            log("ERRO", f"Erro no cliente retificador: {e}")
            time.sleep(1.0)

    log("DEBUG", "Cliente do retificador terminado.")


# ─────────────────────────────────────────────
#  VISUALIZAÇÃO AO VIVO (matplotlib)
# ─────────────────────────────────────────────
class VisualizadorAoVivo:
    def __init__(self, estado: EstadoGrafo):
        self.estado = estado
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7), num="GraphProcessor — Estado")
        self.fig.canvas.manager.set_window_title("GraphProcessor — Estado ao Vivo")
        self._configurar_eixos()

    def _configurar_eixos(self):
        bounds = self.estado.poligono.bounds  # (xmin, ymin, xmax, ymax)
        margem = 0.15 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        self.ax.set_xlim(bounds[0] - margem, bounds[2] + margem)
        self.ax.set_ylim(bounds[3] + margem, bounds[1] - margem)  # y invertido (top-down)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("#0f0f12")
        self.ax.grid(True, alpha=0.15)
        self.fig.patch.set_facecolor("#0a0a0d")
        for spine in self.ax.spines.values():
            spine.set_color("#444")
        self.ax.tick_params(colors="#aaa")
        self.ax.set_xlabel("x (m)", color="#aaa")
        self.ax.set_ylabel("y (m)", color="#aaa")

    def atualizar(self):
        with self.estado.lock:
            agora = time.time()
            faixas = list(self.estado.faixas)
            bolas  = [(bc.x, bc.y, bc.faixa_id, bc.estavel(agora))
                      for bc in self.estado.bolas]
            robo   = dict(self.estado.robo)
            poligono = self.estado.poligono
            disparo  = self.estado.disparo_ativo
            fase     = self.estado.fase_varrimento
            t_disp   = self.estado.t_disparo_iniciado
            total    = self.estado.total_peso()

        self.ax.clear()
        self._configurar_eixos()

        # Polígono da quadra
        x_pol, y_pol = poligono.exterior.xy
        self.ax.fill(x_pol, y_pol, color="#1a3a4a", alpha=0.35, zorder=1)
        self.ax.plot(x_pol, y_pol, color="#22cccc", lw=1.5, zorder=2)

        # Faixas
        for f in faixas:
            if f.poligono.is_empty:
                continue
            cor_borda = "#ff5050" if (disparo is not None and f.id == disparo.id) else "#555"
            cor_pre   = "#aa2020" if (disparo is not None and f.id == disparo.id) else None
            if f.poligono.geom_type == "Polygon":
                xs, ys = f.poligono.exterior.xy
                if cor_pre:
                    self.ax.fill(xs, ys, color=cor_pre, alpha=0.35, zorder=2)
                self.ax.plot(xs, ys, color=cor_borda, lw=1.0, zorder=3)
            # rótulo
            cx = (f.x_inicial + f.x_final) / 2
            self.ax.text(cx, f.y_centro, f"{f.label}\npeso={f.peso}",
                         color="#ffcf66" if f.peso >= K_MIN else "#888",
                         ha="center", va="center", fontsize=8, zorder=4)
            # marcador da posição inicial
            self.ax.plot(f.x_inicial, f.y_centro, marker="x",
                         color="#66ff99", markersize=6, zorder=5)

        # Bolas — estáveis (cor) vs instáveis (cinzento, mais pequeno)
        for x, y, fid, estavel in bolas:
            if not estavel:
                self.ax.plot(x, y, marker="o",
                             color="#666666",
                             markersize=4, markeredgecolor="#222",
                             markeredgewidth=0.3, alpha=0.6, zorder=6)
            else:
                in_disparo = disparo is not None and fid == disparo.id
                self.ax.plot(x, y, marker="o",
                             color="#ffe060" if in_disparo else "#ffaa30",
                             markersize=6, markeredgecolor="black",
                             markeredgewidth=0.4, zorder=6)

        # Robô (cor depende do estado de alinhamento durante aguarda_inicio)
        robo_posicao = None
        if robo.get("frontal") and robo.get("traseiro"):
            rfx = robo["frontal"]["x"];   rfy = robo["frontal"]["y"]
            rtx = robo["traseiro"]["x"];  rty = robo["traseiro"]["y"]
            robo_posicao = (rfx, rfy)

            # Decidir cor: durante aguarda_inicio mostramos verde se já está
            # alinhado e perto do ponto inicial, vermelho se ainda não.
            cor_robo = "#ff8030"   # default
            if disparo is not None and fase == "aguarda_inicio":
                dist = ((rfx-disparo.pos_inicial[0])**2 +
                        (rfy-disparo.pos_inicial[1])**2) ** 0.5
                v_alvo = (disparo.pos_final[0]-disparo.pos_inicial[0],
                          disparo.pos_final[1]-disparo.pos_inicial[1])
                v_robo = (rfx - rtx, rfy - rty)
                ang = _alinhamento_graus(v_alvo, v_robo)
                if (dist <= TOLERANCIA_DISTANCIA_AO_PONTO and
                    ang is not None and ang <= TOLERANCIA_ANGULO_GRAUS):
                    cor_robo = "#66ff66"
                elif dist <= TOLERANCIA_DISTANCIA_AO_PONTO:
                    cor_robo = "#ffaa30"   # perto mas desalinhado

            self.ax.annotate("", xy=(rfx, rfy), xytext=(rtx, rty),
                             arrowprops=dict(arrowstyle="->", color=cor_robo, lw=2),
                             zorder=7)
            self.ax.plot([rtx, rfx], [rty, rfy], color=cor_robo, lw=2, zorder=7)

        # Trajetória de 3 pontos (se disparo ativo — MODO FAIXAS)
        if disparo is not None:
            # Segmento 1: ArUco frontal → inicial (aproximação)
            if robo_posicao is not None:
                self.ax.annotate("",
                                 xy=disparo.pos_inicial, xytext=robo_posicao,
                                 arrowprops=dict(arrowstyle="->",
                                                 color="#33ccff", lw=2,
                                                 linestyle="--"),
                                 zorder=8)
            # Segmento 2: inicial → final (varrimento)
            self.ax.annotate("",
                             xy=disparo.pos_final, xytext=disparo.pos_inicial,
                             arrowprops=dict(arrowstyle="->", color="#66ff66", lw=2.5),
                             zorder=8)
            # Marcadores dos pontos
            self.ax.plot(*disparo.pos_inicial, marker="o",
                         color="#66ff66", markersize=10,
                         markeredgecolor="white", markeredgewidth=1, zorder=9)
            self.ax.plot(*disparo.pos_final, marker="s",
                         color="#66ff66", markersize=10,
                         markeredgecolor="white", markeredgewidth=1, zorder=9)

        # ── Modo GLOBAL: desenhar rota TSP ──────────────────
        with self.estado.lock:
            modo_op      = MODO_OPERACAO
            fase_g       = self.estado.fase_global
            planner      = self.estado.planner_global
            wp_idx       = planner.idx_atual
            waypoints    = list(planner._waypoints)   # cópia rápida

        if modo_op == "GLOBAL" and waypoints:
            # Desenhar todos os segmentos da rota (cinzento fino)
            pts = waypoints
            if robo_posicao is not None and wp_idx < len(pts):
                # Seta ArUco frontal → waypoint atual (destaque)
                self.ax.annotate("",
                                 xy=pts[wp_idx], xytext=robo_posicao,
                                 arrowprops=dict(arrowstyle="->",
                                                 color="#33ccff", lw=2,
                                                 linestyle="--"),
                                 zorder=8)
            for i in range(len(pts) - 1):
                cor = "#888888" if i < wp_idx else "#66ff66"
                lw  = 1.0      if i < wp_idx else 2.0
                self.ax.annotate("",
                                 xy=pts[i + 1], xytext=pts[i],
                                 arrowprops=dict(arrowstyle="->", color=cor, lw=lw),
                                 zorder=7)
            for i, (wx, wy) in enumerate(pts):
                if i < wp_idx:
                    self.ax.plot(wx, wy, marker="o", color="#444444",
                                 markersize=6, zorder=8)
                elif i == wp_idx:
                    self.ax.plot(wx, wy, marker="*", color="#ffe060",
                                 markersize=14, markeredgecolor="white",
                                 markeredgewidth=0.8, zorder=9)
                else:
                    self.ax.plot(wx, wy, marker="o", color="#66ff66",
                                 markersize=7, markeredgecolor="black",
                                 markeredgewidth=0.4, zorder=8)
                self.ax.text(wx + 0.04, wy - 0.04, str(i + 1),
                             color="#cccccc", fontsize=6, zorder=9)

        # Cabeçalho
        n_bolas = len(bolas)
        n_estaveis = sum(1 for *_, est in bolas if est)
        if MODO_OPERACAO == "GLOBAL":
            with self.estado.lock:
                fg = self.estado.fase_global
                wp_cur  = self.estado.planner_global.idx_atual
                wp_tot  = self.estado.planner_global.n_waypoints
                elapsed = time.time() - self.estado.t_global_iniciado if fg == "executar" else 0
            if fg == "aguardar":
                modo = f"GLOBAL — aguardando ≥{K_MIN_GLOBAL} bolas ({n_estaveis} estáveis)"
            elif fg == "calcular":
                modo = "GLOBAL — a calcular rota TSP..."
            elif fg == "aguardar_confirmacao":
                modo = "GLOBAL — AGUARDA CONFIRMAÇÃO NO TERMINAL"
            else:
                modo = f"GLOBAL ★ EXECUTAR — waypoint {wp_cur}/{wp_tot} ({elapsed:.0f}s)"
        else:
            if disparo is not None:
                elapsed = time.time() - t_disp
                if fase == "aguarda_inicio":
                    modo = f"DISPARO ★ Faixa {disparo.label} — APROXIMAÇÃO ({elapsed:.0f}s)"
                elif fase == "em_varrimento":
                    modo = f"DISPARO ★ Faixa {disparo.label} — VARRIMENTO ({elapsed:.0f}s)"
                else:
                    modo = f"DISPARO ★ Faixa {disparo.label}"
            elif fase == "aguardar_confirmacao":
                modo = "FAIXAS — AGUARDA CONFIRMAÇÃO NO TERMINAL"
            else:
                modo = "Acumulando..."
        self.ax.set_title(
            f"{modo}  |  bolas={n_estaveis} estáveis ({n_bolas} total)  |  "
            + (f"k_min_global={K_MIN_GLOBAL}" if MODO_OPERACAO == "GLOBAL"
               else f"peso={total}  |  threshold={int(THRESHOLD_PCT*100)}% & ≥{K_MIN}"),
            color="#ddd", fontsize=10
        )

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass


# ─────────────────────────────────────────────
#  PROCESSAMENTO DE UM PACOTE
# ─────────────────────────────────────────────
def processar_pacote(estado: EstadoGrafo, pacote: dict):
    """Aplica um pacote vindo do retificador ao estado interno."""
    indice  = pacote.get("indice", -1)
    tipo    = pacote.get("tipo", "vision")
    bolas   = pacote.get("trajetoria", []) or []
    robo    = pacote.get("robo") or {}

    with estado.lock:
        tem_robo = (
            robo.get("frontal") is not None
            or robo.get("traseiro") is not None
            or robo.get("orientacao_graus") is not None
        )
        if tipo == "aruco" or tem_robo:
            estado.robo = {
                "frontal":          robo.get("frontal"),
                "traseiro":         robo.get("traseiro"),
                "orientacao_graus": robo.get("orientacao_graus"),
                "qualidade_localizacao": dict(
                    robo.get("qualidade_localizacao") or {}
                ),
            }
        estado.ultimo_indice_processado = indice
        estado.ultima_latencia_retificador_ms = pacote.get("latencia_ms")

    if MODO_OPERACAO == "GLOBAL":
        with estado.lock:
            fase_g = estado.fase_global

        if fase_g == "executar":
            # YOLO está pausado — só ArUco chega; não acumulamos bolas.
            # Apenas avançamos a máquina de waypoints.
            verificar_progresso_global(estado)
        elif fase_g in ("aguardar", "calcular"):
            # YOLO ativo apenas enquanto aguardamos ou calculamos uma rota.
            deduplicar_e_atualizar(estado, bolas)
            recalcular_pesos(estado)
            if fase_g == "calcular":
                # Tenta computar a rota (pode falhar se ArUco indisponível
                # neste frame — fica em "calcular" até o próximo frame)
                calcular_rota_global(estado)
    else:
        # ── Modo FAIXAS (comportamento original) ──
        with estado.lock:
            em_disparo = estado.disparo_ativo is not None

        if em_disparo:
            verificar_progresso_varrimento(estado)
        elif not estado.aguarda_confirmacao_novo_conjunto:
            deduplicar_e_atualizar(estado, bolas)
            recalcular_pesos(estado)

    broadcast_estado(estado)


# ─────────────────────────────────────────────
#  LOOP PRINCIPAL
# ─────────────────────────────────────────────
def main():
    global LARGURA_ROBO_M, RAIO_DEDUP_M, THRESHOLD_PCT, K_MIN, N_OBS_MIN_ESTAVEL
    global TEMPO_MIN_ESTAVEL_S, VELOCIDADE_MAX_PARADA_M_S, RAIO_ESTACIONARIA_BOLA_M
    global TEMPO_EXPIRAR_BOLA_S
    global TOLERANCIA_DISTANCIA_AO_PONTO, TOLERANCIA_ANGULO_GRAUS
    global TIMEOUT_VARRIMENTO_S, RAIO_RECOLHA_BOLA_M, MODO_OPERACAO, K_MIN_GLOBAL, _CFG

    parser = argparse.ArgumentParser(description="GraphProcessor UFSC/FEUP")
    parser.add_argument("--no-vis", action="store_true",
                        help="Desliga a visualização ao vivo (modo headless)")
    args = parser.parse_args()

    # ── Carregar parâmetros configurados pelo utilizador ──
    _CFG = _params.carregar()
    LARGURA_ROBO_M                 = float(_CFG.get(
        "largura_robo_cm", LARGURA_ROBO_M * 100
    )) / 100.0
    RAIO_DEDUP_M                   = float(_CFG.get("raio_dedup_cm", RAIO_DEDUP_M*100)) / 100.0
    THRESHOLD_PCT                  = float(_CFG.get("threshold_pct", THRESHOLD_PCT*100)) / 100.0
    K_MIN                          = int(_CFG.get("k_min", K_MIN))
    N_OBS_MIN_ESTAVEL              = int(_CFG.get("n_obs_min_estavel", N_OBS_MIN_ESTAVEL))
    TEMPO_MIN_ESTAVEL_S            = float(_CFG.get("tempo_min_estavel_s", TEMPO_MIN_ESTAVEL_S))
    VELOCIDADE_MAX_PARADA_M_S      = float(_CFG.get(
        "velocidade_max_bola_parada_cm_s",
        VELOCIDADE_MAX_PARADA_M_S * 100,
    )) / 100.0
    RAIO_ESTACIONARIA_BOLA_M       = float(_CFG.get(
        "raio_confirmar_bola_parada_cm",
        RAIO_ESTACIONARIA_BOLA_M * 100,
    )) / 100.0
    TEMPO_EXPIRAR_BOLA_S           = float(_CFG.get("tempo_expirar_bola_s", TEMPO_EXPIRAR_BOLA_S))
    TOLERANCIA_DISTANCIA_AO_PONTO  = float(_CFG.get("tolerancia_distancia_cm",
                                                    TOLERANCIA_DISTANCIA_AO_PONTO*100)) / 100.0
    TOLERANCIA_ANGULO_GRAUS        = float(_CFG.get("tolerancia_angulo_graus",
                                                    TOLERANCIA_ANGULO_GRAUS))
    TIMEOUT_VARRIMENTO_S           = float(_CFG.get("timeout_varrimento_s",
                                                    TIMEOUT_VARRIMENTO_S))
    RAIO_RECOLHA_BOLA_M            = float(_CFG.get("raio_recolha_bola_cm",
                                                    RAIO_RECOLHA_BOLA_M*100)) / 100.0
    MODO_OPERACAO                  = str(_CFG.get("modo_operacao", MODO_OPERACAO)).upper()
    K_MIN_GLOBAL                   = int(_CFG.get("k_min_global", K_MIN_GLOBAL))

    iniciar_health_server()

    log("DEBUG", f"parâmetros carregados de {_params.FICH_PARAMS}")
    log("EVENTO", f"Modo de operação: {MODO_OPERACAO}")
    if MODO_OPERACAO == "GLOBAL":
        log("HUMANO", f"Modo GLOBAL ativo — disparo com ≥{K_MIN_GLOBAL} bolas totais.")
    else:
        log("HUMANO", f"Modo FAIXAS ativo — threshold={int(THRESHOLD_PCT*100)}% & ≥{K_MIN}.")

    log("HUMANO", "A carregar calibração e construir geometria da quadra...")
    pontos, calib = carregar_pontos_calibracao()
    log("DEBUG", f"  ppm={calib['ppm']:.1f} | erro={calib.get('erro_medio_m','?')}m | "
              f"{len(pontos)} pontos de calibração")

    poligono = construir_poligono_quadra(pontos)
    faixas   = construir_faixas(poligono, LARGURA_ROBO_M)
    estado   = EstadoGrafo(faixas, poligono, calib)

    log("DEBUG", f"Configuração: faixas={len(faixas)} | "
                f"largura robô={LARGURA_ROBO_M*100:.1f}cm | "
                f"raio={RAIO_DEDUP_M*100:.1f}cm | "
                f"threshold={int(THRESHOLD_PCT*100)}% | K_min={K_MIN} | "
                f"n_obs_min={N_OBS_MIN_ESTAVEL} | tempo_estável={TEMPO_MIN_ESTAVEL_S:.1f}s | "
                f"v_parada≤{VELOCIDADE_MAX_PARADA_M_S*100:.1f}cm/s")
    log("DEBUG", f"  tolerância distância={TOLERANCIA_DISTANCIA_AO_PONTO*100:.0f}cm | "
                f"alinhamento={TOLERANCIA_ANGULO_GRAUS:.0f}° | "
                f"timeout varrimento={TIMEOUT_VARRIMENTO_S:.0f}s | "
                f"raio recolha bola={RAIO_RECOLHA_BOLA_M*100:.0f}cm")

    # Threads
    parar = threading.Event()
    fila_jsons: Queue = Queue(maxsize=8)

    t_cli = threading.Thread(
        target=loop_cliente_retificador,
        args=(estado, fila_jsons, parar),
        daemon=True,
    )
    t_cli.start()

    t_bc = threading.Thread(
        target=loop_servidor_broadcast,
        args=(parar,),
        daemon=True,
    )
    t_bc.start()

    # Visualização (na thread principal)
    vis = VisualizadorAoVivo(estado) if not args.no_vis else None

    log("HUMANO", "GraphProcessor pronto. A acumular bolas...")
    log("DEBUG",  "loop principal iniciado")
    t_ultima_vis = 0.0

    try:
        while True:
            # 1. Processar pacotes pendentes
            try:
                pacote = fila_jsons.get(timeout=0.01)
                processar_pacote(estado, pacote)
                for _ in range(8):
                    try:
                        pacote = fila_jsons.get_nowait()
                    except Empty:
                        break
                    processar_pacote(estado, pacote)

                # 2. Verificar disparo (só quando não há execução ativa)
                if MODO_OPERACAO == "GLOBAL":
                    # processar_pacote já trata "calcular" e "executar";
                    # aqui apenas verificamos se o threshold foi atingido
                    # enquanto estamos em "aguardar".
                    verificar_disparo_global(estado)

                else:
                    # ── Modo FAIXAS — lógica original ──
                    with estado.lock:
                        em_disparo = estado.disparo_ativo is not None
                        aguarda_confirmacao = estado.aguarda_confirmacao_novo_conjunto
                    if not em_disparo and not aguarda_confirmacao:
                        faixa = verificar_disparo(estado)
                        if faixa is not None:
                            with estado.lock:
                                faixa.em_execucao = True
                                estado.disparo_ativo = faixa
                                estado.t_disparo_iniciado = time.time()
                                estado._avisou_alinhamento = False
                                pos_robo = _posicao_robo(estado.robo)
                                v_robo = _vetor_robo(estado.robo)
                                ja_la = False
                                if pos_robo is not None and v_robo is not None:
                                    d = _distancia(pos_robo, faixa.pos_inicial)
                                    v_alvo = (faixa.pos_final[0] - faixa.pos_inicial[0],
                                              faixa.pos_final[1] - faixa.pos_inicial[1])
                                    ang = _alinhamento_graus(v_alvo, v_robo)
                                    ja_la = (d <= TOLERANCIA_DISTANCIA_AO_PONTO
                                             and ang is not None
                                             and ang <= TOLERANCIA_ANGULO_GRAUS)
                                estado.fase_varrimento = "em_varrimento" if ja_la else "aguarda_inicio"
                            log("EVENTO", f"★ DISPARO! Faixa {faixa.label} "
                                           f"(peso={faixa.peso}, total={estado.total_peso()}) "
                                           f"— fase: {estado.fase_varrimento}")
                            idx = gerar_artefactos_disparo(estado, faixa)
                            log("HUMANO", f"Aguardando que o robô percorra a faixa "
                                        f"(tolerância={TOLERANCIA_DISTANCIA_AO_PONTO*100:.0f}cm, "
                                        f"alinhamento={TOLERANCIA_ANGULO_GRAUS:.0f}°)...")

            except Empty:
                pass

            # 3. Atualizar visualização periodicamente
            now = time.time()
            if vis is not None and (now - t_ultima_vis) >= INTERVALO_VIS:
                vis.atualizar()
                t_ultima_vis = now

            # 4. Sair se a janela matplotlib for fechada
            if vis is not None and not plt.fignum_exists(vis.fig.number):
                log("HUMANO", "Janela fechada pelo utilizador.")
                break

    except KeyboardInterrupt:
        log("DEBUG", "Ctrl+C detetado.")

    finally:
        log("HUMANO", "A encerrar GraphProcessor...")
        parar.set()
        time.sleep(0.5)
        log("HUMANO", "GraphProcessor encerrado.")


if __name__ == "__main__":
    main()



