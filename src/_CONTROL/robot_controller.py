"""
RobotController.py — Controlador do Robô (Pi 5) UFSC/FEUP
==========================================================
Liga-se ao broadcaster do GraphProcessor (porta 6021) e recebe em loop
o estado actual: posição/orientação do robô (via ArUco) e o ponto-alvo
de destino. Calcula comandos de velocidade (v_linear, omega) usando uma
estratégia em duas zonas:

    erro_angular > THR_ANG_GROSSO  →  só roda no sítio (v=0, omega=K_ang*erro)
    erro_angular ≤ THR_ANG_GROSSO  →  movimento misto:
                                       v     = V_MAX * cos(erro_ang) * sat(distância)
                                       omega = K_ang * erro_ang

Envia os comandos por UDP ao ESP32 do robô. Se o IP for um placeholder
inválido, entra em MODO SIMULADO: calcula os comandos e regista-os no
log periodicamente, mas não envia nada pela rede. Isto permite testar
todo o pipeline sem o robô físico estar presente.

Convenções:
- omega > 0  →  o robô vira no sentido em que o produto vetorial
                (v_robô × v_alvo) é positivo. O firmware do ESP32
                define como mapear isso aos motores esquerda/direita.
- v_linear   →  m/s positivo = andar para a frente.

Portas:
  6014  health-check
  6021  cliente do GraphProcessor (recebe estado)
  UDP   PORTA_ROBO_UDP → ESP32
"""

import os
import sys
import json
import math
import time
import socket
import argparse
import threading
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from multiprocessing.connection import Client

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO — REDE LOCAL
# ─────────────────────────────────────────────
PORTA_HEALTH         = 6014
PORTA_BROADCAST      = 6021
AUTHKEY_BROADCAST    = b"controlador_ufsc"

# ─────────────────────────────────────────────
#  PARÂMETROS DO ROBÔ E DO CONTROLADOR
# ─────────────────────────────────────────────
# Os valores reais são lidos de resultados/configuracao/parametros.json no
# arranque (em main()). As constantes abaixo são DEFAULTS de fallback,
# usados apenas se o ficheiro não for carregável por alguma razão.
import _CONFIG.system_parameters as _params
_CFG: dict = {}                  # preenchido em main()

# Defaults de fallback
IP_ROBO              = "IP_DO_ROBO"
PORTA_ROBO_UDP       = 5005
V_MAX                = 0.15
OMEGA_MAX            = 1.0
K_ANG                = 1.5
THR_ANG_GROSSO_GRAUS = 20.0
D_LIM                = 0.5
DIST_PARAGEM         = 0.05      # fixo — não configurável (segurança)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
from _COMMON.logging_utils import log as _log

MOD = "CONTROLADOR"

def log(nivel: str, msg: str):
    """Atalho local: encapsula bolas_log.log com o módulo fixo."""
    _log(MOD, nivel, msg)


# ─────────────────────────────────────────────
#  HEALTH-CHECK SERVER
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
#  DETECÇÃO DE MODO SIMULADO
# ─────────────────────────────────────────────
def _ip_valido(ip: str) -> bool:
    try:
        socket.inet_aton(ip)
        return True
    except (OSError, TypeError):
        return False


# ─────────────────────────────────────────────
#  CÁLCULO DOS COMANDOS DE VELOCIDADE
# ─────────────────────────────────────────────
def _saturar(x: float, lim: float) -> float:
    return max(-lim, min(lim, x))


def calcular_comandos(robo: dict | None,
                      alvo: dict | None) -> tuple[float, float, dict]:
    """
    Devolve (v_linear, omega, info_debug).

    Estratégia: quando o erro angular é grande, o robô apenas roda no sítio
    para se alinhar; quando é pequeno, anda em frente com velocidade
    modulada pelo cosseno do erro angular e pela distância ao alvo.

    info_debug contém grandezas internas para logging:
      {distancia, erro_ang_graus, modo}
    """
    info = {"distancia": None, "erro_ang_graus": None, "modo": "parado"}

    # Sem robô detectado ou sem alvo → parar
    if not robo or not robo.get("frontal") or not robo.get("traseiro") or not alvo:
        return 0.0, 0.0, info

    fx, fy = robo["frontal"]["x"], robo["frontal"]["y"]
    tx, ty = robo["traseiro"]["x"], robo["traseiro"]["y"]

    # Centro do robô
    cx, cy = (fx + tx) / 2.0, (fy + ty) / 2.0

    # Vetor para o alvo (centro → destino)
    dx, dy = alvo["x"] - cx, alvo["y"] - cy
    distancia = math.hypot(dx, dy)
    info["distancia"] = distancia

    if distancia < DIST_PARAGEM:
        info["modo"] = "no_alvo"
        return 0.0, 0.0, info

    # Vetor de orientação do robô (traseiro → frontal)
    rx, ry = fx - tx, fy - ty

    norma_r = math.hypot(rx, ry)
    if norma_r < 1e-6:
        # ArUco colapsado — improvável mas possível
        info["modo"] = "robo_invalido"
        return 0.0, 0.0, info

    # Erro angular SIGNED via atan2(cross, dot) — robusto a wrap-around
    cross = rx * dy - ry * dx
    dot   = rx * dx + ry * dy
    erro_ang_rad = math.atan2(cross, dot)   # ∈ [-π, π]
    erro_ang_graus = math.degrees(erro_ang_rad)
    info["erro_ang_graus"] = erro_ang_graus

    # Componente angular: proporcional ao erro, saturada
    omega = _saturar(K_ANG * erro_ang_rad, OMEGA_MAX)

    # Componente linear: depende da magnitude do erro angular
    thr_rad = math.radians(THR_ANG_GROSSO_GRAUS)
    if abs(erro_ang_rad) > thr_rad:
        # Erro grande → só roda
        v_linear = 0.0
        info["modo"] = "rotacao_pura"
    else:
        # Erro pequeno → anda em frente, modulado por cos(erro) e distância
        sat_dist = min(1.0, distancia / D_LIM)
        v_linear = V_MAX * math.cos(erro_ang_rad) * sat_dist
        info["modo"] = "movimento_misto"

    return v_linear, omega, info


# ─────────────────────────────────────────────
#  ENVIO UDP AO ESP32
# ─────────────────────────────────────────────
def enviar_udp(sock: socket.socket, ip: str, porta: int,
               v: float, w: float, seq: int) -> bool:
    """
    Envia um pacote JSON ao ESP32. UDP fire-and-forget, sem ACK.
    Devolve False só em erros locais (socket fechado, etc).
    """
    pacote = json.dumps({
        "v":   round(v, 4),
        "w":   round(w, 4),
        "seq": seq,
        "ts":  round(time.time(), 3),
    }).encode("utf-8")
    try:
        sock.sendto(pacote, (ip, porta))
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
#  LOOP PRINCIPAL
# ─────────────────────────────────────────────
def main():
    global IP_ROBO, PORTA_ROBO_UDP, V_MAX, OMEGA_MAX, K_ANG
    global THR_ANG_GROSSO_GRAUS, D_LIM, _CFG

    parser = argparse.ArgumentParser(description="Controlador do Robô — UFSC/FEUP")
    parser.add_argument("--ip", default=None,
                        help="IP do ESP32 (sobrepõe parametros.json)")
    parser.add_argument("--porta-udp", type=int, default=None,
                        help="Porta UDP do ESP32 (sobrepõe parametros.json)")
    parser.add_argument("--log-cada", type=int, default=5,
                        help="Imprime comandos a cada N pacotes (default 5)")
    args = parser.parse_args()

    # ── Carregar parâmetros configurados pelo utilizador ──
    _CFG = _params.carregar()
    IP_ROBO              = args.ip       or _CFG.get("ip_robo", IP_ROBO)
    PORTA_ROBO_UDP       = args.porta_udp or int(_CFG.get("porta_udp", PORTA_ROBO_UDP))
    V_MAX                = float(_CFG.get("v_max", V_MAX))
    OMEGA_MAX            = float(_CFG.get("omega_max", OMEGA_MAX))
    K_ANG                = float(_CFG.get("k_ang", K_ANG))
    THR_ANG_GROSSO_GRAUS = float(_CFG.get("thr_ang_grosso_graus", THR_ANG_GROSSO_GRAUS))
    D_LIM                = float(_CFG.get("d_lim", D_LIM))

    porta_udp = PORTA_ROBO_UDP

    iniciar_health_server()

    log("DEBUG", f"parâmetros carregados de {_params.FICH_PARAMS}")
    log("DEBUG", f"V_MAX={V_MAX} OMEGA_MAX={OMEGA_MAX} K_ANG={K_ANG} "
                 f"THR_ANG_GROSSO={THR_ANG_GROSSO_GRAUS}° D_LIM={D_LIM}m")

    modo_simulado = not _ip_valido(IP_ROBO)
    if modo_simulado:
        log("AVISO", f"IP do robô '{IP_ROBO}' não é válido — MODO SIMULADO ativo.")
        log("HUMANO", "Comandos serão calculados e registados, mas não enviados.")
    else:
        log("HUMANO",  f"IP do robô: {IP_ROBO}:{porta_udp}")
        log("HUMANO", "Modo: envio UDP ativo.")

    sock_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    backoff = 0.5
    n_pacotes = 0
    n_no_robo = 0

    log("HUMANO", f"A ligar ao broadcaster do GraphProcessor...")
    log("DEBUG",  f"localhost:{PORTA_BROADCAST}")

    while True:
        try:
            with Client(("localhost", PORTA_BROADCAST), authkey=AUTHKEY_BROADCAST) as conn:
                log("HUMANO", "Ligado ao broadcaster do GraphProcessor.")
                backoff = 0.5

                while True:
                    estado = conn.recv()
                    n_pacotes += 1

                    robo = estado.get("robo")
                    alvo = estado.get("alvo_destino")
                    fase = estado.get("fase")
                    faixa_label = estado.get("faixa_label")

                    v, w, info = calcular_comandos(robo, alvo)
                    seq += 1

                    if not robo or not robo.get("frontal") or not robo.get("traseiro"):
                        n_no_robo += 1

                    if not modo_simulado:
                        ok = enviar_udp(sock_udp, IP_ROBO, porta_udp, v, w, seq)
                        if not ok and n_pacotes % 50 == 1:
                            log("AVISO", "Falha a enviar UDP (sem rota?).")

                    if n_pacotes % max(1, args.log_cada) == 0:
                        sufixo = " [SIM]" if modo_simulado else ""
                        modo_op = estado.get("modo_operacao", "FAIXAS")
                        if alvo is None:
                            log("DEBUG", f"idle{sufixo}  v={v:+.3f}  w={w:+.3f}  "
                                         f"(sem disparo ativo)")
                        elif modo_op == "GLOBAL":
                            d_str   = (f"{info['distancia']*100:.1f}cm"
                                       if info["distancia"] is not None else "?")
                            ang_str = (f"{info['erro_ang_graus']:+.1f}°"
                                       if info["erro_ang_graus"] is not None else "?")
                            wp_idx  = estado.get("waypoint_idx", "?")
                            wp_tot  = estado.get("waypoints_total", "?")
                            log("DEBUG",
                                f"GLOBAL{sufixo} [{wp_idx}/{wp_tot}]  "
                                f"v={v:+.3f}m/s  w={w:+.3f}rad/s  "
                                f"d={d_str}  ang={ang_str}  modo={info['modo']}")
                        else:
                            d_str   = (f"{info['distancia']*100:.1f}cm"
                                       if info["distancia"] is not None else "?")
                            ang_str = (f"{info['erro_ang_graus']:+.1f}°"
                                       if info["erro_ang_graus"] is not None else "?")
                            log("DEBUG", f"{fase or '?'} → faixa {faixa_label or '?'}{sufixo}  "
                                         f"v={v:+.3f}m/s  w={w:+.3f}rad/s  "
                                         f"d={d_str}  ang={ang_str}  modo={info['modo']}")

        except (ConnectionRefusedError, OSError):
            log("AVISO", f"GraphProcessor não disponível. A retentar em {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)
        except (EOFError, ConnectionResetError):
            log("AVISO", "Ligação ao GraphProcessor caiu. A reabrir...")
            time.sleep(0.5)
        except KeyboardInterrupt:
            log("HUMANO", "Ctrl+C detetado.")
            break
        except Exception as e:
            log("ERRO", f"Erro inesperado: {e}")
            time.sleep(1.0)

    sock_udp.close()
    log("HUMANO", f"Controlador encerrado. Pacotes recebidos: {n_pacotes} "
                  f"(sem deteção do robô em {n_no_robo}).")


if __name__ == "__main__":
    main()





