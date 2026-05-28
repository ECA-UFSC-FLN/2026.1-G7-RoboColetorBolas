"""
RobotController.py - Supervisor do Robo (UFSC/FEUP)
===================================================
Liga-se ao broadcaster do GraphProcessor (porta 6021), recebe a pose do
robo por ArUco e o proximo alvo da trajetoria, e conversa por UDP com o
ESP32. O controlo fino deixa de ser calculado aqui: o ESP32 usa os
encoders e a IMU para orientar e deslocar o robo localmente.

Protocolo servidor -> ESP32 (JSON UDP):
  orient_goal       pose atual, alvo, heading desejado
  move_permission   autorizacao para seguir ate ao alvo
  orientation_correction / stop_correct / arrived_ok / arrived_bad / stop

Protocolo ESP32 -> servidor (JSON UDP):
  {"event":"orientation_done", "segment_id":"..."}
  {"event":"arrived",          "segment_id":"..."}

O servidor confirma cada evento com a visao. Se a orientacao falhar, envia
correcao ate MAX_TENTATIVAS_ORIENTACAO. Durante o movimento, leituras
consecutivas fora da tolerancia fazem o servidor mandar parar e corrigir.
"""

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from multiprocessing.connection import Client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import _CONFIG.system_parameters as _params
from _COMMON.logging_utils import log as _log

PORTA_HEALTH = 6014
PORTA_BROADCAST = 6021
AUTHKEY_BROADCAST = b"controlador_ufsc"

IP_ROBO = "IP_DO_ROBO"
PORTA_ROBO_UDP = 5005
PORTA_FEEDBACK_UDP = 5006

TOLERANCIA_DISTANCIA_M = 0.20
TOLERANCIA_ANGULO_GRAUS = 15.0
MAX_TENTATIVAS_ORIENTACAO = 5
LEITURAS_DESVIO_CONSECUTIVAS = 3
DESVIO_MOVIMENTO_ANGULO_GRAUS = 25.0
DESVIO_MOVIMENTO_DISTANCIA_M = 0.25
REENVIAR_META_S = 0.75

MOD = "CONTROLADOR"


def log(nivel: str, msg: str):
    _log(MOD, nivel, msg)


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


def _ip_valido(ip: str) -> bool:
    try:
        socket.inet_aton(ip)
        return True
    except (OSError, TypeError):
        return False


def _sinal_angulo_graus(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _pose_robo(robo: dict | None) -> dict | None:
    if not robo or not robo.get("frontal") or not robo.get("traseiro"):
        return None
    f = robo["frontal"]
    t = robo["traseiro"]
    cx = (float(f["x"]) + float(t["x"])) / 2.0
    cy = (float(f["y"]) + float(t["y"])) / 2.0
    heading = math.degrees(math.atan2(float(f["y"]) - float(t["y"]),
                                      float(f["x"]) - float(t["x"])))
    return {
        "x": cx,
        "y": cy,
        "heading_deg": _sinal_angulo_graus(heading),
        "frontal": {"x": float(f["x"]), "y": float(f["y"])},
        "traseiro": {"x": float(t["x"]), "y": float(t["y"])},
    }


def _metricas_pose(robo: dict | None, alvo: dict | None) -> dict | None:
    pose = _pose_robo(robo)
    if pose is None or alvo is None:
        return None
    tx = float(alvo["x"])
    ty = float(alvo["y"])
    dx = tx - pose["x"]
    dy = ty - pose["y"]
    distancia = math.hypot(dx, dy)
    desejado = math.degrees(math.atan2(dy, dx))
    erro = _sinal_angulo_graus(desejado - pose["heading_deg"])
    return {
        "pose": pose,
        "target": {"x": tx, "y": ty},
        "distance_m": distancia,
        "desired_heading_deg": _sinal_angulo_graus(desejado),
        "heading_error_deg": erro,
        "aligned": abs(erro) <= TOLERANCIA_ANGULO_GRAUS,
        "at_target": distancia <= TOLERANCIA_DISTANCIA_M,
    }


def _segment_id(estado: dict) -> str | None:
    alvo = estado.get("alvo_destino")
    if not alvo:
        return None
    modo = estado.get("modo_operacao", "?")
    fase = estado.get("fase") or "sem_fase"
    faixa = estado.get("faixa_label") or "-"
    wp = estado.get("waypoint_idx", "-")
    x = round(float(alvo["x"]), 3)
    y = round(float(alvo["y"]), 3)
    return f"{modo}:{fase}:{faixa}:{wp}:{x}:{y}"


def _pacote_base(tipo: str, segment_id: str | None, estado: dict, metricas: dict | None,
                 seq: int) -> dict:
    pacote = {
        "type": tipo,
        "seq": seq,
        "segment_id": segment_id,
        "ts": round(time.time(), 3),
        "vision_index": estado.get("indice_visao"),
        "mode": estado.get("modo_operacao"),
        "phase": estado.get("fase"),
        "rectifier_latency_ms": estado.get("latencia_retificador_ms"),
    }
    if metricas is not None:
        pacote.update({
            "robot_pose": metricas["pose"],
            "target": metricas["target"],
            "distance_m": round(metricas["distance_m"], 4),
            "desired_heading_deg": round(metricas["desired_heading_deg"], 2),
            "heading_error_deg": round(metricas["heading_error_deg"], 2),
            "tolerance_distance_m": TOLERANCIA_DISTANCIA_M,
            "tolerance_heading_deg": TOLERANCIA_ANGULO_GRAUS,
        })
    return pacote


def _enviar(sock: socket.socket, ip: str, porta: int, pacote: dict,
            modo_simulado: bool) -> bool:
    if modo_simulado:
        return True
    try:
        sock.sendto(json.dumps(pacote, ensure_ascii=False).encode("utf-8"), (ip, porta))
        return True
    except Exception as e:
        log("AVISO", f"Falha UDP para ESP32: {e}")
        return False


def _receber_eventos(sock: socket.socket) -> list[dict]:
    eventos = []
    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except BlockingIOError:
            break
        except Exception as e:
            log("AVISO", f"Falha ao receber feedback UDP: {e}")
            break
        try:
            evento = json.loads(data.decode("utf-8"))
            evento["_addr"] = addr[0]
            eventos.append(evento)
        except (UnicodeDecodeError, json.JSONDecodeError):
            log("AVISO", f"Feedback UDP inválido de {addr[0]}")
    return eventos


def _log_metricas(prefixo: str, metricas: dict | None, estado: dict):
    if metricas is None:
        log("DEBUG", f"{prefixo}: sem pose ArUco completa.")
        return
    lat_ret = estado.get("latencia_retificador_ms")
    lat_graph = (time.time() - float(estado.get("timestamp", time.time()))) * 1000.0
    lat_txt = f"ret={lat_ret}ms graph->ctrl={lat_graph:.0f}ms"
    log("DEBUG",
        f"{prefixo}: d={metricas['distance_m']*100:.1f}cm "
        f"erro={metricas['heading_error_deg']:+.1f}° {lat_txt}")


def main():
    global IP_ROBO, PORTA_ROBO_UDP, PORTA_FEEDBACK_UDP
    global TOLERANCIA_DISTANCIA_M, TOLERANCIA_ANGULO_GRAUS
    global MAX_TENTATIVAS_ORIENTACAO, LEITURAS_DESVIO_CONSECUTIVAS
    global DESVIO_MOVIMENTO_ANGULO_GRAUS, DESVIO_MOVIMENTO_DISTANCIA_M

    parser = argparse.ArgumentParser(description="Supervisor UDP do Robo - UFSC/FEUP")
    parser.add_argument("--ip", default=None, help="IP do ESP32")
    parser.add_argument("--porta-udp", type=int, default=None,
                        help="Porta UDP onde o ESP32 recebe comandos")
    parser.add_argument("--porta-feedback", type=int, default=None,
                        help="Porta UDP local para eventos do ESP32")
    parser.add_argument("--max-orient", type=int, default=MAX_TENTATIVAS_ORIENTACAO,
                        help="Máximo de tentativas de correção de orientação")
    parser.add_argument("--leituras-desvio", type=int,
                        default=LEITURAS_DESVIO_CONSECUTIVAS,
                        help="Leituras consecutivas fora da tolerância antes de parar")
    args = parser.parse_args()

    cfg = _params.carregar()
    IP_ROBO = args.ip or cfg.get("ip_robo", IP_ROBO)
    PORTA_ROBO_UDP = args.porta_udp or int(cfg.get("porta_udp", PORTA_ROBO_UDP))
    PORTA_FEEDBACK_UDP = args.porta_feedback or int(
        cfg.get("porta_udp_feedback", PORTA_ROBO_UDP + 1)
    )
    TOLERANCIA_DISTANCIA_M = float(
        cfg.get("tolerancia_distancia_cm", TOLERANCIA_DISTANCIA_M * 100)
    ) / 100.0
    TOLERANCIA_ANGULO_GRAUS = float(
        cfg.get("tolerancia_angulo_graus", TOLERANCIA_ANGULO_GRAUS)
    )
    MAX_TENTATIVAS_ORIENTACAO = max(1, int(args.max_orient))
    LEITURAS_DESVIO_CONSECUTIVAS = max(1, int(args.leituras_desvio))
    DESVIO_MOVIMENTO_ANGULO_GRAUS = max(
        TOLERANCIA_ANGULO_GRAUS,
        float(cfg.get("supervisor_desvio_angulo_graus", DESVIO_MOVIMENTO_ANGULO_GRAUS)),
    )
    DESVIO_MOVIMENTO_DISTANCIA_M = float(
        cfg.get("supervisor_desvio_distancia_cm", DESVIO_MOVIMENTO_DISTANCIA_M * 100)
    ) / 100.0

    iniciar_health_server()
    modo_simulado = not _ip_valido(IP_ROBO)
    if modo_simulado:
        log("AVISO", f"IP do robô '{IP_ROBO}' não é válido - MODO SIMULADO ativo.")
    else:
        log("HUMANO", f"ESP32 comandos: {IP_ROBO}:{PORTA_ROBO_UDP}")
    log("HUMANO", f"ESP32 feedback: UDP local :{PORTA_FEEDBACK_UDP}")
    log("DEBUG",
        f"tolerâncias: alvo={TOLERANCIA_DISTANCIA_M*100:.0f}cm "
        f"orientação={TOLERANCIA_ANGULO_GRAUS:.0f}° "
        f"desvio_mov={DESVIO_MOVIMENTO_ANGULO_GRAUS:.0f}°/"
        f"{DESVIO_MOVIMENTO_DISTANCIA_M*100:.0f}cm")

    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_rx.bind(("", PORTA_FEEDBACK_UDP))
    sock_rx.setblocking(False)

    seq = 0
    current_segment = None
    phase = "idle"
    orient_attempts = 0
    bad_readings = 0
    last_distance = None
    last_send = 0.0
    stopped_sent = False
    rx_janela = 0
    udp_janela = 0
    event_janela = 0
    t_taxa = time.time()
    backoff = 0.5

    def send(tipo: str, segment_id: str | None, estado: dict, metricas: dict | None):
        nonlocal seq, udp_janela, last_send
        seq += 1
        pacote = _pacote_base(tipo, segment_id, estado, metricas, seq)
        ok = _enviar(sock_tx, IP_ROBO, PORTA_ROBO_UDP, pacote, modo_simulado)
        if ok:
            udp_janela += 1
            last_send = time.time()
        return ok

    log("HUMANO", "A ligar ao broadcaster do GraphProcessor...")
    while True:
        try:
            with Client(("localhost", PORTA_BROADCAST), authkey=AUTHKEY_BROADCAST) as conn:
                log("HUMANO", "Ligado ao broadcaster. Supervisão ativa.")
                backoff = 0.5

                while True:
                    estado = conn.recv()
                    rx_janela += 1
                    eventos = _receber_eventos(sock_rx)
                    event_janela += len(eventos)

                    alvo = estado.get("alvo_destino")
                    robo = estado.get("robo")
                    segment_id = _segment_id(estado)
                    metricas = _metricas_pose(robo, alvo)

                    if segment_id is None:
                        current_segment = None
                        phase = "idle"
                        orient_attempts = 0
                        bad_readings = 0
                        last_distance = None
                        if not stopped_sent:
                            send("stop", None, estado, metricas)
                            stopped_sent = True
                    else:
                        stopped_sent = False

                    if segment_id is not None and segment_id != current_segment:
                        current_segment = segment_id
                        phase = "orienting"
                        orient_attempts = 0
                        bad_readings = 0
                        last_distance = None
                        send("orient_goal", current_segment, estado, metricas)
                        _log_metricas("nova meta de orientação", metricas, estado)

                    for evento in eventos:
                        if segment_id is None:
                            continue
                        if evento.get("segment_id") not in (None, current_segment):
                            continue
                        nome = str(evento.get("event", "")).lower()
                        if nome == "orientation_done" and phase == "orienting":
                            if metricas and metricas["aligned"]:
                                phase = "moving"
                                bad_readings = 0
                                last_distance = metricas["distance_m"]
                                send("move_permission", current_segment, estado, metricas)
                                _log_metricas("orientação validada", metricas, estado)
                            else:
                                orient_attempts += 1
                                tipo = "orientation_correction"
                                if orient_attempts >= MAX_TENTATIVAS_ORIENTACAO:
                                    tipo = "stop"
                                    phase = "blocked"
                                send(tipo, current_segment, estado, metricas)
                                _log_metricas(
                                    f"orientação fora ({orient_attempts}/{MAX_TENTATIVAS_ORIENTACAO})",
                                    metricas,
                                    estado,
                                )
                        elif nome == "arrived":
                            if metricas and metricas["at_target"]:
                                send("arrived_ok", current_segment, estado, metricas)
                                _log_metricas("chegada validada", metricas, estado)
                                phase = "waiting_next"
                            else:
                                send("arrived_bad", current_segment, estado, metricas)
                                _log_metricas("chegada rejeitada", metricas, estado)

                    now = time.time()
                    if segment_id is not None and phase == "orienting" and now - last_send >= REENVIAR_META_S:
                        send("orient_goal", current_segment, estado, metricas)

                    if phase == "moving" and metricas is not None:
                        piorou = (
                            abs(metricas["heading_error_deg"]) > DESVIO_MOVIMENTO_ANGULO_GRAUS
                            or (
                                last_distance is not None
                                and metricas["distance_m"] > last_distance + DESVIO_MOVIMENTO_DISTANCIA_M
                            )
                        )
                        bad_readings = bad_readings + 1 if piorou else 0
                        last_distance = metricas["distance_m"]
                        if bad_readings >= LEITURAS_DESVIO_CONSECUTIVAS:
                            send("stop_correct", current_segment, estado, metricas)
                            phase = "orienting"
                            orient_attempts = 0
                            bad_readings = 0
                            _log_metricas("desvio persistente - parar/corrigir", metricas, estado)

                    agora_taxa = time.time()
                    if agora_taxa - t_taxa >= 1.0:
                        dt = agora_taxa - t_taxa
                        sufixo = " [SIM]" if modo_simulado else ""
                        log("DEBUG",
                            f"taxa controlo: rx={rx_janela/dt:.1f}Hz "
                            f"udp={udp_janela/dt:.1f}Hz "
                            f"fb={event_janela/dt:.1f}Hz{sufixo} "
                            f"fase={phase} indice_visao={estado.get('indice_visao')}")
                        rx_janela = udp_janela = event_janela = 0
                        t_taxa = agora_taxa

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

    sock_tx.close()
    sock_rx.close()
    log("HUMANO", "Supervisor encerrado.")


if __name__ == "__main__":
    main()
