"""Supervisor UDP do robo.

O ESP32 faz o controlo local com encoders/IMU. Este processo envia metas,
autoriza movimento e valida os eventos recebidos usando a visao.
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
MARGEM_LIBERACAO_ANGULO_GRAUS = 3.0
MAX_TENTATIVAS_ORIENTACAO = 5
LEITURAS_DESVIO_CONSECUTIVAS = 3
DESVIO_MOVIMENTO_ANGULO_GRAUS = 25.0
DESVIO_MOVIMENTO_DISTANCIA_M = 0.25
DESVIO_LATERAL_M = 0.08
RECUPERACAO_PERDA_VISAO_ATIVA = False
TIMEOUT_PERDA_VISAO_S = 0.6
DISTANCIA_RECUPERACAO_CM = 30.0
REENVIAR_META_S = 0.75
TIMEOUT_FEEDBACK_S = 6.0
MAX_REENVIOS_ORIENT_GOAL = 20
ORIENTATION_SETTLE_S = 0.45
MODO_SUPERVISAO_UDP = "PONTO_A_PONTO"
MODO_CORRECAO_ORIENTACAO_ESP32 = "PRIMEIRA_DEVAGAR"

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
    qualidade = robo.get("qualidade_localizacao") or {}
    if str(qualidade.get("fonte", "ARUCO")).upper() == "COR" and not qualidade.get(
        "valida_controle", False
    ):
        return None
    f = robo["frontal"]
    t = robo["traseiro"]
    px = float(f["x"])
    py = float(f["y"])
    heading = math.degrees(math.atan2(float(f["y"]) - float(t["y"]),
                                      float(f["x"]) - float(t["x"])))
    return {
        "x": px,
        "y": py,
        "heading_deg": _sinal_angulo_graus(heading),
        "frontal": {"x": float(f["x"]), "y": float(f["y"])},
        "traseiro": {"x": float(t["x"]), "y": float(t["y"])},
        "position_reference": "aruco_frontal",
        "localization_source": qualidade.get("fonte", "ARUCO"),
        "localization_confidence": qualidade.get("confianca"),
    }


def _metricas_pose(robo: dict | None, alvo: dict | None,
                   origem: dict | None = None) -> dict | None:
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
    desvio_lateral = None
    if origem is not None:
        ox = float(origem["x"])
        oy = float(origem["y"])
        vx = tx - ox
        vy = ty - oy
        norma = math.hypot(vx, vy)
        if norma > 1e-9:
            desvio_lateral = abs(vx * (pose["y"] - oy) - vy * (pose["x"] - ox)) / norma
    return {
        "pose": pose,
        "target": {"x": tx, "y": ty},
        "distance_m": distancia,
        "desired_heading_deg": _sinal_angulo_graus(desejado),
        "heading_error_deg": erro,
        "cross_track_error_m": desvio_lateral,
        "aligned": abs(erro) <= TOLERANCIA_ANGULO_GRAUS,
        "aligned_release": abs(erro) <= (
            TOLERANCIA_ANGULO_GRAUS + MARGEM_LIBERACAO_ANGULO_GRAUS
        ),
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
        "trajectory_id": estado.get("trajetoria_id"),
        "rectifier_latency_ms": estado.get("latencia_retificador_ms"),
        "orientation_correction_mode": MODO_CORRECAO_ORIENTACAO_ESP32,
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
            "target_heading_deg": round(metricas["desired_heading_deg"], 2),
        })
    return pacote


def _enviar(sock: socket.socket, ip: str, porta: int, pacote: dict,
            modo_simulado: bool) -> bool:
    if modo_simulado:
        return True
    try:
        dados = json.dumps(pacote, separators=(",", ":")).encode("utf-8")
        sock.sendto(dados, (ip, porta))
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
            log("AVISO", f"Feedback UDP invalido de {addr[0]}")
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
        f"erro={metricas['heading_error_deg']:+.1f}deg {lat_txt}")


def _log_trajetoria_completa(pose: dict, waypoints: list[dict], trajetoria_id: str):
    x_atual = float(pose["x"])
    y_atual = float(pose["y"])
    heading_atual = float(pose["heading_deg"])
    log("EVENTO",
        f"TRAJETORIA TX id={trajetoria_id} | pose inicial ArUco frontal="
        f"({x_atual:.3f},{y_atual:.3f})m | heading inicial={heading_atual:+.2f}deg | "
        f"pontos={len(waypoints)}")

    for indice, waypoint in enumerate(waypoints, 1):
        alvo_x = float(waypoint["x"])
        alvo_y = float(waypoint["y"])
        dx = alvo_x - x_atual
        dy = alvo_y - y_atual
        distancia = math.hypot(dx, dy)
        heading_desejado = _sinal_angulo_graus(math.degrees(math.atan2(dy, dx)))
        delta = _sinal_angulo_graus(heading_desejado - heading_atual)
        log("HUMANO",
            f"TRAJETORIA TX ponto {indice}/{len(waypoints)} | "
            f"de=({x_atual:.3f},{y_atual:.3f})m "
            f"para=({alvo_x:.3f},{alvo_y:.3f})m | "
            f"dist={distancia*100:.1f}cm | heading={heading_desejado:+.2f}deg | "
            f"delta_visao={delta:+.2f}deg")
        x_atual = alvo_x
        y_atual = alvo_y
        heading_atual = heading_desejado


def _resumo_execucao(fase: str, metricas: dict | None, estado: dict,
                     segundos_sem_feedback: float | None) -> str:
    alvo = estado.get("alvo_destino")
    if metricas is None:
        alvo_txt = "alvo=sem_alvo" if alvo is None else (
            f"alvo=({float(alvo['x']):.2f},{float(alvo['y']):.2f})m"
        )
        fb_txt = "fb=sem_eventos" if segundos_sem_feedback is None else f"fb_ha={segundos_sem_feedback:.1f}s"
        return f"fase={fase} pose=sem_ArUco {alvo_txt} {fb_txt}"

    pose = metricas["pose"]
    target = metricas["target"]
    fb_txt = "fb=sem_eventos" if segundos_sem_feedback is None else f"fb_ha={segundos_sem_feedback:.1f}s"
    return (
        f"fase={fase} "
        f"pose=({pose['x']:.2f},{pose['y']:.2f})m "
        f"theta={pose['heading_deg']:+.1f}deg "
        f"alvo=({target['x']:.2f},{target['y']:.2f})m "
        f"d={metricas['distance_m']*100:.1f}/{TOLERANCIA_DISTANCIA_M*100:.0f}cm "
        f"erro_ang={metricas['heading_error_deg']:+.1f}/{TOLERANCIA_ANGULO_GRAUS:.0f}deg "
        f"libera<={TOLERANCIA_ANGULO_GRAUS + MARGEM_LIBERACAO_ANGULO_GRAUS:.0f}deg "
        f"{fb_txt}"
    )


def main():
    global IP_ROBO, PORTA_ROBO_UDP, PORTA_FEEDBACK_UDP
    global TOLERANCIA_DISTANCIA_M, TOLERANCIA_ANGULO_GRAUS
    global MARGEM_LIBERACAO_ANGULO_GRAUS
    global MAX_TENTATIVAS_ORIENTACAO, LEITURAS_DESVIO_CONSECUTIVAS
    global DESVIO_MOVIMENTO_ANGULO_GRAUS, DESVIO_MOVIMENTO_DISTANCIA_M
    global DESVIO_LATERAL_M, RECUPERACAO_PERDA_VISAO_ATIVA
    global TIMEOUT_PERDA_VISAO_S, DISTANCIA_RECUPERACAO_CM
    global REENVIAR_META_S, TIMEOUT_FEEDBACK_S, MAX_REENVIOS_ORIENT_GOAL
    global ORIENTATION_SETTLE_S, MODO_SUPERVISAO_UDP
    global MODO_CORRECAO_ORIENTACAO_ESP32

    parser = argparse.ArgumentParser(description="Supervisor UDP do Robo - UFSC/FEUP")
    parser.add_argument("--ip", default=None, help="IP do ESP32")
    parser.add_argument("--porta-udp", type=int, default=None,
                        help="Porta UDP onde o ESP32 recebe comandos")
    parser.add_argument("--porta-feedback", type=int, default=None,
                        help="Porta UDP local para eventos do ESP32")
    parser.add_argument("--max-orient", type=int, default=MAX_TENTATIVAS_ORIENTACAO,
                        help="Maximo de tentativas de correcao de orientacao")
    parser.add_argument("--leituras-desvio", type=int, default=None,
                        help="Leituras consecutivas fora da tolerancia antes de parar")
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
    MARGEM_LIBERACAO_ANGULO_GRAUS = float(cfg.get(
        "supervisor_margem_liberacao_angulo_graus",
        MARGEM_LIBERACAO_ANGULO_GRAUS,
    ))
    MAX_TENTATIVAS_ORIENTACAO = max(1, int(args.max_orient))
    LEITURAS_DESVIO_CONSECUTIVAS = max(1, int(
        args.leituras_desvio
        if args.leituras_desvio is not None
        else cfg.get("supervisor_leituras_desvio_consecutivas", LEITURAS_DESVIO_CONSECUTIVAS)
    ))
    DESVIO_MOVIMENTO_ANGULO_GRAUS = max(
        TOLERANCIA_ANGULO_GRAUS,
        float(cfg.get("supervisor_desvio_angulo_graus", DESVIO_MOVIMENTO_ANGULO_GRAUS)),
    )
    DESVIO_MOVIMENTO_DISTANCIA_M = float(
        cfg.get("supervisor_desvio_distancia_cm", DESVIO_MOVIMENTO_DISTANCIA_M * 100)
    ) / 100.0
    DESVIO_LATERAL_M = float(cfg.get(
        "supervisor_desvio_lateral_cm", DESVIO_LATERAL_M * 100
    )) / 100.0
    RECUPERACAO_PERDA_VISAO_ATIVA = bool(int(cfg.get(
        "recuperacao_perda_visao_ativa", int(RECUPERACAO_PERDA_VISAO_ATIVA)
    )))
    TIMEOUT_PERDA_VISAO_S = float(cfg.get(
        "timeout_perda_visao_s", TIMEOUT_PERDA_VISAO_S
    ))
    DISTANCIA_RECUPERACAO_CM = float(cfg.get(
        "distancia_recuperacao_cm", DISTANCIA_RECUPERACAO_CM
    ))
    REENVIAR_META_S = float(cfg.get("supervisor_reenviar_meta_s", REENVIAR_META_S))
    TIMEOUT_FEEDBACK_S = float(cfg.get("supervisor_timeout_feedback_s", TIMEOUT_FEEDBACK_S))
    MAX_REENVIOS_ORIENT_GOAL = max(1, int(cfg.get(
        "supervisor_max_reenvios_orient_goal",
        MAX_REENVIOS_ORIENT_GOAL,
    )))
    ORIENTATION_SETTLE_S = float(cfg.get(
        "supervisor_assentamento_orientacao_s",
        ORIENTATION_SETTLE_S,
    ))
    MODO_SUPERVISAO_UDP = str(
        cfg.get("modo_supervisao_udp", MODO_SUPERVISAO_UDP)
    ).upper()
    MODO_CORRECAO_ORIENTACAO_ESP32 = str(cfg.get(
        "modo_correcao_orientacao_esp32",
        MODO_CORRECAO_ORIENTACAO_ESP32,
    )).upper()

    iniciar_health_server()
    modo_simulado = not _ip_valido(IP_ROBO)
    if modo_simulado:
        log("AVISO", f"IP do robo '{IP_ROBO}' nao e valido - MODO SIMULADO ativo.")
    else:
        log("HUMANO", f"ESP32 comandos: {IP_ROBO}:{PORTA_ROBO_UDP}")
    log("HUMANO", f"ESP32 feedback: UDP local :{PORTA_FEEDBACK_UDP}")
    log("HUMANO", f"Modo supervisor UDP: {MODO_SUPERVISAO_UDP}")
    log("HUMANO", f"Correcao angular ESP32: {MODO_CORRECAO_ORIENTACAO_ESP32}")
    log("DEBUG",
        f"tolerancias: alvo={TOLERANCIA_DISTANCIA_M*100:.0f}cm "
        f"orientacao={TOLERANCIA_ANGULO_GRAUS:.0f}deg "
        f"desvio_mov={DESVIO_MOVIMENTO_ANGULO_GRAUS:.0f}deg/"
        f"{DESVIO_MOVIMENTO_DISTANCIA_M*100:.0f}cm "
        f"lateral={DESVIO_LATERAL_M*100:.0f}cm/{LEITURAS_DESVIO_CONSECUTIVAS} frames "
        f"recuperacao_visao={'ON' if RECUPERACAO_PERDA_VISAO_ATIVA else 'OFF'} "
        f"reenviar_meta={REENVIAR_META_S:.2f}s "
        f"timeout_feedback={TIMEOUT_FEEDBACK_S:.1f}s "
        f"max_reenvios={MAX_REENVIOS_ORIENT_GOAL}")

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
    phase_started_at = time.time()
    last_event_at = None
    last_comm_warn = 0.0
    orient_resends = 0
    orientation_done_at = None
    orientation_done_vision_index = -1
    stopped_sent = False
    rx_janela = 0
    udp_janela = 0
    event_janela = 0
    full_route_sent = None
    full_route_running = False
    t_taxa = time.time()
    backoff = 0.5
    last_vision_at = time.time()
    last_seen_near_border = False
    recovery_attempted = False

    def send(tipo: str, segment_id: str | None, estado: dict, metricas: dict | None,
             extra: dict | None = None):
        nonlocal seq, udp_janela, last_send
        seq += 1
        pacote = _pacote_base(tipo, segment_id, estado, metricas, seq)
        if extra:
            pacote.update(extra)
        ok = _enviar(sock_tx, IP_ROBO, PORTA_ROBO_UDP, pacote, modo_simulado)
        if ok:
            udp_janela += 1
            last_send = time.time()
            destino = "SIMULADO" if modo_simulado else f"{IP_ROBO}:{PORTA_ROBO_UDP}"
            log("DEBUG", f"UDP -> ESP32 {destino} type={tipo} seq={seq} segment={segment_id}")
        else:
            log("AVISO", f"UDP -> ESP32 FALHOU type={tipo} seq={seq} segment={segment_id}")
        return ok

    log("HUMANO", "A ligar ao broadcaster do GraphProcessor...")
    while True:
        try:
            with Client(("localhost", PORTA_BROADCAST), authkey=AUTHKEY_BROADCAST) as conn:
                log("HUMANO", "Ligado ao broadcaster. Supervisao ativa.")
                backoff = 0.5

                while True:
                    estado = conn.recv()
                    rx_janela += 1
                    eventos = _receber_eventos(sock_rx)
                    event_janela += len(eventos)

                    alvo = estado.get("alvo_destino")
                    robo = estado.get("robo")
                    segment_id = _segment_id(estado)
                    metricas = _metricas_pose(
                        robo, alvo, estado.get("origem_segmento")
                    )
                    now_vision = time.time()
                    if metricas is not None:
                        last_vision_at = now_vision
                        last_seen_near_border = bool(estado.get("robo_perto_borda", False))

                    if MODO_SUPERVISAO_UDP == "TRAJETORIA_COMPLETA":
                        trajetoria_id = estado.get("trajetoria_id")
                        waypoints = estado.get("waypoints") or []

                        for evento in eventos:
                            nome = str(evento.get("event", "")).lower()
                            last_event_at = time.time()
                            origem = evento.get("_addr", "?")
                            log("DEBUG", f"UDP <- ESP32 {origem} event={nome} "
                                         f"segment={evento.get('segment_id')}")
                            if nome == "trajectory_done" and evento.get("segment_id") == full_route_sent:
                                full_route_running = False
                                phase = "full_route_done"
                                log("EVENTO", "ESP32 terminou a trajetoria completa em malha aberta.")

                        pose_inicial = _pose_robo(robo)
                        if (
                            trajetoria_id
                            and waypoints
                            and pose_inicial is not None
                            and trajetoria_id != full_route_sent
                        ):
                            full_route_sent = trajetoria_id
                            full_route_running = True
                            phase = "full_route_running"
                            current_segment = trajetoria_id
                            _log_trajetoria_completa(
                                pose_inicial,
                                waypoints,
                                trajetoria_id,
                            )
                            send(
                                "trajectory_full",
                                trajetoria_id,
                                estado,
                                metricas,
                                extra={
                                    "open_loop": True,
                                    "start_pose": pose_inicial,
                                    "initial_heading_deg": pose_inicial["heading_deg"],
                                    "coordinate_frame": "court_meters_aruco_front",
                                    "heading_convention": "atan2_y_x_degrees",
                                    "waypoints": waypoints,
                                    "waypoint_count": len(waypoints),
                                },
                            )
                            log("EVENTO",
                                f"Trajetoria completa enviada ao ESP32: "
                                f"{len(waypoints)} pontos | id={trajetoria_id}")

                        agora_taxa = time.time()
                        if agora_taxa - t_taxa >= 1.0:
                            dt = agora_taxa - t_taxa
                            estado_txt = "executando" if full_route_running else phase
                            log("HUMANO",
                                f"modo=TRAJETORIA_COMPLETA estado={estado_txt} "
                                f"pontos={len(waypoints)} rx={rx_janela/dt:.1f}Hz "
                                f"udp={udp_janela/dt:.1f}Hz fb={event_janela/dt:.1f}Hz")
                            rx_janela = udp_janela = event_janela = 0
                            t_taxa = agora_taxa
                        continue

                    if segment_id is None:
                        current_segment = None
                        phase = "idle"
                        phase_started_at = time.time()
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
                        phase_started_at = time.time()
                        orient_attempts = 0
                        orient_resends = 0
                        orientation_done_at = None
                        orientation_done_vision_index = -1
                        bad_readings = 0
                        last_distance = None
                        recovery_attempted = False
                        if metricas is not None:
                            send("orient_goal", current_segment, estado, metricas)
                            _log_metricas("nova meta de orientacao", metricas, estado)
                        else:
                            log("AVISO", "Novo segmento sem pose ArUco completa; aguardando visao.")

                    for evento in eventos:
                        if segment_id is None:
                            continue
                        if evento.get("segment_id") not in (None, current_segment):
                            continue
                        nome = str(evento.get("event", "")).lower()
                        last_event_at = time.time()
                        origem = evento.get("_addr", "?")
                        log("DEBUG", f"UDP <- ESP32 {origem} event={nome} segment={evento.get('segment_id')}")
                        if nome == "orientation_done" and phase == "orienting":
                            phase = "orientation_settling"
                            phase_started_at = time.time()
                            orientation_done_at = phase_started_at
                            orientation_done_vision_index = int(
                                estado.get("indice_visao", -1) or -1
                            )
                            log("DEBUG",
                                f"orientation_done recebido; aguardando {ORIENTATION_SETTLE_S:.2f}s "
                                f"e frame ArUco posterior a {orientation_done_vision_index}.")
                        elif nome == "arrived":
                            if metricas and metricas["at_target"]:
                                send("arrived_ok", current_segment, estado, metricas)
                                _log_metricas("chegada validada", metricas, estado)
                                phase = "waiting_next"
                                phase_started_at = time.time()
                            else:
                                send("arrived_bad", current_segment, estado, metricas)
                                if metricas:
                                    falta = max(0.0, metricas["distance_m"] - TOLERANCIA_DISTANCIA_M)
                                    log("AVISO",
                                        f"arrived_bad: faltam {falta*100:.1f}cm "
                                        f"(d={metricas['distance_m']*100:.1f}cm, "
                                        f"tol={TOLERANCIA_DISTANCIA_M*100:.0f}cm)")
                                phase = "orienting"
                                phase_started_at = time.time()
                                orient_attempts = 0
                                orient_resends = 0
                                bad_readings = 0
                                last_distance = None
                                send("orient_goal", current_segment, estado, metricas)
                                _log_metricas("chegada rejeitada", metricas, estado)
                        elif nome == "recovery_done" and phase == "recovering":
                            phase = "orienting"
                            phase_started_at = time.time()
                            orient_resends = 0
                            log("EVENTO", "Recuperacao local concluida; aguardando regresso da visao.")

                    now = time.time()
                    if (
                        segment_id is not None
                        and phase == "orientation_settling"
                        and orientation_done_at is not None
                        and now - orientation_done_at >= ORIENTATION_SETTLE_S
                        and int(estado.get("indice_visao", -1) or -1) > orientation_done_vision_index
                        and metricas is not None
                    ):
                        orientation_done_at = None
                        if metricas["aligned_release"]:
                            phase = "moving"
                            phase_started_at = now
                            bad_readings = 0
                            orient_attempts = 0
                            orient_resends = 0
                            last_distance = metricas["distance_m"]
                            send("move_permission", current_segment, estado, metricas)
                            _log_metricas(
                                "orientacao validada em frame fresco com margem",
                                metricas,
                                estado,
                            )
                        else:
                            phase = "orienting"
                            phase_started_at = now
                            orient_attempts += 1
                            if orient_attempts >= MAX_TENTATIVAS_ORIENTACAO:
                                log("AVISO",
                                    f"Orientacao ainda fora apos {orient_attempts} passos; "
                                    "continuando sem bloquear.")
                                orient_attempts = 0
                            send("orientation_correction", current_segment, estado, metricas)
                            _log_metricas("nova correcao apos assentamento", metricas, estado)

                    if (
                        segment_id is not None
                        and phase == "orienting"
                        and metricas is not None
                        and now - last_send >= REENVIAR_META_S
                    ):
                        if orient_resends >= MAX_REENVIOS_ORIENT_GOAL:
                            log("AVISO",
                                f"Sem orientation_done apos {orient_resends} reenvios; "
                                "contador reiniciado sem bloquear.")
                            orient_resends = 0
                        orient_resends += 1
                        send("orient_goal", current_segment, estado, metricas)

                    if phase == "moving" and metricas is not None:
                        desvio_lateral = metricas.get("cross_track_error_m")
                        piorou = (
                            abs(metricas["heading_error_deg"]) > DESVIO_MOVIMENTO_ANGULO_GRAUS
                            or (
                                desvio_lateral is not None
                                and desvio_lateral > DESVIO_LATERAL_M
                            )
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
                            phase_started_at = time.time()
                            orient_attempts = 0
                            orient_resends = 0
                            bad_readings = 0
                            _log_metricas("desvio persistente - parar/corrigir", metricas, estado)

                    if (
                        phase == "moving"
                        and metricas is None
                        and RECUPERACAO_PERDA_VISAO_ATIVA
                        and last_seen_near_border
                        and not recovery_attempted
                        and now - last_vision_at >= TIMEOUT_PERDA_VISAO_S
                    ):
                        recovery_attempted = True
                        send(
                            "vision_recovery",
                            current_segment,
                            estado,
                            None,
                            extra={"recovery_distance_cm": DISTANCIA_RECUPERACAO_CM},
                        )
                        phase = "recovering"
                        phase_started_at = time.time()
                        log("AVISO",
                            f"Visao perdida junto a borda por {now-last_vision_at:.2f}s; "
                            f"recuperacao local de {DISTANCIA_RECUPERACAO_CM:.0f}cm solicitada.")

                    agora_taxa = time.time()
                    if agora_taxa - t_taxa >= 1.0:
                        dt = agora_taxa - t_taxa
                        sufixo = " [SIM]" if modo_simulado else ""
                        sem_feedback = None if last_event_at is None else agora_taxa - last_event_at
                        log("DEBUG",
                            f"taxa controlo: rx={rx_janela/dt:.1f}Hz "
                            f"udp={udp_janela/dt:.1f}Hz "
                            f"fb={event_janela/dt:.1f}Hz{sufixo} "
                            f"fase={phase} indice_visao={estado.get('indice_visao')}")
                        log("HUMANO", _resumo_execucao(phase, metricas, estado, sem_feedback))
                        if (
                            not modo_simulado
                            and segment_id is not None
                            and phase in ("orienting", "orientation_settling", "moving")
                            and (agora_taxa - phase_started_at) >= TIMEOUT_FEEDBACK_S
                            and event_janela == 0
                            and (agora_taxa - last_comm_warn) >= TIMEOUT_FEEDBACK_S
                        ):
                            last_comm_warn = agora_taxa
                            log("AVISO",
                                f"Sem feedback do ESP32 ha {agora_taxa - phase_started_at:.1f}s "
                                f"na fase {phase}. Verificar IP, portas, Wi-Fi e segment_id.")
                        rx_janela = udp_janela = event_janela = 0
                        t_taxa = agora_taxa

        except (ConnectionRefusedError, OSError):
            log("AVISO", f"GraphProcessor nao disponivel. A retentar em {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)
        except (EOFError, ConnectionResetError):
            log("AVISO", "Ligacao ao GraphProcessor caiu. A reabrir...")
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
