"""
VisionProcessing.py — Deteção de Bolas (YOLO) + Robô (ArUco) UFSC/FEUP
========================================================================
Servidor na porta 6000. Recebe frames do imageStreaming, executa
inferência YOLO para bolas, deteta marcadores ArUco para localização
e orientação do robô, e encaminha os resultados para o retificador
(porta 6001).

Fluxo:
  imageStreaming → [porta 6000] → VisionProcessing → [porta 6001] → retificador
                                                                          ↓
  imageStreaming ← [LIBERADO] ←────────────────────────────────────────────

OTIMIZAÇÃO DURANTE DISPARO (Fase 2.1):
  Liga-se ao broadcaster do GraphProcessor (porta 6021) e recebe o
  estado em tempo real. Quando há disparo ativo, *salta a inferência
  YOLO* — só corre ArUco. Isto multiplica drasticamente a cadência
  da deteção do robô, importante para o controlador.
  Detalhe: o handshake LIBERADO mantém-se inalterado; o que aumenta
  é apenas a frequência efetiva por frame ser muito menor (sem YOLO).

Marcadores ArUco:
  ID 0 → Frente do robô  (DICT_4X4_50)
  ID 1 → Traseira do robô (DICT_4X4_50)
  A orientação é o ângulo do vector traseiro→frente (em graus, 0°=direita)

Health-check na porta 6002 (TCP simples para o MasterControl).
"""

import cv2
import numpy as np
import time
import sys
import socket
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from multiprocessing.connection import Listener, Client
from datetime import datetime

from _COMMON.logging_utils import log as _log
import _CONFIG.system_parameters as _params

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
BASE_PATH = Path(__file__).resolve().parents[1]
MOD       = "VISAO"

MODELO_PATH = (BASE_PATH / "runs" / "detect" / "treino_bolas_v24"
               / "weights" / "best.pt")

PORTA_ENTRADA       = 6000
PORTA_HEALTH        = 6002
PORTA_RET           = 6001
PORTA_BROADCAST     = 6021    # NOVO — escutamos o GraphProcessor
AUTHKEY_VIS         = b"bolas_ufsc"
AUTHKEY_RET         = b"retificador_ufsc"
AUTHKEY_BROADCAST   = b"controlador_ufsc"

CONFIANCA_MIN  = 0.50
DISPOSITIVO    = 0             # 0=GPU CUDA, "cpu" para CPU

# ── ArUco ─────────────────────────────────────
ARUCO_DICT     = cv2.aruco.DICT_4X4_50
ID_FRONTAL     = 0
ID_TRASEIRO    = 1

# Pré-processamento CLAHE
CLAHE_CLIP     = 2.0
CLAHE_GRID     = (8, 8)
ARUCO_LARGURA_PX = 640
ARUCO_USAR_CLAHE = False
ARUCO_PERSISTENCIA_S = 0.35
ARUCO_SUAVIZACAO = 0.35


# ─────────────────────────────────────────────
#  WRAPPER DO LOG (compatibilidade com chamadas antigas)
# ─────────────────────────────────────────────
def log(nivel: str, msg: str):
    _log(MOD, nivel, msg)


# ─────────────────────────────────────────────
#  HEALTH-CHECK SERVER (porta 6002)
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
#  CLIENTE DO BROADCASTER DO GRAPHPROCESSOR
# ─────────────────────────────────────────────
# Estado partilhado: True quando o GraphProcessor está em disparo
# (aguarda_inicio ou em_varrimento). Acedido pelo loop principal sem
# lock — escritas atómicas em CPython, leitura de bool é segura.
_em_disparo: bool = False


def loop_cliente_broadcaster():
    """
    Liga-se ao broadcaster do GraphProcessor e mantém atualizado o
    flag global _em_disparo. Reconecta automaticamente se a ligação
    cair. Pode arrancar antes do GraphProcessor estar pronto — fica
    em retry silencioso até o GraphProcessor abrir a porta.
    """
    global _em_disparo
    backoff = 0.5

    while True:
        try:
            with Client(("localhost", PORTA_BROADCAST),
                        authkey=AUTHKEY_BROADCAST) as conn:
                log("DEBUG", "Ligado ao broadcaster do GraphProcessor "
                             f"(porta {PORTA_BROADCAST})")
                backoff = 0.5

                while True:
                    estado = conn.recv()  # bloqueia até chegar pacote
                    fase = estado.get("fase")
                    novo = (fase in ("aguarda_inicio", "em_varrimento", "global_executar"))
                    if novo != _em_disparo:
                        _em_disparo = novo
                        if novo:
                            modo_op = estado.get("modo_operacao", "FAIXAS")
                            if modo_op == "GLOBAL":
                                wp_idx = estado.get("waypoint_idx", "?")
                                wp_tot = estado.get("waypoints_total", "?")
                                log("HUMANO",
                                    f"Disparo GLOBAL ativo — YOLO desligada "
                                    f"(waypoint {wp_idx}/{wp_tot}).")
                            else:
                                log("HUMANO", "Disparo ativo — YOLO desligada (só ArUco).")
                            log("DEBUG",  f"fase={fase}, faixa={estado.get('faixa_label')}")
                        else:
                            log("HUMANO", "Disparo terminado — YOLO reativada.")

        except (ConnectionRefusedError, OSError):
            # GraphProcessor ainda não está pronto — esperar um pouco
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)
        except (EOFError, ConnectionResetError):
            log("DEBUG", "Ligação ao broadcaster caiu. A reabrir...")
            _em_disparo = False
            time.sleep(0.5)
        except Exception as e:
            log("DEBUG", f"Erro no cliente do broadcaster: {e}")
            _em_disparo = False
            time.sleep(1.0)


# ─────────────────────────────────────────────
#  DETEÇÃO ARUCO
# ─────────────────────────────────────────────
def criar_detetor_aruco():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()

    parameters.adaptiveThreshWinSizeMin   = 3
    parameters.adaptiveThreshWinSizeMax   = 83
    parameters.adaptiveThreshWinSizeStep  = 4
    parameters.adaptiveThreshConstant     = 7
    parameters.minMarkerPerimeterRate     = 0.01
    parameters.maxMarkerPerimeterRate     = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.cornerRefinementMethod     = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize    = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.05
    parameters.errorCorrectionRate        = 0.7

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    log("HUMANO", f"Detetor ArUco pronto (IDs {ID_FRONTAL}=frente, {ID_TRASEIRO}=trás).")
    return detector


class PersistenciaAruco:
    """Mantém a última pose ArUco por alguns frames para evitar flicker."""

    def __init__(self, persistencia_s: float, suavizacao: float):
        self.persistencia_s = max(0.0, float(persistencia_s))
        self.suavizacao = min(max(float(suavizacao), 0.0), 0.95)
        self._ultimo = {
            "frontal": None,
            "traseiro": None,
        }
        self._t_ultimo = {
            "frontal": 0.0,
            "traseiro": 0.0,
        }
        self._orientacao = None

    def atualizar(self, det: dict, agora: float | None = None) -> dict:
        agora = time.time() if agora is None else agora
        saida = {
            "frontal": None,
            "traseiro": None,
            "orientacao_graus": None,
        }

        for chave in ("frontal", "traseiro"):
            pos = det.get(chave)
            if pos:
                anterior = self._ultimo.get(chave)
                if anterior:
                    a = self.suavizacao
                    pos = {
                        "cx": round(anterior["cx"] * a + pos["cx"] * (1.0 - a), 1),
                        "cy": round(anterior["cy"] * a + pos["cy"] * (1.0 - a), 1),
                    }
                self._ultimo[chave] = pos
                self._t_ultimo[chave] = agora
                saida[chave] = dict(pos)
            else:
                anterior = self._ultimo.get(chave)
                if anterior and (agora - self._t_ultimo[chave]) <= self.persistencia_s:
                    saida[chave] = {**anterior, "persistido": True}

        if saida["frontal"] and saida["traseiro"]:
            dx = saida["frontal"]["cx"] - saida["traseiro"]["cx"]
            dy = saida["frontal"]["cy"] - saida["traseiro"]["cy"]
            self._orientacao = round(float(np.degrees(np.arctan2(-dy, dx))), 2)
            saida["orientacao_graus"] = self._orientacao
        elif self._orientacao is not None:
            mais_recente = max(self._t_ultimo.values())
            if (agora - mais_recente) <= self.persistencia_s:
                saida["orientacao_graus"] = self._orientacao

        return saida


def detetar_robo(frame_gray, detector, clahe=None, escala_saida: float = 1.0) -> dict:
    """
    Deteta os dois marcadores ArUco do robô num frame grayscale.
    Aplica CLAHE antes para robustez a iluminação.
    """
    resultado = {
        "frontal":          None,
        "traseiro":         None,
        "orientacao_graus": None,
    }

    candidatos = [frame_gray]
    if clahe is not None:
        candidatos.append(clahe.apply(frame_gray))
    candidatos.append(cv2.equalizeHist(frame_gray))

    corners = ids = None
    for frame_proc in candidatos:
        corners, ids, _ = detector.detectMarkers(frame_proc)
        if ids is not None:
            ids_set = set(int(v) for v in ids.flatten())
            if ID_FRONTAL in ids_set or ID_TRASEIRO in ids_set:
                break

    if ids is None:
        return resultado

    for i, marker_id in enumerate(ids.flatten()):
        cx = float(corners[i][0][:, 0].mean()) * escala_saida
        cy = float(corners[i][0][:, 1].mean()) * escala_saida
        if marker_id == ID_FRONTAL:
            resultado["frontal"]  = {"cx": round(cx, 1), "cy": round(cy, 1)}
        elif marker_id == ID_TRASEIRO:
            resultado["traseiro"] = {"cx": round(cx, 1), "cy": round(cy, 1)}

    if resultado["frontal"] and resultado["traseiro"]:
        dx = resultado["frontal"]["cx"]  - resultado["traseiro"]["cx"]
        dy = resultado["frontal"]["cy"]  - resultado["traseiro"]["cy"]
        angulo = float(np.degrees(np.arctan2(-dy, dx)))
        resultado["orientacao_graus"] = round(angulo, 2)

    return resultado


def anotar_robo(frame, robo: dict):
    COR_FRONTAL  = (255, 80,  220)
    COR_TRASEIRO = (180, 0,   255)
    COR_SETA     = (0,   255, 180)
    RAIO_PONTO   = 10

    for chave, cor, label in [
        ("frontal",  COR_FRONTAL,  "FRENTE"),
        ("traseiro", COR_TRASEIRO, "TRAS"),
    ]:
        pos = robo.get(chave)
        if pos:
            cx, cy = int(pos["cx"]), int(pos["cy"])
            cv2.circle(frame, (cx, cy), RAIO_PONTO,     cor, -1)
            cv2.circle(frame, (cx, cy), RAIO_PONTO + 2, (255, 255, 255), 2)

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame,
                          (cx + 14, cy - th - 6),
                          (cx + 14 + tw + 4, cy + 2),
                          cor, -1)
            cv2.putText(frame, label,
                        (cx + 16, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    if robo["frontal"] and robo["traseiro"] and robo["orientacao_graus"] is not None:
        fx, fy = int(robo["frontal"]["cx"]),  int(robo["frontal"]["cy"])
        tx, ty = int(robo["traseiro"]["cx"]), int(robo["traseiro"]["cy"])
        cv2.line(frame, (tx, ty), (fx, fy), COR_SETA, 2)
        ang_rad = np.radians(robo["orientacao_graus"])
        ex = int(fx + 35 * np.cos(ang_rad))
        ey = int(fy - 35 * np.sin(ang_rad))
        cv2.arrowedLine(frame, (fx, fy), (ex, ey), COR_SETA, 2, tipLength=0.35)
        cv2.putText(frame, f"{robo['orientacao_graus']:.1f}deg",
                    (tx + 5, ty + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COR_SETA, 1)

    return frame


# ─────────────────────────────────────────────
#  ENVIO PARA RETIFICADOR (com retry)
# ─────────────────────────────────────────────
def enviar_para_retificador(pacote_ret: dict, tentativas: int = 3) -> bool:
    for i in range(tentativas):
        try:
            with Client(("localhost", PORTA_RET), authkey=AUTHKEY_RET) as c:
                c.send(pacote_ret)
                resposta = c.recv()
                return resposta == "LIBERADO"
        except ConnectionRefusedError:
            if i < tentativas - 1:
                espera = 1.0 * (i + 1)
                log("AVISO",
                    f"Retificador não responde (tentativa {i+1}/{tentativas}). "
                    f"A aguardar {espera:.0f}s...")
                time.sleep(espera)
            else:
                log("ERRO", "Retificador inacessível após todas as tentativas.")
        except Exception as e:
            log("ERRO", f"Erro ao contactar retificador: {e}")
            break
    return False


# ─────────────────────────────────────────────
#  SERVIDOR PRINCIPAL
# ─────────────────────────────────────────────
def iniciar_visao():
    global ARUCO_LARGURA_PX, ARUCO_USAR_CLAHE, ARUCO_PERSISTENCIA_S, ARUCO_SUAVIZACAO
    cfg = _params.carregar()
    enviar_frame_debug = bool(int(cfg.get("guardar_imagens_debug", 0)))
    intervalo_frame_debug_s = float(cfg.get("intervalo_guardar_imagens_s", 5.0))
    proximo_frame_debug = 0.0
    ARUCO_LARGURA_PX = int(cfg.get("aruco_largura_px", ARUCO_LARGURA_PX))
    ARUCO_USAR_CLAHE = bool(int(cfg.get("aruco_usar_clahe", 0)))
    ARUCO_PERSISTENCIA_S = float(cfg.get("aruco_persistencia_s", ARUCO_PERSISTENCIA_S))
    ARUCO_SUAVIZACAO = float(cfg.get("aruco_suavizacao", ARUCO_SUAVIZACAO))

    # ── Carregar modelo YOLO ───────────────────────────────
    log("HUMANO", "A carregar modelo YOLO...")
    log("DEBUG",  f"caminho do modelo: {MODELO_PATH}")
    if not MODELO_PATH.exists():
        log("ERRO", f"Modelo não encontrado: {MODELO_PATH}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
        t0    = time.time()
        model = YOLO(str(MODELO_PATH))
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(source=dummy, conf=CONFIANCA_MIN,
                      device=DISPOSITIVO, verbose=False)
        t_load = time.time() - t0
        log("HUMANO", f"Modelo YOLO carregado em {t_load:.1f}s.")
        log("DEBUG",  f"dispositivo={DISPOSITIVO}, confiança_min={CONFIANCA_MIN}")
    except Exception as e:
        log("ERRO", f"Falha ao carregar YOLO: {e}")
        sys.exit(1)

    # ── Inicializar detetor ArUco + CLAHE ──────────────────
    aruco_detector = criar_detetor_aruco()
    clahe = (cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
             if ARUCO_USAR_CLAHE else None)
    persistencia_aruco = PersistenciaAruco(ARUCO_PERSISTENCIA_S, ARUCO_SUAVIZACAO)

    # ── Health-server ──────────────────────────────────────
    iniciar_health_server()

    # ── Cliente do broadcaster (em background) ─────────────
    threading.Thread(target=loop_cliente_broadcaster, daemon=True).start()

    # ── Estatísticas de sessão ─────────────────────────────
    stats = {
        "frames":           0,
        "bolas_total":      0,
        "robo_detetado":    0,
        "latencia_soma":    0.0,
        "erros":            0,
        "frames_yolo_skip": 0,
    }

    log("HUMANO", "VisionProcessing pronto. A aguardar frames...")
    log("DEBUG",  f"servidor ativo na porta {PORTA_ENTRADA}")
    log("DEBUG",  f"envio de frames para escrita/debug={'ON' if enviar_frame_debug else 'OFF'}")
    log("DEBUG",  f"ArUco: largura={ARUCO_LARGURA_PX}px | CLAHE={'ON' if ARUCO_USAR_CLAHE else 'OFF'} | "
                  f"persistência={ARUCO_PERSISTENCIA_S:.2f}s | suavização={ARUCO_SUAVIZACAO:.2f}")

    address = ("localhost", PORTA_ENTRADA)
    with Listener(address, authkey=AUTHKEY_VIS) as listener:
        while True:
            try:
                conn = listener.accept()
            except ConnectionAbortedError:
                log("AVISO", "Ligação rejeitada (sem autenticação) — a ignorar.")
                continue
            except Exception as e:
                log("ERRO", f"Erro ao aceitar ligação: {e}")
                continue

            with conn:
                try:
                    pacote = conn.recv()
                    indice = stats["frames"]
                    t_recv = time.time()
                    frame  = pacote["frame"]
                    escala_origem_x = float(pacote.get("escala_origem_x", 1.0))
                    escala_origem_y = float(pacote.get("escala_origem_y", 1.0))

                    # Lê o flag global no início do processamento (snapshot
                    # consistente para este frame mesmo se o flag mudar a meio)
                    skip_yolo = _em_disparo

                    # ── Inferência YOLO (bolas) — saltada em disparo ───
                    bolas = []
                    ms_yolo = 0.0
                    if not skip_yolo:
                        t_inf = time.time()
                        results = model.predict(
                            source=frame,
                            conf=CONFIANCA_MIN,
                            device=DISPOSITIVO,
                            verbose=False,
                        )
                        ms_yolo = (time.time() - t_inf) * 1000

                        for r in results:
                            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                            boxes_conf = (r.boxes.conf.cpu().numpy()
                                          if hasattr(r.boxes, "conf") else [])
                            for idx_b, box in enumerate(boxes_xyxy):
                                conf = (float(boxes_conf[idx_b])
                                        if len(boxes_conf) > idx_b else 0.0)
                                x1p, y1p = int(box[0]), int(box[1])
                                x2p, y2p = int(box[2]), int(box[3])
                                bolas.append({
                                    "x1": int(round(x1p * escala_origem_x)),
                                    "y1": int(round(y1p * escala_origem_y)),
                                    "x2": int(round(x2p * escala_origem_x)),
                                    "y2": int(round(y2p * escala_origem_y)),
                                    "x1_proc": x1p,
                                    "y1_proc": y1p,
                                    "x2_proc": x2p,
                                    "y2_proc": y2p,
                                    "conf": round(conf, 3),
                                })
                    else:
                        stats["frames_yolo_skip"] += 1

                    # ── Deteção ArUco (robô) — sempre ───────────────────
                    t_aruco  = time.time()
                    frame_aruco = frame
                    escala_aruco = 1.0
                    h_proc, w_proc = frame.shape[:2]
                    if ARUCO_LARGURA_PX > 0 and ARUCO_LARGURA_PX < w_proc:
                        escala_aruco = w_proc / float(ARUCO_LARGURA_PX)
                        novo_h = max(1, int(round(h_proc / escala_aruco)))
                        frame_aruco = cv2.resize(
                            frame, (ARUCO_LARGURA_PX, novo_h),
                            interpolation=cv2.INTER_AREA,
                        )
                    gray = cv2.cvtColor(frame_aruco, cv2.COLOR_BGR2GRAY)
                    robo_det = detetar_robo(
                        gray, aruco_detector, clahe,
                        escala_saida=escala_aruco,
                    )
                    robo = persistencia_aruco.atualizar(robo_det)
                    for chave in ("frontal", "traseiro"):
                        if robo.get(chave):
                            robo[chave]["cx"] = round(robo[chave]["cx"] * escala_origem_x, 1)
                            robo[chave]["cy"] = round(robo[chave]["cy"] * escala_origem_y, 1)
                    ms_aruco = (time.time() - t_aruco) * 1000

                    # ── Log resumido ───────────────────────────────────
                    robo_str = "—"
                    if robo["frontal"] or robo["traseiro"]:
                        partes = []
                        if robo["frontal"]:
                            partes.append("F~" if robo["frontal"].get("persistido") else "F✓")
                        if robo["traseiro"]:
                            partes.append("T~" if robo["traseiro"].get("persistido") else "T✓")
                        if robo["orientacao_graus"] is not None:
                            partes.append(f"{robo['orientacao_graus']:.1f}°")
                        robo_str = " ".join(partes)
                        stats["robo_detetado"] += 1

                    if skip_yolo:
                        log("DEBUG",
                            f"Frame {indice:04d} [DISPARO — sem YOLO] | "
                            f"robô={robo_str} [{ms_aruco:.0f}ms]")
                    else:
                        log("DEBUG",
                            f"Frame {indice:04d} | bolas={len(bolas)} "
                            f"[{ms_yolo:.0f}ms] | robô={robo_str} [{ms_aruco:.0f}ms]")

                    frame_debug = None
                    agora_debug = time.time()
                    if enviar_frame_debug and agora_debug >= proximo_frame_debug:
                        proximo_frame_debug = agora_debug + intervalo_frame_debug_s
                        frame_debug = frame.copy()
                        for idx_b, b in enumerate(bolas):
                            x1 = b.get("x1_proc", b["x1"])
                            y1 = b.get("y1_proc", b["y1"])
                            x2 = b.get("x2_proc", b["x2"])
                            y2 = b.get("y2_proc", b["y2"])
                            cv2.rectangle(frame_debug,
                                          (x1, y1), (x2, y2),
                                          (0, 255, 0), 2)
                            label = f"bola {idx_b+1}  {b['conf']:.2f}"
                            (tw, th), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            ty = max(y1 - 8, th + 4)
                            cv2.rectangle(frame_debug,
                                          (x1, ty - th - 4),
                                          (x1 + tw + 4, ty + 2),
                                          (0, 255, 0), -1)
                            cv2.putText(frame_debug, label,
                                        (x1 + 2, ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        frame_debug = anotar_robo(frame_debug, robo)

                    # ── Encaminhar para retificador ─────────────────────
                    pacote_ret = {
                        "frame":           frame_debug,
                        "bolas_px":        bolas,
                        "robo_px":         robo,
                        "indice":          indice,
                        "timestamp_visao": pacote["timestamp"],
                    }

                    # Em disparo o robô tem de ser sempre enviado mesmo sem bolas.
                    # Fora de disparo mantém-se a otimização: só envia se há dados.
                    tem_dados = bolas or robo["frontal"] or robo["traseiro"] or frame_debug is not None
                    if tem_dados or skip_yolo:
                        ok = enviar_para_retificador(pacote_ret)
                        if not ok:
                            log("AVISO", f"Frame {indice:04d}: retificação falhou.")
                    else:
                        log("DEBUG", f"Frame {indice:04d}: sem bolas nem robô — a ignorar.")

                    conn.send("LIBERADO")

                    # ── Estatísticas ───────────────────────────────────
                    stats["frames"]       += 1
                    stats["bolas_total"]  += len(bolas)
                    ms_total = (time.time() - t_recv) * 1000
                    stats["latencia_soma"] += ms_total

                    if stats["frames"] % 50 == 0:
                        media_lat = stats["latencia_soma"] / stats["frames"]
                        skips = stats["frames_yolo_skip"]
                        log("HUMANO",
                            f"{stats['frames']} frames processados "
                            f"(latência média {media_lat:.0f}ms"
                            + (f", {skips} sem YOLO" if skips else "")
                            + ").")

                except Exception as e:
                    log("ERRO", f"Erro ao processar frame: {e}")
                    stats["erros"] += 1
                    try:
                        conn.send("LIBERADO")
                    except Exception:
                        pass


# ─────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    iniciar_visao()



