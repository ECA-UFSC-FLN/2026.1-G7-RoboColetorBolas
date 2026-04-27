"""
VisionProcessing.py — Deteção de Bolas (YOLO) + Robô (ArUco) UFSC/FEUP
========================================================================
Servidor na porta 6000. Recebe frames do imageStreaming,
executa inferência com o modelo YOLO personalizado para bolas,
deteta marcadores ArUco para localização e orientação do robô,
e encaminha os resultados para o retificador (porta 6001).

Fluxo:
  imageStreaming → [porta 6000] → VisionProcessing → [porta 6001] → retificador
                                                                          ↓
  imageStreaming ← [LIBERADO] ←────────────────────────────────────────────

Marcadores ArUco:
  ID 0 → Frente do robô  (DICT_4X4_50)
  ID 1 → Traseira do robô (DICT_4X4_50)
  A orientação é o ângulo do vector traseiro→frente (em graus, 0°=direita)

NOTA: Uma thread separada fica à escuta na porta 6002 (PORTA_HEALTH)
      exclusivamente para responder a health-checks do MasterControl.
"""

import cv2
import numpy as np
import time
import sys
import socket
import threading
from pathlib import Path
from multiprocessing.connection import Listener, Client
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
BASE_PATH = Path(__file__).parent.resolve()

MODELO_PATH = (BASE_PATH / "runs" / "detect" / "treino_bolas_v24"
               / "weights" / "best.pt")

PORTA_ENTRADA  = 6000
PORTA_HEALTH   = 6002
PORTA_RET      = 6001
AUTHKEY_VIS    = b"bolas_ufsc"
AUTHKEY_RET    = b"retificador_ufsc"

CONFIANCA_MIN  = 0.50
DISPOSITIVO    = 0             # 0=GPU CUDA, "cpu" para CPU

# ── ArUco ─────────────────────────────────────
ARUCO_DICT     = cv2.aruco.DICT_4X4_50
ID_FRONTAL     = 0             # marcador rosa na frente do robô
ID_TRASEIRO    = 1             # marcador roxo na traseira do robô

# Pré-processamento CLAHE para robustez a luz frontal intensa
CLAHE_CLIP     = 2.0
CLAHE_GRID     = (8, 8)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
ICONS = {"INFO": "·", "OK": "✓", "ERRO": "✗", "AVISO": "!", "FASE": "▶️"}

def log(nivel: str, msg: str):
    ts   = datetime.now().strftime("%H:%M:%S")
    icon = ICONS.get(nivel, "·")
    cor  = {
        "OK":    "\033[92m",
        "ERRO":  "\033[91m",
        "AVISO": "\033[93m",
        "FASE":  "\033[96m",
        "INFO":  "\033[0m",
    }.get(nivel, "\033[0m")
    print(f"{cor}[{ts}] [VISAO        ] {icon} {msg}\033[0m", flush=True)

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

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    log("INFO", f"Health-check ativo na porta {porta}")

# ─────────────────────────────────────────────
#  DETEÇÃO ARUCO
# ─────────────────────────────────────────────
def criar_detetor_aruco():
    """
    Inicializa o detetor ArUco com parâmetros ajustados para maior
    robustez em condições de iluminação variável.
    """
    dictionary  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters  = cv2.aruco.DetectorParameters()

    # Mais tolerante a variações de contraste e perspetiva
    parameters.adaptiveThreshWinSizeMin  = 3
    parameters.adaptiveThreshWinSizeMax  = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate    = 0.02   # aceita marcadores mais distantes
    parameters.maxMarkerPerimeterRate    = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    log("OK", f"Detetor ArUco iniciado (DICT_4X4_50 | ID_FRONTAL={ID_FRONTAL} ID_TRASEIRO={ID_TRASEIRO})")
    return detector


def detetar_robo(frame_gray, detector, clahe) -> dict:
    """
    Deteta os dois marcadores ArUco do robô num frame em grayscale.

    Aplica CLAHE antes da deteção para compensar overexposure e
    variações de iluminação (luz frontal, artificial, natural).

    Devolve um dicionário com:
      frontal  : {"cx", "cy"} em píxeis, ou None
      traseiro : {"cx", "cy"} em píxeis, ou None
      orientacao_graus : ângulo do vector traseiro→frontal, ou None
                         (0° = direita, 90° = cima, sentido anti-horário)
    """
    resultado = {
        "frontal":           None,
        "traseiro":          None,
        "orientacao_graus":  None,
    }

    # CLAHE equaliza contraste localmente — robusto a overexposure
    frame_eq = clahe.apply(frame_gray)

    corners, ids, _ = detector.detectMarkers(frame_eq)

    if ids is None:
        return resultado

    ids_flat = ids.flatten()

    for i, marker_id in enumerate(ids_flat):
        # Centróide = média dos 4 cantos do marcador
        cx = float(corners[i][0][:, 0].mean())
        cy = float(corners[i][0][:, 1].mean())

        if marker_id == ID_FRONTAL:
            resultado["frontal"]  = {"cx": round(cx, 1), "cy": round(cy, 1)}
        elif marker_id == ID_TRASEIRO:
            resultado["traseiro"] = {"cx": round(cx, 1), "cy": round(cy, 1)}

    # Calcula orientação apenas quando ambos os marcadores são visíveis
    if resultado["frontal"] and resultado["traseiro"]:
        dx = resultado["frontal"]["cx"]  - resultado["traseiro"]["cx"]
        dy = resultado["frontal"]["cy"]  - resultado["traseiro"]["cy"]
        # arctan2 em coordenadas de imagem (y cresce para baixo)
        angulo = float(np.degrees(np.arctan2(-dy, dx)))  # -dy para eixo Y standard
        resultado["orientacao_graus"] = round(angulo, 2)

    return resultado


def anotar_robo(frame, robo: dict):
    COR_FRONTAL  = (255, 80,  220)   # rosa/magenta — frente
    COR_TRASEIRO = (180, 0,   255)   # roxo          — traseira
    COR_SETA     = (0,   255, 180)   # verde-azulado — orientação
    RAIO_PONTO   = 10                # mesmo peso visual das bolas

    for chave, cor, label in [
        ("frontal",  COR_FRONTAL,  "FRENTE"),
        ("traseiro", COR_TRASEIRO, "TRAS"),
    ]:
        pos = robo.get(chave)
        if pos:
            cx, cy = int(pos["cx"]), int(pos["cy"])

            # Ponto cheio bem visível (igual ao estilo das bolas)
            cv2.circle(frame, (cx, cy), RAIO_PONTO,     cor, -1)
            # Anel branco por fora para contraste em qualquer fundo
            cv2.circle(frame, (cx, cy), RAIO_PONTO + 2, (255, 255, 255), 2)

            # Legenda com fundo escuro (igual às bolas)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame,
                          (cx + 14, cy - th - 6),
                          (cx + 14 + tw + 4, cy + 2),
                          cor, -1)
            cv2.putText(frame, label,
                        (cx + 16, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Seta de orientação quando ambos estão visíveis
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
    # ── Carregar modelo YOLO ───────────────────────────────
    log("FASE", "Carregando modelo YOLO...")
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
        log("OK", f"Modelo carregado em {t_load:.1f}s | dispositivo={DISPOSITIVO}")
    except Exception as e:
        log("ERRO", f"Falha ao carregar YOLO: {e}")
        sys.exit(1)

    # ── Inicializar detetor ArUco + CLAHE ──────────────────
    aruco_detector = criar_detetor_aruco()
    clahe          = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

    # ── Health-server ──────────────────────────────────────
    iniciar_health_server()

    # ── Estatísticas de sessão ─────────────────────────────
    stats = {
        "frames":        0,
        "bolas_total":   0,
        "robo_detetado": 0,
        "latencia_soma": 0.0,
        "erros":         0,
    }

    log("FASE", f"Servidor ativo na porta {PORTA_ENTRADA}. Aguardando frames...")

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

                    # ── Inferência YOLO (bolas) ────────────
                    t_inf = time.time()
                    results = model.predict(
                        source=frame,
                        conf=CONFIANCA_MIN,
                        device=DISPOSITIVO,
                        verbose=False,
                    )
                    ms_yolo = (time.time() - t_inf) * 1000

                    bolas = []
                    for r in results:
                        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                        boxes_conf = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else []
                        for idx_b, box in enumerate(boxes_xyxy):
                            conf = float(boxes_conf[idx_b]) if len(boxes_conf) > idx_b else 0.0
                            bolas.append({
                                "x1": int(box[0]), "y1": int(box[1]),
                                "x2": int(box[2]), "y2": int(box[3]),
                                "conf": round(conf, 3),
                            })

                    # ── Deteção ArUco (robô) ───────────────
                    t_aruco = time.time()
                    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    robo    = detetar_robo(gray, aruco_detector, clahe)
                    ms_aruco = (time.time() - t_aruco) * 1000

                    # ── Log resumido ───────────────────────
                    robo_str = "—"
                    if robo["frontal"] or robo["traseiro"]:
                        partes = []
                        if robo["frontal"]:
                            partes.append("F✓")
                        if robo["traseiro"]:
                            partes.append("T✓")
                        if robo["orientacao_graus"] is not None:
                            partes.append(f"{robo['orientacao_graus']:.1f}°")
                        robo_str = " ".join(partes)
                        stats["robo_detetado"] += 1

                    log("INFO",
                        f"Frame {indice:04d} | bolas={len(bolas)} "
                        f"[{ms_yolo:.0f}ms] | robô={robo_str} [{ms_aruco:.0f}ms]")

                    # ── Anotar frame ───────────────────────
                    frame_anotado = frame.copy()

                    # Bolas
                    for idx_b, b in enumerate(bolas):
                        cv2.rectangle(frame_anotado,
                                      (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                                      (0, 255, 0), 2)
                        label = f"bola {idx_b+1}  {b['conf']:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        ty = max(b["y1"] - 8, th + 4)
                        cv2.rectangle(frame_anotado,
                                      (b["x1"], ty - th - 4),
                                      (b["x1"] + tw + 4, ty + 2),
                                      (0, 255, 0), -1)
                        cv2.putText(frame_anotado, label,
                                    (b["x1"] + 2, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # Robô
                    frame_anotado = anotar_robo(frame_anotado, robo)

                    # ── Encaminhar para retificador ────────
                    # Envia sempre (mesmo sem bolas) para que o retificador
                    # possa registar a posição do robô.
                    pacote_ret = {
                        "frame":           frame_anotado,
                        "bolas_px":        bolas,
                        "robo_px":         robo,          # NOVO: posição ArUco
                        "indice":          indice,
                        "timestamp_visao": pacote["timestamp"],
                    }

                    tem_dados = bolas or robo["frontal"] or robo["traseiro"]
                    if tem_dados:
                        ok = enviar_para_retificador(pacote_ret)
                        if ok:
                            log("OK", f"Frame {indice:04d} retificado com sucesso.")
                        else:
                            log("AVISO", f"Frame {indice:04d}: retificação falhou.")
                    else:
                        log("INFO", f"Frame {indice:04d}: sem bolas nem robô — a ignorar.")

                    conn.send("LIBERADO")

                    # ── Estatísticas ───────────────────────
                    stats["frames"]       += 1
                    stats["bolas_total"]  += len(bolas)
                    ms_total = (time.time() - t_recv) * 1000
                    stats["latencia_soma"] += ms_total

                    if stats["frames"] % 10 == 0:
                        media_lat = stats["latencia_soma"] / stats["frames"]
                        log("INFO",
                            f"─── Resumo {stats['frames']} frames | "
                            f"bolas={stats['bolas_total']} | "
                            f"robô detetado={stats['robo_detetado']}x | "
                            f"latência média={media_lat:.0f}ms ───")

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