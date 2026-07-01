"""
aruco_processor.py - Servico dedicado de deteccao ArUco UFSC/FEUP
=================================================================
Recebe frames diretamente do imageStreaming numa porta propria, deteta
os marcadores do robo e envia a pose para o retificador. Este processo
nao carrega YOLO e nao partilha fila com a inferencia de bolas.

Fluxo:
  imageStreaming -> [6003] -> ArUcoProcessor -> [6001] -> retificador
  imageStreaming <- [LIBERADO] <-------------------------------

Marcadores ArUco:
  ID 0 -> Frente do robo  (DICT_4X4_50)
  ID 1 -> Traseira do robo (DICT_4X4_50)
"""

import cv2
import numpy as np
import time
import sys
import socket
import threading
import queue
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiprocessing.connection import Listener, Client

from _COMMON.logging_utils import log as _log
import _CONFIG.system_parameters as _params

MOD = "ARUCO"


def log(nivel: str, msg: str):
    _log(MOD, nivel, msg)


PORTA_ENTRADA = 6003
PORTA_HEALTH = 6004
PORTA_RET = 6001
AUTHKEY_ARUCO = b"aruco_ufsc"
AUTHKEY_RET = b"retificador_ufsc"

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ID_FRONTAL = 0
ID_TRASEIRO = 1

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
ARUCO_LARGURA_PX = 1280
ARUCO_USAR_CLAHE = False
ARUCO_PERSISTENCIA_S = 0.35
ARUCO_SUAVIZACAO = 0.35
ARUCO_FILA_MAX = 1
ARUCO_ROBUSTO_INTERVALO = 6
ARUCO_ROI_MARGEM_PX = 180
ARUCO_ROI_FALHAS_MAX = 6
MODO_LOCALIZACAO_ROBO = "ARUCO"

# Círculos planos: azul à frente e vermelho atrás. Os intervalos HSV são
# deliberadamente conservadores para rejeitar o piso cinzento. O vermelho
# ocupa as duas extremidades do eixo H do OpenCV (0..179).
COR_MIN_AREA_PX = 90.0
COR_MIN_CIRCULARIDADE = 0.52
COR_MAX_AREA_FRACAO = 0.04
COR_RATIO_AREA_MIN = 0.35
COR_DISTANCIA_MIN_PX = 12.0
COR_DISTANCIA_MAX_FRACAO_DIAGONAL = 0.60
HSV_VERMELHO_1 = (np.array([0, 105, 65]), np.array([12, 255, 255]))
HSV_VERMELHO_2 = (np.array([168, 105, 65]), np.array([179, 255, 255]))
HSV_AZUL = (np.array([90, 85, 55]), np.array([138, 255, 255]))


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

    threading.Thread(target=_serve, daemon=True, name="health-aruco").start()
    log("DEBUG", f"Health-check ativo na porta {porta}")


def criar_detetor_aruco():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.005
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.04
    parameters.minCornerDistanceRate = 0.03
    parameters.minDistanceToBorder = 2
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.05
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.errorCorrectionRate = 0.7

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    log("HUMANO", f"Detetor ArUco pronto (IDs {ID_FRONTAL}=frente, {ID_TRASEIRO}=tras).")
    return detector


class PersistenciaAruco:
    """Mantem a ultima pose ArUco por alguns frames para evitar flicker."""

    def __init__(self, persistencia_s: float, suavizacao: float):
        self.persistencia_s = max(0.0, float(persistencia_s))
        self.suavizacao = min(max(float(suavizacao), 0.0), 0.95)
        self._ultimo = {"frontal": None, "traseiro": None}
        self._t_ultimo = {"frontal": 0.0, "traseiro": 0.0}
        self._orientacao = None

    def atualizar(self, det: dict, agora: float | None = None) -> dict:
        agora = time.time() if agora is None else agora
        frescos = {
            "frontal": det.get("frontal") is not None,
            "traseiro": det.get("traseiro") is not None,
        }
        qualidade_det = dict(det.get("qualidade_localizacao") or {})
        saida = {
            "frontal": None,
            "traseiro": None,
            "orientacao_graus": None,
            "qualidade_localizacao": qualidade_det,
        }

        for chave in ("frontal", "traseiro"):
            pos = det.get(chave)
            if pos:
                anterior = self._ultimo.get(chave)
                if anterior:
                    a = self.suavizacao
                    pos = {**pos,
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

        idades_ms = [
            max(0.0, (agora - self._t_ultimo[chave]) * 1000.0)
            for chave in ("frontal", "traseiro")
            if self._ultimo.get(chave)
        ]
        ambos_frescos = frescos["frontal"] and frescos["traseiro"]
        fonte = str(qualidade_det.get("fonte", MODO_LOCALIZACAO_ROBO)).upper()
        valida_detetor = bool(qualidade_det.get("valida_controle", ambos_frescos))
        saida["qualidade_localizacao"].update({
            "fonte": fonte,
            "deteccao_fresca": ambos_frescos,
            "valida_controle": ambos_frescos and valida_detetor,
            "frontal_fresco": frescos["frontal"],
            "traseiro_fresco": frescos["traseiro"],
            "persistida": not ambos_frescos,
            "idade_ms": round(max(idades_ms), 1) if idades_ms else None,
        })

        return saida


def detetar_robo(frame_gray, detector, clahe=None, escala_saida: float = 1.0,
                 robusto: bool = False) -> dict:
    resultado = {
        "frontal": None,
        "traseiro": None,
        "orientacao_graus": None,
        "qualidade_localizacao": {
            "fonte": "ARUCO",
            "deteccao_fresca": False,
            "valida_controle": False,
            "confianca": 0.0,
        },
    }

    candidatos = [frame_gray]
    if robusto and clahe is not None:
        candidatos.append(clahe.apply(frame_gray))
    if robusto:
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
            resultado["frontal"] = {"cx": round(cx, 1), "cy": round(cy, 1)}
        elif marker_id == ID_TRASEIRO:
            resultado["traseiro"] = {"cx": round(cx, 1), "cy": round(cy, 1)}

    if resultado["frontal"] and resultado["traseiro"]:
        dx = resultado["frontal"]["cx"] - resultado["traseiro"]["cx"]
        dy = resultado["frontal"]["cy"] - resultado["traseiro"]["cy"]
        resultado["orientacao_graus"] = round(float(np.degrees(np.arctan2(-dy, dx))), 2)

    n_det = int(resultado["frontal"] is not None) + int(resultado["traseiro"] is not None)
    resultado["qualidade_localizacao"].update({
        "deteccao_fresca": n_det == 2,
        "valida_controle": n_det == 2,
        "confianca": round(n_det / 2.0, 3),
        "marcadores_detetados": n_det,
    })

    return resultado


def _mascara_cor(hsv, nome: str):
    if nome == "vermelho":
        m1 = cv2.inRange(hsv, HSV_VERMELHO_1[0], HSV_VERMELHO_1[1])
        m2 = cv2.inRange(hsv, HSV_VERMELHO_2[0], HSV_VERMELHO_2[1])
        mascara = cv2.bitwise_or(m1, m2)
    else:
        mascara = cv2.inRange(hsv, HSV_AZUL[0], HSV_AZUL[1])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)


def _candidatos_circulares(mascara, nome: str, area_frame: float) -> list[dict]:
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    area_max = max(COR_MIN_AREA_PX, area_frame * COR_MAX_AREA_FRACAO)
    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        if area < COR_MIN_AREA_PX or area > area_max:
            continue
        perimetro = float(cv2.arcLength(contorno, True))
        if perimetro <= 1e-6:
            continue
        circularidade = float(4.0 * np.pi * area / (perimetro * perimetro))
        if circularidade < COR_MIN_CIRCULARIDADE:
            continue
        momentos = cv2.moments(contorno)
        if abs(momentos["m00"]) <= 1e-6:
            continue
        cx = float(momentos["m10"] / momentos["m00"])
        cy = float(momentos["m01"] / momentos["m00"])
        (_, _), raio = cv2.minEnclosingCircle(contorno)
        preenchimento = min(1.0, area / max(np.pi * raio * raio, 1.0))
        confianca = min(1.0, 0.65 * circularidade + 0.35 * preenchimento)
        candidatos.append({
            "nome": nome,
            "cx": cx,
            "cy": cy,
            "area_px": round(area, 1),
            "circularidade": round(circularidade, 3),
            "raio_px": round(float(raio), 2),
            "confianca": round(float(confianca), 3),
        })
    return sorted(candidatos, key=lambda c: (c["confianca"], c["area_px"]), reverse=True)[:8]


def _dist_px(a: dict, b: dict) -> float:
    return float(np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))


def _selecionar_par_cor(vermelhos: list[dict], azuis: list[dict],
                        w: int, h: int, anterior: dict | None = None):
    diagonal = float(np.hypot(w, h))
    distancia_max = diagonal * COR_DISTANCIA_MAX_FRACAO_DIAGONAL
    melhor = None
    melhor_score = -1e9
    ant_f = (anterior or {}).get("frontal")
    ant_t = (anterior or {}).get("traseiro")

    for azul in azuis:
        for vermelho in vermelhos:
            distancia = _dist_px(vermelho, azul)
            if not (COR_DISTANCIA_MIN_PX <= distancia <= distancia_max):
                continue
            ratio_area = min(vermelho["area_px"], azul["area_px"]) / max(
                vermelho["area_px"], azul["area_px"])
            if ratio_area < COR_RATIO_AREA_MIN:
                continue
            score = (
                vermelho["confianca"] + azul["confianca"]
                + 0.8 * ratio_area
            )
            if ant_f and ant_t:
                deslocamento = _dist_px(azul, ant_f) + _dist_px(vermelho, ant_t)
                score -= min(2.0, deslocamento / max(diagonal * 0.20, 1.0))
            if score > melhor_score:
                melhor_score = score
                melhor = (azul, vermelho, ratio_area, distancia)
    return melhor


def detetar_robo_cor(frame_bgr, anterior: dict | None = None) -> dict:
    """Deteta círculo azul frontal e vermelho traseiro num frame BGR."""
    resultado = {
        "frontal": None,
        "traseiro": None,
        "orientacao_graus": None,
        "qualidade_localizacao": {
            "fonte": "COR",
            "deteccao_fresca": False,
            "valida_controle": False,
            "confianca": 0.0,
        },
    }
    if frame_bgr is None or len(frame_bgr.shape) != 3:
        return resultado

    suavizado = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(suavizado, cv2.COLOR_BGR2HSV)
    h, w = frame_bgr.shape[:2]
    area_frame = float(w * h)
    vermelhos = _candidatos_circulares(
        _mascara_cor(hsv, "vermelho"), "vermelho", area_frame)
    azuis = _candidatos_circulares(
        _mascara_cor(hsv, "azul"), "azul", area_frame)
    par = _selecionar_par_cor(vermelhos, azuis, w, h, anterior)

    if par is None:
        # Mantém deteções parciais apenas para diagnóstico/persistência visual.
        if azuis:
            resultado["frontal"] = dict(azuis[0])
        if vermelhos:
            resultado["traseiro"] = dict(vermelhos[0])
        resultado["qualidade_localizacao"].update({
            "candidatos_vermelhos": len(vermelhos),
            "candidatos_azuis": len(azuis),
        })
        return resultado

    azul, vermelho, ratio_area, distancia = par
    resultado["frontal"] = dict(azul)
    resultado["traseiro"] = dict(vermelho)
    dx = azul["cx"] - vermelho["cx"]
    dy = azul["cy"] - vermelho["cy"]
    resultado["orientacao_graus"] = round(
        float(np.degrees(np.arctan2(-dy, dx))), 2)
    confianca = min(
        azul["confianca"], vermelho["confianca"],
        0.5 + 0.5 * ratio_area,
    )
    resultado["qualidade_localizacao"].update({
        "deteccao_fresca": True,
        "valida_controle": True,
        "confianca": round(float(confianca), 3),
        "distancia_centros_px": round(float(distancia), 2),
        "ratio_areas": round(float(ratio_area), 3),
        "frontal": {
            "area_px": azul["area_px"],
            "circularidade": azul["circularidade"],
            "raio_px": azul["raio_px"],
        },
        "traseiro": {
            "area_px": vermelho["area_px"],
            "circularidade": vermelho["circularidade"],
            "raio_px": vermelho["raio_px"],
        },
        "candidatos_vermelhos": len(vermelhos),
        "candidatos_azuis": len(azuis),
    })
    return resultado


def _roi_a_partir_robo(robo: dict, w: int, h: int, margem: int) -> tuple[int, int, int, int] | None:
    pontos = [
        robo.get(chave)
        for chave in ("frontal", "traseiro")
        if robo.get(chave)
    ]
    if not pontos:
        return None

    xs = [float(p["cx"]) for p in pontos]
    ys = [float(p["cy"]) for p in pontos]
    x0 = max(0, int(min(xs) - margem))
    y0 = max(0, int(min(ys) - margem))
    x1 = min(w, int(max(xs) + margem))
    y1 = min(h, int(max(ys) + margem))
    if (x1 - x0) < 80 or (y1 - y0) < 80:
        return None
    return x0, y0, x1, y1


def _somar_offset_robo(robo: dict, dx: int, dy: int) -> dict:
    for chave in ("frontal", "traseiro"):
        if robo.get(chave):
            robo[chave]["cx"] = round(robo[chave]["cx"] + dx, 1)
            robo[chave]["cy"] = round(robo[chave]["cy"] + dy, 1)
    return robo


def _escalar_robo(robo: dict, escala: float) -> dict:
    if escala == 1.0:
        return robo
    for chave in ("frontal", "traseiro"):
        if robo.get(chave):
            robo[chave]["cx"] = round(robo[chave]["cx"] * escala, 1)
            robo[chave]["cy"] = round(robo[chave]["cy"] * escala, 1)
    return robo


def _enfileirar_mais_recente(fila: queue.Queue, pacote: dict) -> bool:
    try:
        fila.put_nowait(pacote)
        return False
    except queue.Full:
        substituiu = False
        try:
            fila.get_nowait()
            fila.task_done()
            substituiu = True
        except queue.Empty:
            pass
        try:
            fila.put_nowait(pacote)
            return substituiu
        except queue.Full:
            return substituiu


def _worker_processamento_aruco(fila_frames: queue.Queue, detector, clahe,
                                persistencia: PersistenciaAruco,
                                stats: dict):
    conn_ret = None
    ultimo_robo_proc = {"frontal": None, "traseiro": None, "orientacao_graus": None}
    falhas_roi = 0

    def enviar_para_retificador_persistente(pacote_ret: dict) -> bool:
        nonlocal conn_ret
        for tentativa in range(2):
            try:
                if conn_ret is None:
                    conn_ret = Client(("localhost", PORTA_RET), authkey=AUTHKEY_RET)
                    log("DEBUG", "Ligação persistente ao retificador ativa.")
                conn_ret.send(pacote_ret)
                return conn_ret.recv() == "LIBERADO"
            except ConnectionRefusedError:
                if tentativa == 1:
                    log("ERRO", "Retificador inacessivel.")
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                log("DEBUG", "Ligação ao retificador caiu. A reabrir...")
            except Exception as e:
                log("DEBUG", f"Erro ao contactar retificador: {e}")
                break

            try:
                if conn_ret is not None:
                    conn_ret.close()
            except Exception:
                pass
            conn_ret = None

        return False

    while True:
        pacote = fila_frames.get()
        try:
            indice = int(pacote.get("indice", stats["frames"]))
            t_recv = time.time()
            frame = pacote["frame"]
            escala_origem_x = float(pacote.get("escala_origem_x", 1.0))
            escala_origem_y = float(pacote.get("escala_origem_y", 1.0))

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

            if MODO_LOCALIZACAO_ROBO == "COR":
                if len(frame_aruco.shape) == 2:
                    # Salvaguarda para clientes antigos; não inventa cor a
                    # partir do grayscale, apenas produz pose inválida.
                    frame_cor = None
                    h_det, w_det = frame_aruco.shape[:2]
                else:
                    frame_cor = frame_aruco
                    h_det, w_det = frame_cor.shape[:2]
                roi = _roi_a_partir_robo(
                    ultimo_robo_proc, w_det, h_det, ARUCO_ROI_MARGEM_PX)
                if roi is not None and frame_cor is not None:
                    x0, y0, x1, y1 = roi
                    robo_det = detetar_robo_cor(frame_cor[y0:y1, x0:x1])
                    robo_det = _somar_offset_robo(robo_det, x0, y0)
                    # Em modo COR uma falha da ROI provoca busca global já no
                    # mesmo frame, em vez de esperar várias falhas.
                    if not robo_det["qualidade_localizacao"].get("valida_controle"):
                        robo_det = detetar_robo_cor(frame_cor, ultimo_robo_proc)
                        falhas_roi += 1
                elif frame_cor is not None:
                    robo_det = detetar_robo_cor(frame_cor, ultimo_robo_proc)
                else:
                    robo_det = detetar_robo_cor(None)
            else:
                if pacote.get("frame_gray") or len(frame_aruco.shape) == 2:
                    gray = frame_aruco
                else:
                    gray = cv2.cvtColor(frame_aruco, cv2.COLOR_BGR2GRAY)
                robusto = (stats["frames"] % ARUCO_ROBUSTO_INTERVALO) == 0
                h_gray, w_gray = gray.shape[:2]
                roi = None if robusto or falhas_roi >= ARUCO_ROI_FALHAS_MAX else _roi_a_partir_robo(
                    ultimo_robo_proc, w_gray, h_gray, ARUCO_ROI_MARGEM_PX
                )
                if roi is not None:
                    x0, y0, x1, y1 = roi
                    robo_det = detetar_robo(
                        gray[y0:y1, x0:x1], detector, clahe,
                        escala_saida=1.0,
                        robusto=False,
                    )
                    robo_det = _somar_offset_robo(robo_det, x0, y0)
                else:
                    robo_det = detetar_robo(
                        gray, detector, clahe,
                        escala_saida=1.0,
                        robusto=robusto,
                    )

            if robo_det["frontal"] or robo_det["traseiro"]:
                ultimo_robo_proc = {
                    "frontal": robo_det.get("frontal") or ultimo_robo_proc.get("frontal"),
                    "traseiro": robo_det.get("traseiro") or ultimo_robo_proc.get("traseiro"),
                    "orientacao_graus": robo_det.get("orientacao_graus"),
                }
                falhas_roi = 0
            elif roi is not None:
                falhas_roi += 1

            robo_det = _escalar_robo(robo_det, escala_aruco)
            robo = persistencia.atualizar(robo_det)
            for chave in ("frontal", "traseiro"):
                if robo.get(chave):
                    robo[chave]["cx"] = round(robo[chave]["cx"] * escala_origem_x, 1)
                    robo[chave]["cy"] = round(robo[chave]["cy"] * escala_origem_y, 1)

            ok = enviar_para_retificador_persistente({
                "frame": None,
                "bolas_px": [],
                "robo_px": robo,
                "indice": indice,
                "timestamp_visao": pacote["timestamp"],
                "tipo": "aruco",
            })
            if ok and (robo["frontal"] or robo["traseiro"]):
                stats["detetados"] += 1
            elif not ok:
                log("DEBUG", f"Frame {indice:04d}: envio da localização falhou.")

            stats["frames"] += 1
            stats["latencia_soma"] += (time.time() - t_recv) * 1000
            if stats["frames"] % 100 == 0:
                media = stats["latencia_soma"] / stats["frames"]
                log("HUMANO", f"{stats['frames']} frames de localização processados "
                              f"(latencia media {media:.0f}ms, "
                              f"drops={stats['drops']}).")
        except Exception as e:
            stats["erros"] += 1
            log("ERRO", f"Erro ao processar frame ArUco: {e}")
        finally:
            fila_frames.task_done()


def iniciar_aruco():
    global ARUCO_LARGURA_PX, ARUCO_USAR_CLAHE, ARUCO_PERSISTENCIA_S, ARUCO_SUAVIZACAO
    global MODO_LOCALIZACAO_ROBO

    cfg = _params.carregar()
    MODO_LOCALIZACAO_ROBO = str(
        cfg.get("modo_localizacao_robo", "ARUCO")
    ).upper()
    if MODO_LOCALIZACAO_ROBO not in ("ARUCO", "COR"):
        log("AVISO", f"modo_localizacao_robo={MODO_LOCALIZACAO_ROBO!r} inválido; a usar ARUCO.")
        MODO_LOCALIZACAO_ROBO = "ARUCO"
    ARUCO_LARGURA_PX = int(cfg.get("aruco_largura_px", ARUCO_LARGURA_PX))
    ARUCO_USAR_CLAHE = bool(int(cfg.get("aruco_usar_clahe", 0)))
    ARUCO_PERSISTENCIA_S = float(cfg.get("aruco_persistencia_s", ARUCO_PERSISTENCIA_S))
    ARUCO_SUAVIZACAO = float(cfg.get("aruco_suavizacao", ARUCO_SUAVIZACAO))

    detector = criar_detetor_aruco()
    clahe = (cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
             if ARUCO_USAR_CLAHE else None)
    persistencia = PersistenciaAruco(ARUCO_PERSISTENCIA_S, ARUCO_SUAVIZACAO)

    iniciar_health_server()

    stats = {
        "frames": 0,
        "detetados": 0,
        "erros": 0,
        "drops": 0,
        "recebidos": 0,
        "latencia_soma": 0.0,
    }
    fila_frames: queue.Queue = queue.Queue(maxsize=ARUCO_FILA_MAX)
    threading.Thread(
        target=_worker_processamento_aruco,
        args=(fila_frames, detector, clahe, persistencia, stats),
        daemon=True,
        name="aruco-processamento",
    ).start()
    log("HUMANO", f"Localização do robô pronta em modo {MODO_LOCALIZACAO_ROBO}. A aguardar frames...")
    log("DEBUG", f"servidor ativo na porta {PORTA_ENTRADA}")
    log("DEBUG", f"Localização={MODO_LOCALIZACAO_ROBO} | largura={ARUCO_LARGURA_PX}px | CLAHE={'ON' if ARUCO_USAR_CLAHE else 'OFF'} | "
                 f"persistencia={ARUCO_PERSISTENCIA_S:.2f}s | suavizacao={ARUCO_SUAVIZACAO:.2f}")

    with Listener(("localhost", PORTA_ENTRADA), authkey=AUTHKEY_ARUCO) as listener:
        while True:
            try:
                conn = listener.accept()
            except ConnectionAbortedError:
                log("AVISO", "Ligacao rejeitada (sem autenticacao) - a ignorar.")
                continue
            except Exception as e:
                log("ERRO", f"Erro ao aceitar ligacao: {e}")
                continue

            with conn:
                while True:
                    try:
                        pacote = conn.recv()
                        stats["recebidos"] += 1
                        if _enfileirar_mais_recente(fila_frames, pacote):
                            stats["drops"] += 1
                        conn.send("LIBERADO")
                    except (EOFError, ConnectionResetError, BrokenPipeError):
                        break
                    except Exception as e:
                        stats["erros"] += 1
                        log("ERRO", f"Erro ao receber frame ArUco: {e}")
                        try:
                            conn.send("LIBERADO")
                        except Exception:
                            break


if __name__ == "__main__":
    iniciar_aruco()
