"""
imageStreaming.py — Captura e Envio de Frames UFSC/FEUP
========================================================
Dois modos de operação detectados automaticamente:

  CALIBRAÇÃO  (VisionProcessing offline, retificador na 6001)
    → Preview em tempo real; tecla C captura e envia UM frame para calibração.

  PRODUÇÃO    (VisionProcessing online na 6000)
    → Loop automático: captura contínua e envia cada frame.
      Tecla P pausa/retoma | Tecla E encerra.

      O ritmo é ditado pelo VisionProcessing — o cliente só envia o
      próximo frame quando o anterior já foi processado (handshake
      "LIBERADO"). Sem timers artificiais.

Teclas universais:
  E / ESC — Encerrar
  I       — Mostrar estado no terminal
"""

import cv2
import time
import sys
import os
import ctypes
from datetime import datetime
from multiprocessing.connection import Client

# Suprime warnings internos do OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
try:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetErrorMode(0x8007)
except Exception:
    pass

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
INDICE_CAMERA   = 1
BACKEND_CAMERA  = cv2.CAP_DSHOW
PORTA_VIS       = 6000
PORTA_RET       = 6001
AUTHKEY_VIS     = b"bolas_ufsc"
AUTHKEY_RET     = b"retificador_ufsc"
MAX_TENTATIVAS  = 3

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
from bolas_log import log as _log

MOD = "STREAMING"

def log(nivel: str, msg: str):
    """Atalho local: encapsula bolas_log.log com o módulo fixo."""
    _log(MOD, nivel, msg)


# ─────────────────────────────────────────────
#  ENVIO PARA VISÃO (produção)
# ─────────────────────────────────────────────
def enviar_para_visao(frame, dispositivo: str):
    """
    Envia frame para VisionProcessing (porta 6000) e aguarda LIBERADO.
    Devolve True se bem-sucedido, False em erro recuperável, None para terminar.
    """
    for tentativa in range(MAX_TENTATIVAS):
        try:
            with Client(("localhost", PORTA_VIS), authkey=AUTHKEY_VIS) as conn:
                conn.send({
                    "frame":     frame,
                    "timestamp": time.time(),
                    "nome":      dispositivo,
                })
                resposta = conn.recv()
                return resposta == "LIBERADO"
        except ConnectionRefusedError:
            log("ERRO", "VisionProcessing desligou. A encerrar loop de produção.")
            return None
        except EOFError:
            if tentativa < MAX_TENTATIVAS - 1:
                log("AVISO", f"Ligação interrompida (tentativa {tentativa+1}/{MAX_TENTATIVAS}). A repetir...")
                time.sleep(0.3)
        except Exception as e:
            log("ERRO", f"Erro ao enviar para visão: {e}")
            return False
    return False


# ─────────────────────────────────────────────
#  ENVIO PARA CALIBRAÇÃO
# ─────────────────────────────────────────────
def enviar_para_calibracao(frame) -> bool:
    try:
        with Client(("localhost", PORTA_RET), authkey=AUTHKEY_RET) as conn:
            log("DEBUG", f"Enviando frame de calibração para porta {PORTA_RET}...")
            conn.send({"frame": frame})
            log("HUMANO", "Frame de calibração enviado.")
            return True
    except ConnectionRefusedError:
        log("ERRO", "Retificador (calibração) não está disponível na porta 6001.")
        return False
    except Exception as e:
        log("ERRO", f"Erro ao enviar para calibração: {e}")
        return False


# ─────────────────────────────────────────────
#  OVERLAY
# ─────────────────────────────────────────────
def desenhar_overlay(frame, stats: dict, modo: str, pausado: bool):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    if modo == "CALIBRACAO":
        teclas = "C: Capturar frame de calibracao  |  E: Sair  |  I: Info"
        cor_modo = (0, 200, 255)
    else:
        teclas = "P: Pausa/Retoma  |  E: Sair  |  I: Info"
        cor_modo = (0, 255, 120)

    cv2.putText(frame, teclas,
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    estado_modo = f"MODO: {modo}"
    if modo == "PRODUCAO" and pausado:
        estado_modo += "  [PAUSADO]"
    cv2.putText(frame, estado_modo,
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_modo, 1)

    info = (f"Enviados: {stats['enviados']}  |  "
            f"Erros: {stats['erros']}  |  "
            f"FPS captura: {stats.get('fps', 0.0):.1f}  |  "
            f"FPS envio: {stats.get('fps_envio', 0.0):.2f}")
    cv2.putText(frame, info,
                (w // 2 - 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 180), 1)

    return frame


# ─────────────────────────────────────────────
#  LOOP PRINCIPAL
# ─────────────────────────────────────────────
def stream():
    BACKENDS = [
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_ANY,   "ANY"),
    ]

    def tentar_abrir_camera():
        for idx in range(5):
            for backend, nome in BACKENDS:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    log("DEBUG", f"Câmera encontrada: índice={idx} backend={nome}")
                    return c
                c.release()
        return None

    log("DEBUG", "Procurando câmera disponível (índices 0–4, todos os backends)...")
    cap = tentar_abrir_camera()

    if cap is None:
        log("ERRO", "Nenhuma câmera encontrada.")
        input("\033[93m>> Confirma que o Iriun Webcam está ativo e prime ENTER para tentar novamente...\033[0m")
        cap = tentar_abrir_camera()
        if cap is None:
            log("ERRO", "Câmera ainda indisponível. Encerra e tenta novamente.")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    w_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log("HUMANO", f"Câmara pronta ({w_real}×{h_real}px).")

    modo = "CALIBRACAO"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--modo" and i < len(sys.argv):
            modo = sys.argv[i + 1].upper()
            break

    log("DEBUG", f"Modo: {modo}")
    if modo == "PRODUCAO":
        log("HUMANO", "Captura em produção iniciada.")
        log("HUMANO", "Tecla P → pausa/retoma  |  Tecla E → encerrar")
    else:
        log("HUMANO", "Modo calibração — prima C na janela para capturar o frame.")

    cv2.namedWindow("Monitor de Captura", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Monitor de Captura", min(w_real, 1280), min(h_real, 720))

    stats   = {"enviados": 0, "erros": 0, "fps": 0.0, "fps_envio": 0.0}
    pausado = False
    t_fps   = time.time()
    frames_fps = 0
    t_envios = time.time()
    envios_janela = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            log("AVISO", "Frame inválido — câmera desligada?")
            time.sleep(0.1)
            continue

        # FPS de captura
        frames_fps += 1
        dt = time.time() - t_fps
        if dt >= 1.0:
            stats["fps"] = frames_fps / dt
            frames_fps   = 0
            t_fps        = time.time()

        # FPS de envio (rolling 5s)
        dt_e = time.time() - t_envios
        if dt_e >= 5.0:
            stats["fps_envio"] = envios_janela / dt_e
            envios_janela = 0
            t_envios = time.time()

        preview = desenhar_overlay(frame.copy(), stats, modo, pausado)
        cv2.imshow("Monitor de Captura", preview)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla in (ord("e"), ord("E"), 27):
            log("HUMANO", "A encerrar streaming...")
            break

        if tecla in (ord("i"), ord("I")):
            log("DEBUG", f"Modo={modo} | Câmera={w_real}×{h_real}px | "
                        f"Enviados={stats['enviados']} | Erros={stats['erros']} | "
                        f"FPS captura={stats['fps']:.1f} | FPS envio={stats['fps_envio']:.2f}")

        # ══════════════════════════════════════════════════
        #  MODO CALIBRAÇÃO — tecla C captura e envia
        # ══════════════════════════════════════════════════
        if modo == "CALIBRACAO":
            if tecla in (ord("c"), ord("C")):
                log("HUMANO", "A capturar frame de calibração...")
                ok = enviar_para_calibracao(frame.copy())
                if ok:
                    stats["enviados"] += 1
                    log("HUMANO", "Frame enviado. O retificador abrirá a janela de marcação.")
                    log("HUMANO", "Este processo pode encerrar — o retificador toma conta do resto.")
                    break
                else:
                    stats["erros"] += 1

        # ══════════════════════════════════════════════════
        #  MODO PRODUÇÃO — captura contínua, sem timer
        # ══════════════════════════════════════════════════
        else:
            if tecla in (ord("p"), ord("P")):
                pausado = not pausado
                log("HUMANO", f"{'Pausado' if pausado else 'Retomado'}.")

            if not pausado:
                resultado = enviar_para_visao(frame.copy(), "cam_principal")

                if resultado is None:
                    log("AVISO", "VisionProcessing indisponível. A encerrar.")
                    break
                elif resultado:
                    stats["enviados"] += 1
                    envios_janela += 1
                else:
                    stats["erros"] += 1

    cap.release()
    cv2.destroyAllWindows()
    log("HUMANO", f"Streaming encerrado. "
              f"Enviados={stats['enviados']} | Erros={stats['erros']}")


if __name__ == "__main__":
    stream()