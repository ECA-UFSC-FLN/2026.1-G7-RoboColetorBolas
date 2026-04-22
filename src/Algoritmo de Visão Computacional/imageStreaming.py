"""
imageStreaming.py — Captura e Envio de Frames UFSC/FEUP
========================================================
Dois modos de operação detectados automaticamente:

  CALIBRAÇÃO  (VisionProcessing offline, retificador na 6001)
    → Preview em tempo real; tecla C captura e envia UM frame para calibração.

  PRODUÇÃO    (VisionProcessing online na 6000)
    → Loop automático: captura contínua e envia cada frame sem intervenção.
      Tecla P pausa/retoma | Tecla E encerra.

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

# Suprime warnings internos do OpenCV (VIDEOIO, obsensor, etc.)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
try:
    # Redireciona stderr a nível de SO para suprimir mensagens do C++ do OpenCV
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

# Intervalo mínimo entre capturas em produção (segundos).
INTERVALO_PRODUCAO = 5.0

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
    print(f"{cor}[{ts}] [STREAMING   ] {icon} {msg}\033[0m", flush=True)


#  ENVIO PARA VISÃO (produção)
# ─────────────────────────────────────────────
def enviar_para_visao(frame, dispositivo: str) -> bool:
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
            return None   # sinal para terminar o loop
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
    """
    Envia frame para retificador em modo calibração (porta 6001).
    Devolve True se bem-sucedido.
    """
    try:
        with Client(("localhost", PORTA_RET), authkey=AUTHKEY_RET) as conn:
            log("INFO", f"Enviando frame de calibração para porta {PORTA_RET}...")
            conn.send({"frame": frame})
            log("OK", "Frame de calibração enviado.")
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
            f"FPS: {stats.get('fps', 0.0):.1f}")
    cv2.putText(frame, info,
                (w // 2, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 180), 1)

    # Countdown até próxima captura
    espera = stats.get("espera_restante", 0.0)
    if modo == "PRODUCAO" and not pausado and espera > 0:
        countdown_txt = f"Próxima captura em: {espera:.1f}s"
        cv2.putText(frame, countdown_txt,
                    (w // 2 - 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)

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
                    log("OK", f"Câmera encontrada: índice={idx} backend={nome}")
                    return c
                c.release()
        return None

    log("FASE", "Procurando câmera disponível (índices 0–4, todos os backends)...")
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
    log("OK", f"Câmera ativa: {w_real}×{h_real}px")

    # Modo recebido via argumento --modo (definido pelo MasterControl)
    modo = "CALIBRACAO"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--modo" and i < len(sys.argv):
            modo = sys.argv[i + 1].upper()
            break

    log("FASE", f"Modo: {modo}")
    if modo == "PRODUCAO":
        log("INFO", "Captura automática contínua. Tecla P → pausa/retoma | Tecla E → encerrar")
    else:
        log("INFO", "Modo calibração — prima C para capturar o frame de referência.")

    cv2.namedWindow("Monitor de Captura", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Monitor de Captura", min(w_real, 1280), min(h_real, 720))

    stats   = {"enviados": 0, "erros": 0, "fps": 0.0, "espera_restante": 0.0}
    pausado = False
    t_fps   = time.time()
    frames_fps = 0
    t_proximo_envio = 0.0   # timestamp a partir do qual é permitido enviar

    while True:
        ret, frame = cap.read()
        if not ret:
            log("AVISO", "Frame inválido — câmera desligada?")
            time.sleep(0.1)
            continue

        # Cálculo de FPS
        frames_fps += 1
        dt = time.time() - t_fps
        if dt >= 1.0:
            stats["fps"] = frames_fps / dt
            frames_fps   = 0
            t_fps        = time.time()

        # Preview
        preview = desenhar_overlay(frame.copy(), stats, modo, pausado)
        cv2.imshow("Monitor de Captura", preview)

        tecla = cv2.waitKey(1) & 0xFF

        # ── Encerrar ──────────────────────────────────────
        if tecla in (ord("e"), ord("E"), 27):
            log("FASE", "A encerrar streaming...")
            break

        # ── Info ──────────────────────────────────────────
        if tecla in (ord("i"), ord("I")):
            log("INFO", f"Modo={modo} | Câmera={w_real}×{h_real}px | "
                        f"Enviados={stats['enviados']} | Erros={stats['erros']} | "
                        f"FPS={stats['fps']:.1f}")

        # ══════════════════════════════════════════════════
        #  MODO CALIBRAÇÃO — tecla C captura e envia
        # ══════════════════════════════════════════════════
        if modo == "CALIBRACAO":
            if tecla in (ord("c"), ord("C")):
                log("FASE", "A capturar frame de calibração...")
                ok = enviar_para_calibracao(frame.copy())
                if ok:
                    stats["enviados"] += 1
                    log("OK", "Frame enviado. O retificador abrirá a janela de marcação.")
                    log("INFO", "Este processo pode encerrar — o retificador toma conta do resto.")
                    break   # imageStreaming já não é necessário após enviar
                else:
                    stats["erros"] += 1

        # ══════════════════════════════════════════════════
        #  MODO PRODUÇÃO — loop automático
        # ══════════════════════════════════════════════════
        else:
            if tecla in (ord("p"), ord("P")):
                pausado = not pausado
                log("INFO", f"{'Pausado' if pausado else 'Retomado'}.")

            if not pausado:
                agora = time.time()
                espera = t_proximo_envio - agora
                stats["espera_restante"] = max(0.0, espera)

                if agora >= t_proximo_envio:
                    resultado = enviar_para_visao(frame.copy(), "cam_principal")

                    if resultado is None:
                        log("AVISO", "VisionProcessing indisponível. A encerrar.")
                        break
                    elif resultado:
                        stats["enviados"] += 1
                        t_proximo_envio = time.time() + INTERVALO_PRODUCAO
                        log("INFO", f"Próxima captura em {INTERVALO_PRODUCAO:.0f}s.")
                    else:
                        stats["erros"] += 1

    cap.release()
    cv2.destroyAllWindows()
    log("OK", f"Streaming encerrado. "
              f"Enviados={stats['enviados']} | Erros={stats['erros']}")

# ─────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    stream()