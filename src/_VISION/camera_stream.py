"""
imageStreaming.py — Captura e Envio de Frames UFSC/FEUP
========================================================
Câmara suportada: Camo Studio (Reincubate) via USB, exposta como
dispositivo DirectShow virtual. O Camo Studio deve estar ativo e a
câmara física (iPhone) ligada via USB antes de iniciar.

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
import subprocess
import threading
from queue import Queue, Empty, Full
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from multiprocessing.connection import Client
import _CONFIG.system_parameters as _params

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
# Nome parcial da câmara preferida (case-insensitive).
# O Camo Studio instala um dispositivo DirectShow chamado "Camo" ou
# "Camo Virtual Webcam" — basta deixar "Camo" para o detetar.
NOME_CAMERA_PREFERIDO = "Camo"

# Backends a testar (por ordem de preferência)
BACKENDS = [
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_MSMF,  "MSMF"),
    (cv2.CAP_ANY,   "ANY"),
]

PORTA_VIS       = 6000
PORTA_RET       = 6001
AUTHKEY_VIS     = b"bolas_ufsc"
AUTHKEY_RET     = b"retificador_ufsc"
MAX_TENTATIVAS  = 3

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
from _COMMON.logging_utils import log as _log

MOD = "STREAMING"

def log(nivel: str, msg: str):
    _log(MOD, nivel, msg)


# ─────────────────────────────────────────────
#  DETECÇÃO DE CÂMARA POR NOME (Windows)
# ─────────────────────────────────────────────
def listar_cameras_windows() -> list[tuple[int, str]]:
    """
    Usa PowerShell para listar os dispositivos de câmara instalados e
    devolve uma lista de (índice_ordem, nome_amigável).
    O índice corresponde tipicamente à ordem de enumeração DirectShow.
    Devolve lista vazia em caso de erro ou em sistemas não-Windows.
    """
    if sys.platform != "win32":
        return []
    try:
        resultado = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-PnpDevice -Class Camera -Status OK "
                "| Sort-Object FriendlyName "
                "| Select-Object -ExpandProperty FriendlyName",
            ],
            capture_output=True,
            text=True,
            timeout=6,
        )
        nomes = [
            linha.strip()
            for linha in resultado.stdout.splitlines()
            if linha.strip()
        ]
        return list(enumerate(nomes))
    except Exception as e:
        log("DEBUG", f"listar_cameras_windows falhou: {e}")
        return []


def indices_preferidos() -> list[int]:
    """
    Devolve os índices a testar, colocando o Camo à frente se detectado.
    Fallback: testa índices 0–4 sem preferência.
    """
    cameras = listar_cameras_windows()
    if not cameras:
        log("DEBUG", "Enumeração de câmaras indisponível — a testar índices 0–4.")
        return list(range(5))

    log("DEBUG", f"Câmaras detectadas: {[n for _, n in cameras]}")

    preferenciais = [i for i, n in cameras
                     if NOME_CAMERA_PREFERIDO.lower() in n.lower()]
    outros        = [i for i, _ in cameras
                     if i not in preferenciais]
    ordem = preferenciais + outros + [i for i in range(5) if i not in preferenciais + outros]
    log("DEBUG", f"Ordem de tentativa de índices: {ordem[:5]}")
    return ordem[:5]


# ─────────────────────────────────────────────
#  ENVIO PARA VISÃO (produção)
# ─────────────────────────────────────────────
def enviar_para_visao(pacote: dict):
    """
    Envia frame para VisionProcessing (porta 6000) e aguarda LIBERADO.
    Devolve True se bem-sucedido, False em erro recuperável, None para terminar.
    """
    for tentativa in range(MAX_TENTATIVAS):
        try:
            with Client(("localhost", PORTA_VIS), authkey=AUTHKEY_VIS) as conn:
                conn.send(pacote)
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


def preparar_frame_producao(frame, largura_alvo: int) -> tuple:
    h, w = frame.shape[:2]
    if largura_alvo <= 0 or largura_alvo >= w:
        return frame, 1.0, 1.0
    escala = largura_alvo / float(w)
    novo_h = max(1, int(round(h * escala)))
    pequeno = cv2.resize(frame, (largura_alvo, novo_h), interpolation=cv2.INTER_AREA)
    return pequeno, w / float(largura_alvo), h / float(novo_h)


def worker_envio(fila: Queue, stats: dict, parar: threading.Event):
    while not parar.is_set():
        try:
            pacote = fila.get(timeout=0.1)
        except Empty:
            continue

        resultado = enviar_para_visao(pacote)
        if resultado is None:
            stats["vision_offline"] = True
            parar.set()
        elif resultado:
            stats["enviados"] += 1
            stats["envios_janela"] += 1
        else:
            stats["erros"] += 1
        fila.task_done()


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
            f"Drop: {stats.get('descartados', 0)}  |  "
            f"FPS captura: {stats.get('fps', 0.0):.1f}  |  "
            f"FPS envio: {stats.get('fps_envio', 0.0):.2f}")
    cv2.putText(frame, info,
                (w // 2 - 100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 180), 1)

    return frame


# ─────────────────────────────────────────────
#  LOOP PRINCIPAL
# ─────────────────────────────────────────────
def stream():
    cfg = _params.carregar()
    largura_processamento = int(cfg.get("processamento_largura_px", 960))
    enviar_debug_original = bool(int(cfg.get("guardar_imagens_debug", 0)))
    intervalo_debug_original_s = float(cfg.get("intervalo_guardar_imagens_s", 5.0))

    def tentar_abrir_camera():
        """
        Tenta abrir a câmara preferida (Camo Studio por defeito) por nome,
        depois por índice, com múltiplos backends DirectShow/MSMF/ANY.
        Devolve o objeto VideoCapture ou None.
        """
        indices = indices_preferidos()
        for idx in indices:
            for backend, nome_backend in BACKENDS:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    log("DEBUG",
                        f"Câmera aberta: índice={idx} backend={nome_backend}")
                    return c
                c.release()
        return None

    log("DEBUG",
        f"Procurando câmera '{NOME_CAMERA_PREFERIDO}' "
        f"(Camo Studio USB) ou qualquer câmera disponível...")
    cap = tentar_abrir_camera()

    if cap is None:
        log("ERRO", "Nenhuma câmera encontrada.")
        input(
            "\033[93m>> Confirma que o Camo Studio está ativo, o iPhone ligado "
            "via USB e o Camo ativo no iPhone — depois prime ENTER para tentar "
            "novamente...\033[0m"
        )
        cap = tentar_abrir_camera()
        if cap is None:
            log("ERRO",
                "Câmera ainda indisponível. Verifica o Camo Studio e tenta novamente.")
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

    try:
        cv2.startWindowThread()
    except Exception:
        pass
    cv2.namedWindow("Monitor de Captura", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        cv2.setWindowProperty("Monitor de Captura", cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass
    cv2.resizeWindow("Monitor de Captura", min(w_real, 1280), min(h_real, 720))

    stats   = {
        "enviados": 0, "erros": 0, "fps": 0.0, "fps_envio": 0.0,
        "envios_janela": 0, "descartados": 0, "vision_offline": False,
    }
    pausado = False
    t_fps   = time.time()
    frames_fps = 0
    t_envios = time.time()
    proximo_debug_original = 0.0
    fila_envio: Queue = Queue(maxsize=1)
    parar_envio = threading.Event()
    if modo == "PRODUCAO":
        threading.Thread(
            target=worker_envio,
            args=(fila_envio, stats, parar_envio),
            daemon=True,
            name="envio-visao",
        ).start()
        log("DEBUG", f"Processamento: largura={largura_processamento}px "
                     f"(0 = resolução original).")

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
            stats["fps_envio"] = stats["envios_janela"] / dt_e
            stats["envios_janela"] = 0
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
                if stats.get("vision_offline"):
                    log("AVISO", "VisionProcessing indisponível. A encerrar.")
                    break
                if fila_envio.empty():
                    frame_proc, sx, sy = preparar_frame_producao(
                        frame, largura_processamento
                    )
                    agora_pacote = time.time()
                    pacote = {
                        "frame": frame_proc,
                        "timestamp": agora_pacote,
                        "nome": "cam_principal",
                        "escala_origem_x": sx,
                        "escala_origem_y": sy,
                        "resolucao_original": [w_real, h_real],
                    }
                    if enviar_debug_original and agora_pacote >= proximo_debug_original:
                        pacote["frame_debug_original"] = frame.copy()
                        proximo_debug_original = agora_pacote + intervalo_debug_original_s
                    try:
                        fila_envio.put_nowait(pacote)
                    except Full:
                        stats["descartados"] += 1
                else:
                    stats["descartados"] += 1

    cap.release()
    parar_envio.set()
    cv2.destroyAllWindows()
    log("HUMANO", f"Streaming encerrado. "
              f"Enviados={stats['enviados']} | Erros={stats['erros']}")


if __name__ == "__main__":
    stream()



