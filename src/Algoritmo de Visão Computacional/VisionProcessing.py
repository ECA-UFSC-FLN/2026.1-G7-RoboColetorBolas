"""
VisionProcessing.py — Deteção de Bolas com YOLO UFSC/FEUP
===========================================================
Servidor na porta 6000. Recebe frames do imageStreaming,
executa inferência com o modelo YOLO personalizado,
e encaminha os resultados (bounding boxes) para o retificador (porta 6001).

Fluxo:
  imageStreaming → [porta 6000] → VisionProcessing → [porta 6001] → retificador
                                                                          ↓
  imageStreaming ← [LIBERADO] ←────────────────────────────────────────────

NOTA: Uma thread separada fica à escuta na porta 6002 (PORTA_HEALTH)
      exclusivamente para responder a health-checks do MasterControl,
      evitando colisões com o Listener autenticado na porta 6000.
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

PORTA_ENTRADA  = 6000          # recebe do imageStreaming (autenticado)
PORTA_HEALTH   = 6002          # health-check do MasterControl (TCP simples)
PORTA_RET      = 6001          # envia para o retificador
AUTHKEY_VIS    = b"bolas_ufsc"
AUTHKEY_RET    = b"retificador_ufsc"

CONFIANCA_MIN  = 0.50          # threshold de confiança YOLO
DISPOSITIVO    = 0             # 0=GPU CUDA, "cpu" para CPU

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
    """
    Socket TCP simples que aceita ligações e fecha-as imediatamente.
    Serve apenas para o MasterControl verificar que o processo está vivo.
    Corre numa daemon thread — termina automaticamente com o processo.
    """
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
#  ENVIO PARA RETIFICADOR (com retry)
# ─────────────────────────────────────────────
def enviar_para_retificador(pacote_ret: dict, tentativas: int = 3) -> bool:
    """
    Tenta enviar o pacote ao retificador com retry e backoff.
    Devolve True se a resposta LIBERADO foi recebida.
    """
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
    # ── Carregar modelo ────────────────────────────────────
    log("FASE", "Carregando modelo YOLO...")
    if not MODELO_PATH.exists():
        log("ERRO", f"Modelo não encontrado: {MODELO_PATH}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
        import numpy as np
        t0    = time.time()
        model = YOLO(str(MODELO_PATH))
        # Warm-up: inferência num frame preto para pré-compilar o modelo
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(source=dummy, conf=CONFIANCA_MIN,
                      device=DISPOSITIVO, verbose=False)
        t_load = time.time() - t0
        log("OK", f"Modelo carregado em {t_load:.1f}s | dispositivo={DISPOSITIVO}")
    except Exception as e:
        log("ERRO", f"Falha ao carregar YOLO: {e}")
        sys.exit(1)

    # ── Health-server sobe ANTES de abrir o Listener autenticado ──
    # O MasterControl monitoriza a porta 6002, nunca a 6000.
    iniciar_health_server()

    # ── Estatísticas de sessão ─────────────────────────────
    stats = {
        "frames":        0,
        "bolas_total":   0,
        "latencia_soma": 0.0,
        "erros":         0,
    }

    # ── Servidor ───────────────────────────────────────────
    log("FASE", f"Servidor ativo na porta {PORTA_ENTRADA}. Aguardando frames...")

    address = ("localhost", PORTA_ENTRADA)
    with Listener(address, authkey=AUTHKEY_VIS) as listener:
        while True:
            # ── Aceitar ligação com proteção contra health-checks ──────
            # Se algo ligar sem authkey (ex: scanner TCP) o handshake
            # aborta com ConnectionAbortedError — ignoramos e continuamos.
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

                    # ── Inferência YOLO ────────────────────
                    t_inf = time.time()
                    results = model.predict(
                        source=pacote["frame"],
                        conf=CONFIANCA_MIN,
                        device=DISPOSITIVO,
                        verbose=False,
                    )
                    ms_inf = (time.time() - t_inf) * 1000

                    bolas = []
                    for r in results:
                        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                        boxes_conf = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else []
                        for idx_b, box in enumerate(boxes_xyxy):
                            conf = float(boxes_conf[idx_b]) if len(boxes_conf) > idx_b else 0.0
                            bolas.append({
                                "x1": int(box[0]),
                                "y1": int(box[1]),
                                "x2": int(box[2]),
                                "y2": int(box[3]),
                                "conf": round(conf, 3),
                            })

                    log("INFO",
                        f"Frame {indice:04d} | {len(bolas)} bola(s) "
                        f"detetada(s) | inferência={ms_inf:.0f}ms")

                    # ── Anotar frame com bounding boxes ───
                    frame_anotado = pacote["frame"].copy()
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
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 0), 2)

                    # ── Encaminhar para retificador ────────
                    if bolas:
                        pacote_ret = {
                            "frame":           frame_anotado,
                            "bolas_px":        bolas,
                            "indice":          indice,
                            "timestamp_visao": pacote["timestamp"],
                        }
                        ok = enviar_para_retificador(pacote_ret)
                        if ok:
                            log("OK", f"Frame {indice:04d} retificado com sucesso.")
                        else:
                            log("AVISO",
                                f"Frame {indice:04d}: retificação falhou, "
                                "a libertar na mesma.")
                    else:
                        log("INFO", f"Frame {indice:04d}: nenhuma bola — a ignorar.")

                    # ── Liberta o capturador ───────────────
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