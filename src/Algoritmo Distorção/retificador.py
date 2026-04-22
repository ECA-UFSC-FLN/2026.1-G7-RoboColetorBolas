import cv2
import numpy as np
import json
import sys
import argparse
import time
import socket
import threading
from pathlib import Path
from multiprocessing.connection import Listener
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
BASE_PATH        = Path(__file__).parent.resolve()
PASTA_SAIDA      = BASE_PATH / "resultados"
PASTA_POSICOES   = PASTA_SAIDA / "posicoes"
PASTA_IMAGENS    = PASTA_SAIDA / "imagens"
CALIB_FILE       = BASE_PATH / "homografia_calibracao.json"
PORTA        = 6001
PORTA_HEALTH = 6011
AUTHKEY      = b"retificador_ufsc"

# Parâmetros intrínsecos iPhone 16 (landscape, 4032×3024)
K_CAM = np.array([[5823,    0, 2016],
                   [   0, 5823, 1512],
                   [   0,    0,    1]], dtype=np.float64)
D_CAM = np.array([0.122, -0.246, 0.0001, -0.0002, 0.176], dtype=np.float64)

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
    print(f"{cor}[{ts}] [RETIFICADOR ] {icon} {msg}\033[0m", flush=True)

# ─────────────────────────────────────────────
#  HEALTH-CHECK SERVER (porta 6011)
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
#  SERIALIZAÇÃO JSON SEGURA
# ─────────────────────────────────────────────
def numpy_para_python(obj):
    """
    Converte recursivamente tipos NumPy para tipos Python nativos,
    tornando qualquer estrutura segura para json.dump.
    float32/float64 → float  |  int32/int64 → int  |  ndarray → list
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: numpy_para_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_para_python(v) for v in obj]
    return obj

# ─────────────────────────────────────────────
#  FUNÇÕES DE CÂMERA
# ─────────────────────────────────────────────
def aplicar_undistort(img: np.ndarray) -> np.ndarray:
    return cv2.undistort(img, K_CAM, D_CAM)

def undistort_ponto(cx: float, cy: float) -> tuple[float, float]:
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    pt_corr = cv2.undistortPoints(pt, K_CAM, D_CAM, None, K_CAM)
    return float(pt_corr[0][0][0]), float(pt_corr[0][0][1])

# ─────────────────────────────────────────────
#  MODO CALIBRAÇÃO
# ─────────────────────────────────────────────
def calibrar_via_socket():
    """
    Fase de calibração interativa:
      1. Aguarda frame via socket (enviado pelo imageStreaming)
      2. Aplica undistort ao frame
      3. Utilizador marca N pontos na imagem
      4. Imagem fica VISÍVEL e aberta enquanto o utilizador insere coordenadas
      5. Calcula homografia, guarda JSON e imagem de referência anotada
    """
    iniciar_health_server()

    log("FASE", "Servidor de calibração ativo na porta 6001")
    log("INFO", "Aguardando frame do imageStreaming... (captura com tecla C)")

    address = ("localhost", PORTA)
    with Listener(address, authkey=AUTHKEY) as listener:
        with listener.accept() as conn:
            pacote = conn.recv()
            img    = pacote["frame"]
    log("OK", "Frame recebido. A preparar janela de calibração...")

    img_undist = aplicar_undistort(img)
    h_img, w_img = img_undist.shape[:2]

    # ── Número de pontos ────────────────────────────────────
    print()
    log("INFO", "Quantos pontos de referência vai marcar? (mínimo 4, recomendado 6+)")
    try:
        n = int(input("  >>> "))
        if n < 4:
            log("AVISO", "Mínimo 4 pontos. A usar 4.")
            n = 4
    except ValueError:
        log("AVISO", "Entrada inválida. A usar 4 pontos.")
        n = 4

    # ── Recolha de pontos na imagem ────────────────────────
    pts_px   = []
    img_draw = img_undist.copy()

    cv2.putText(img_draw,
                f"Marque {n} pontos (clique esquerdo). ENTER para confirmar.",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    def on_clique(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_px) < n:
            pts_px.append((x, y))
            idx = len(pts_px)
            cv2.circle(img_draw, (x, y), 10, (0, 0, 255), -1)
            cv2.circle(img_draw, (x, y), 12, (255, 255, 255), 2)
            cv2.putText(img_draw, str(idx),
                        (x + 15, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(img_draw, f"({x},{y})",
                        (x + 15, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
            cv2.imshow("CALIBRACAO — Marque os pontos", img_draw)
            log("INFO", f"Ponto {idx}/{n} marcado em px=({x}, {y})")
            if idx == n:
                cv2.putText(img_draw,
                            "Todos os pontos marcados! Prima ENTER para continuar.",
                            (20, h_img - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
                cv2.imshow("CALIBRACAO — Marque os pontos", img_draw)

    cv2.namedWindow("CALIBRACAO — Marque os pontos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CALIBRACAO — Marque os pontos", min(w_img, 1400), min(h_img, 900))
    cv2.imshow("CALIBRACAO — Marque os pontos", img_draw)
    cv2.setMouseCallback("CALIBRACAO — Marque os pontos", on_clique)

    log("INFO", f"Janela aberta. Marque os {n} pontos e prima ENTER na janela.")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(pts_px) == n:   # ENTER
            break
        if key == 27:
            log("AVISO", "Calibração cancelada pelo utilizador (ESC).")
            cv2.destroyAllWindows()
            sys.exit(1)

    # ── Atualiza a mesma janela para modo de consulta (sem abrir nova) ──
    cv2.putText(img_draw,
                "Consulta os numeros enquanto inserires as coordenadas no terminal.",
                (20, h_img - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.setWindowTitle("CALIBRACAO — Marque os pontos",
                       "CALIBRACAO — Referência (consulta no terminal)")
    cv2.imshow("CALIBRACAO — Marque os pontos", img_draw)
    cv2.resizeWindow("CALIBRACAO — Marque os pontos", min(w_img, 1400), min(h_img, 900))
    cv2.waitKey(1)

    # ── Recolha de coordenadas reais ───────────────────────
    print()
    log("FASE", "Inserção de coordenadas reais (metros)")
    log("INFO", "A janela com os pontos numerados está ABERTA para consulta.")
    log("INFO", "Alterna entre o terminal e a janela conforme precisares.")
    print()

    pts_reais = []
    for i, (px, py) in enumerate(pts_px):
        cv2.waitKey(1)
        print(f"  \033[93m--- Ponto {i+1} (px={px}, py={py}) ---\033[0m")
        while True:
            try:
                xr = float(input(f"    X real (metros): "))
                yr = float(input(f"    Y real (metros): "))
                pts_reais.append([xr, yr])
                log("OK", f"Ponto {i+1}: real=({xr:.3f}m, {yr:.3f}m)")
                break
            except ValueError:
                log("AVISO", "Valor inválido. Insere um número (ex: 1.50)")
        print()

    cv2.destroyAllWindows()

    # ── Cálculo da homografia ──────────────────────────────
    xs     = [p[0] for p in pts_reais]
    ys     = [p[1] for p in pts_reais]
    x_min  = min(xs);  y_min = min(ys)
    W_real = max(xs) - x_min
    D_real = max(ys) - y_min

    if max(W_real, D_real) > 0:
        ppm = max(w_img, h_img) / max(W_real, D_real)
    else:
        ppm = 200.0
        log("AVISO", "Extensão real nula — ppm forçado a 200. Verifica os pontos.")

    pts_destino_px = np.array(
        [((p[0] - x_min) * ppm, (p[1] - y_min) * ppm) for p in pts_reais],
        dtype=np.float32
    )
    pts_origem_px = np.array(pts_px, dtype=np.float32)

    if n == 4:
        H, mask = cv2.findHomography(pts_origem_px, pts_destino_px)
        inliers  = 4
    else:
        H, mask = cv2.findHomography(pts_origem_px, pts_destino_px,
                                      cv2.RANSAC, ransacReprojThreshold=2.0)
        inliers  = int(mask.sum()) if mask is not None else 0

    if H is None:
        log("ERRO", "Não foi possível calcular a homografia. Pontos colineares?")
        sys.exit(1)

    # ── Erro de reprojeção ─────────────────────────────────
    erros = []
    for i in range(n):
        ux, uy = undistort_ponto(pts_px[i][0], pts_px[i][1])
        dst_pred = cv2.perspectiveTransform(
            np.array([[[ux, uy]]], dtype=np.float32), H
        )
        dx = float(dst_pred[0][0][0]) - float(pts_destino_px[i][0])
        dy = float(dst_pred[0][0][1]) - float(pts_destino_px[i][1])
        erros.append(np.sqrt(dx**2 + dy**2))

    erro_medio_px = float(np.mean(erros))
    erro_medio_m  = float(erro_medio_px / ppm)
    log("INFO", f"Homografia: {inliers}/{n} inliers | "
                f"Erro médio: {erro_medio_px:.1f}px ({erro_medio_m*100:.1f}cm)")

    if erro_medio_m > 0.05:
        log("AVISO", f"Erro elevado ({erro_medio_m*100:.1f}cm > 5cm). "
                     "Considera recalibrar com mais pontos.")

    # ── Guardar calibração ─────────────────────────────────
    calib = numpy_para_python({
        "H_mat":           H,
        "ppm":             ppm,
        "x_min":           x_min,
        "y_min":           y_min,
        "n_pontos":        n,
        "inliers":         inliers,
        "erro_medio_px":   round(erro_medio_px, 3),
        "erro_medio_m":    round(erro_medio_m,  5),
        "resolucao_calib": [w_img, h_img],
        "data":            datetime.now().isoformat(timespec="seconds"),
    })

    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=4)
    log("OK", f"Calibração guardada: {CALIB_FILE.name}")

    # ── Guardar imagem de referência anotada ───────────────
    IMG_REF_PATH = BASE_PATH / "calibracao_referencia.png"
    img_ref = img_undist.copy()

    for i, (px, py) in enumerate(pts_px):
        xr, yr = pts_reais[i]
        cv2.circle(img_ref, (px, py), 14, (0, 0, 255), -1)
        cv2.circle(img_ref, (px, py), 16, (255, 255, 255), 2)
        cv2.putText(img_ref, str(i + 1),
                    (px + 18, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.putText(img_ref, f"X={xr:.2f}m  Y={yr:.2f}m",
                    (px + 18, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        cv2.line(img_ref, (px, py), (px + 15, py - 5), (255, 255, 255), 1)

    cv2.putText(img_ref,
                f"{n} pontos | ppm={ppm:.1f} | erro={erro_medio_m*100:.1f}cm | "
                f"{datetime.now().strftime('%H:%M:%S')}",
                (20, h_img - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)

    cv2.imwrite(str(IMG_REF_PATH), img_ref)
    log("OK", f"Imagem de referência guardada: {IMG_REF_PATH.name}")
    log("OK", f"  ppm={ppm:.1f} | área={W_real:.2f}×{D_real:.2f}m")
    sys.exit(0)

# ─────────────────────────────────────────────
#  MODO PRODUÇÃO
# ─────────────────────────────────────────────
def servidor_producao(calib: dict):
    """
    Servidor de retificação em loop contínuo.
    Recebe pacotes do VisionProcessing com bounding boxes em píxeis,
    converte para metros e guarda:
      - resultados/posicoes/posicao_NNNN.json  — coordenadas em metros
      - resultados/imagens/frame_NNNN.jpg      — frame original
    """
    H     = np.array(calib["H_mat"])
    ppm   = calib["ppm"]
    x_min = calib.get("x_min", 0.0)
    y_min = calib.get("y_min", 0.0)

    PASTA_POSICOES.mkdir(parents=True, exist_ok=True)
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    log("OK", f"Pastas prontas: {PASTA_POSICOES} | {PASTA_IMAGENS}")

    log("FASE", "Servidor de retificação ativo (porta 6001)")
    log("INFO", f"Calibração: ppm={ppm:.1f} | erro médio={calib.get('erro_medio_m','?')}m")
    log("INFO", f"Posições  → .../{PASTA_POSICOES.relative_to(BASE_PATH)}")
    log("INFO", f"Imagens   → .../{PASTA_IMAGENS.relative_to(BASE_PATH)}")
    log("INFO", "Aguardando pacotes do VisionProcessing...")

    total_frames = 0
    total_bolas  = 0

    address = ("localhost", PORTA)
    with Listener(address, authkey=AUTHKEY) as listener:
        while True:
            with listener.accept() as conn:
                try:
                    pacote   = conn.recv()
                    indice   = pacote["indice"]
                    bolas_px = pacote["bolas_px"]
                    frame    = pacote["frame"]

                    res_bolas = []
                    for b in bolas_px:
                        cx = (b["x1"] + b["x2"]) / 2.0
                        cy = (b["y1"] + b["y2"]) / 2.0

                        ux, uy = undistort_ponto(cx, cy)

                        pt_warp = cv2.perspectiveTransform(
                            np.array([[[ux, uy]]], dtype=np.float32), H
                        )
                        wx = float(pt_warp[0][0][0])
                        wy = float(pt_warp[0][0][1])

                        x_metros = wx / ppm + x_min
                        y_metros = wy / ppm + y_min

                        res_bolas.append({
                            "x": round(x_metros, 4),
                            "y": round(y_metros, 4),
                        })

                    latencia = round(
                        (time.time() - pacote["timestamp_visao"]) * 1000, 2
                    )
                    saida = {
                        "indice":      indice,
                        "latencia_ms": latencia,
                        "n_bolas":     len(res_bolas),
                        "trajetoria":  res_bolas,
                    }

                    # ── Guardar posições ───────────────────
                    fich_json = PASTA_POSICOES / f"posicao_{indice:04d}.json"
                    with open(fich_json, "w") as f:
                        json.dump(saida, f, indent=4)

                    # ── Guardar imagem ─────────────────────
                    # Usa imencode + escrita binária em vez de cv2.imwrite,
                    # evitando falhas com caminhos Unicode/acentuados no Windows
                    # (ex.: OneDrive com "4º Ano", espaços, etc.)
                    fich_img = PASTA_IMAGENS / f"frame_{indice:04d}.jpg"
                    if frame is not None and hasattr(frame, "shape"):
                        try:
                            ok_enc, buf = cv2.imencode(
                                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                            )
                            if ok_enc:
                                fich_img.write_bytes(buf.tobytes())
                                log("OK", f"Imagem guardada: {fich_img.name}")
                            else:
                                log("ERRO", f"cv2.imencode falhou para {fich_img.name} "
                                            f"(shape={frame.shape}, dtype={frame.dtype})")
                        except Exception as e_img:
                            log("ERRO", f"Erro ao guardar imagem {fich_img.name}: {e_img}")
                    else:
                        log("AVISO", f"Frame {indice:04d}: frame ausente ou inválido — imagem não guardada.")

                    conn.send("LIBERADO")

                    total_frames += 1
                    total_bolas  += len(res_bolas)
                    log("OK",
                        f"Frame {indice:04d} | {len(res_bolas)} bola(s) | "
                        f"latência={latencia}ms | total={total_frames} frames")

                except Exception as e:
                    log("ERRO", f"Erro ao processar pacote: {e}")
                    try:
                        conn.send("LIBERADO")
                    except Exception:
                        pass

# ─────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retificador de coordenadas UFSC/FEUP"
    )
    parser.add_argument("--calibrar", action="store_true",
                        help="Modo calibração: recebe frame e calcula homografia")
    args = parser.parse_args()

    if args.calibrar:
        calibrar_via_socket()
    else:
        if not CALIB_FILE.exists():
            log("ERRO", f"Ficheiro de calibração não encontrado: {CALIB_FILE}")
            log("INFO", "Executa com --calibrar primeiro, ou usa o MasterControl.py")
            sys.exit(1)

        # Health-server sobe ANTES de qualquer operação que possa falhar
        iniciar_health_server()

        try:
            with open(CALIB_FILE) as f:
                calib = json.load(f)
        except json.JSONDecodeError as e:
            log("ERRO", f"Ficheiro de calibração corrompido: {e}")
            log("AVISO", "Apaga o ficheiro e recalibra (MasterControl → opção 's').")
            time.sleep(60)
            sys.exit(1)

        log("OK", f"Calibração carregada: {calib.get('data','data desconhecida')} | "
                  f"ppm={calib['ppm']:.1f} | {calib.get('n_pontos','?')} pontos")
        servidor_producao(calib)

