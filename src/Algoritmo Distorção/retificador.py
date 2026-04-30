"""
Modos de Operação:

  - calibrar    Recebe um frame via socket (porta 6001), permite ao utilizador marcar pontos de referência com coordenadas reais conhecidas,
                calcula a homografia e guarda o JSON de calibração.

  (sem args)   Servidor de produção. Recebe pacotes do VisionProcessing com bounding boxes em píxeis, aplica a homografia e guarda
               as coordenadas em metros em ficheiros JSON.

Parâmetros intrínsecos: iPhone 16 (5712×4284), f≈20587 / 20591)

NOTA: Uma thread separada fica à escuta na porta 6011 (PORTA_HEALTH) exclusivamente para responder a health-checks do MasterControl,
      evitando colisões com o Listener de autenticação na porta 6001.

"""

import cv2
import numpy as np
import json
import sys
import argparse
import time
import socket
import threading
import queue
from pathlib import Path
from multiprocessing.connection import Listener
from datetime import datetime

from bolas_log import log as _log

MOD = "RETIFICADOR"

def log(nivel: str, msg: str):
    """Atalho local: encapsula bolas_log.log com o módulo fixo."""
    _log(MOD, nivel, msg)

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
BASE_PATH         = Path(__file__).parent.resolve()
PASTA_SAIDA       = BASE_PATH / "resultados"
PASTA_POSICOES    = PASTA_SAIDA / "posicoes"
PASTA_IMAGENS     = PASTA_SAIDA / "imagens"            # frame original (anotado)
PASTA_IMAGENS_TD  = PASTA_SAIDA / "imagens_topdown"    # frame corrigido (vista de cima)
PASTA_CALIB_REF   = PASTA_SAIDA / "calibracao"   # pontos + imagem de cada calibração
CALIB_FILE       = PASTA_CALIB_REF / "homografia_calibracao.json"
PORTA        = 6001
PORTA_HEALTH = 6011
PORTA_GRAFO  = 6020                              # NOVO — fila para o GraphProcessor
AUTHKEY      = b"retificador_ufsc"
AUTHKEY_GRAFO = b"grafo_ufsc"                    # NOVO

# Parâmetros intrínsecos iPhone 16 câmera principal 26mm, 5712×4284
K_CAM = np.array([[20587,     0, 2856],
                   [    0, 20591, 2142],
                   [    0,     0,    1]], dtype=np.float64)
D_CAM = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# ─────────────────────────────────────────────
#  FILA PARA O GRAPH PROCESSOR
# ─────────────────────────────────────────────
# Fila em memória de JSONs prontos. O servidor da porta 6020 consome desta fila.
# Limite alto para não rebentar memória mas não bloquear o produtor em rajadas.
_fila_grafo: queue.Queue = queue.Queue(maxsize=500)


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
    log("DEBUG", f"Health-check ativo na porta {porta}")


# ─────────────────────────────────────────────
#  SERVIDOR DA FILA PARA O GRAPHPROCESSOR (NOVO)
# ─────────────────────────────────────────────
def iniciar_servidor_grafo(porta: int = PORTA_GRAFO):
    """
    Aceita ligações persistentes do GraphProcessor.
    Protocolo:
      cliente envia: {"acao": "pedir_proximo"}
      servidor responde com o próximo JSON disponível na fila (bloqueia até haver)

    Suporta apenas um cliente de cada vez (o GraphProcessor é único).
    Se um segundo cliente se ligar, fica em espera até o primeiro fechar.
    """
    def _serve():
        try:
            listener = Listener(("localhost", porta), authkey=AUTHKEY_GRAFO)
        except OSError as e:
            log("ERRO", f"Não foi possível abrir porta {porta}: {e}")
            return

        log("HUMANO", f"Servidor de fila para GraphProcessor ativo na porta {porta}")

        while True:
            try:
                conn = listener.accept()
            except Exception as e:
                log("AVISO", f"accept() falhou no servidor de fila: {e}")
                continue

            log("DEBUG", "GraphProcessor ligou-se à fila.")
            try:
                while True:
                    msg = conn.recv()
                    if not isinstance(msg, dict):
                        conn.send({"erro": "formato_invalido"})
                        continue
                    if msg.get("acao") != "pedir_proximo":
                        conn.send({"erro": "acao_desconhecida"})
                        continue

                    # bloqueia até haver pacote
                    pacote = _fila_grafo.get()
                    conn.send(pacote)
            except (EOFError, ConnectionResetError):
                log("AVISO", "GraphProcessor desligou-se da fila.")
            except Exception as e:
                log("ERRO", f"Erro no servidor de fila: {e}")
            finally:
                try: conn.close()
                except Exception: pass

    threading.Thread(target=_serve, daemon=True).start()


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
# Cache dos mapas de undistort (calculados uma vez por resolução).
_UNDISTORT_MAPS: dict = {}

def _obter_maps_undistort(w: int, h: int):
    chave = (w, h)
    if chave not in _UNDISTORT_MAPS:
        map1, map2 = cv2.initUndistortRectifyMap(
            K_CAM, D_CAM, None, K_CAM, (w, h), cv2.CV_16SC2)
        _UNDISTORT_MAPS[chave] = (map1, map2)
        log("DEBUG", f"Mapas de undistort pré-calculados para {w}×{h}px.")
    return _UNDISTORT_MAPS[chave]

def aplicar_undistort(img):
    h, w = img.shape[:2]
    map1, map2 = _obter_maps_undistort(w, h)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

def undistort_ponto(cx: float, cy: float):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    pt_corr = cv2.undistortPoints(pt, K_CAM, D_CAM, None, K_CAM)
    return float(pt_corr[0][0][0]), float(pt_corr[0][0][1])

def aplicar_topdown(img, H, out_w: int, out_h: int):
    img_undist = aplicar_undistort(img)
    return cv2.warpPerspective(
        img_undist, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# ═════════════════════════════════════════════════════════════════════
#  MODO CALIBRAÇÃO  (inalterado em relação à versão anterior)
# ═════════════════════════════════════════════════════════════════════
def calibrar_via_socket():
    iniciar_health_server()

    log("HUMANO", "Servidor de calibração ativo na porta 6001")
    log("HUMANO", "Aguardando frame do imageStreaming... (captura com tecla C)")
    address = ("localhost", PORTA)
    with Listener(address, authkey=AUTHKEY) as listener:
        with listener.accept() as conn:
            pacote = conn.recv()
            img    = pacote["frame"]
    log("HUMANO", "Frame recebido. A preparar janela de calibração...")

    img_undist = aplicar_undistort(img)
    h_img, w_img = img_undist.shape[:2]

    print()
    log("HUMANO", "Quantos pontos de referência vai marcar? (mínimo 4, recomendado 6+)")
    while True:
        try:
            n = int(input("  >>> "))
            if n < 4:
                log("AVISO", "Mínimo 4 pontos. Insere um valor ≥ 4.")
            else:
                break
        except ValueError:
            log("AVISO", "Entrada inválida. Insere um número inteiro (ex: 6).")

    pts_px   = []
    JANELA   = "CALIBRACAO — Marque os pontos"

    def redesenhar_pontos():
        base = img_undist.copy()
        restam = n - len(pts_px)
        if restam > 0:
            header = (f"Marque {n} pontos | marcados: {len(pts_px)}/{n}  "
                      f"|  D: apagar último  |  ESC: cancelar")
        else:
            header = "Todos os pontos marcados! Prima ENTER para continuar."
        cv2.putText(base, header,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 0) if restam == 0 else (0, 255, 255), 2)
        for idx0, (px, py) in enumerate(pts_px):
            cv2.circle(base, (px, py), 10, (0, 0, 255), -1)
            cv2.circle(base, (px, py), 12, (255, 255, 255), 2)
            cv2.putText(base, str(idx0 + 1),
                        (px + 15, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(base, f"({px},{py})",
                        (px + 15, py + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
        cv2.imshow(JANELA, base)
        return base

    def on_clique(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_px) < n:
            pts_px.append((x, y))
            log("DEBUG", f"Ponto {len(pts_px)}/{n} marcado em px=({x}, {y})")
            redesenhar_pontos()

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JANELA, min(w_img, 1400), min(h_img, 900))
    redesenhar_pontos()
    cv2.setMouseCallback(JANELA, on_clique)

    log("HUMANO", f"Janela aberta. Marque os {n} pontos.")
    log("HUMANO", "  Clique esquerdo — adicionar ponto")
    log("HUMANO", "  Tecla D         — apagar último ponto")
    log("HUMANO", "  ENTER           — confirmar (quando todos marcados)")
    log("HUMANO", "  ESC             — cancelar")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(pts_px) == n:
            break
        if key in (ord("d"), ord("D")) and pts_px:
            removido = pts_px.pop()
            log("AVISO", f"Ponto {len(pts_px) + 1} removido (px={removido}). "
                         f"Restam {len(pts_px)}/{n}.")
            redesenhar_pontos()
        if key == 27:
            log("AVISO", "Calibração cancelada pelo utilizador (ESC).")
            cv2.destroyAllWindows()
            sys.exit(1)

    img_draw = redesenhar_pontos()
    cv2.putText(img_draw,
                "Consulta os numeros enquanto inserires as coordenadas no terminal.",
                (20, h_img - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.setWindowTitle(JANELA, "CALIBRACAO — Referência (consulta no terminal)")
    cv2.imshow(JANELA, img_draw)
    cv2.resizeWindow(JANELA, min(w_img, 1400), min(h_img, 900))
    cv2.waitKey(1)

    print()
    log("HUMANO", "Inserção de coordenadas reais (metros)")
    log("HUMANO", "A janela com os pontos numerados está ABERTA para consulta.")
    log("HUMANO", "Alterna entre o terminal e a janela conforme precisares.")
    print()

    pts_reais = [None] * n

    def pedir_coordenada(i):
        px, py = pts_px[i]
        print(f"  \033[93m--- Ponto {i+1} (px={px}, py={py}) ---\033[0m")
        while True:
            try:
                cv2.waitKey(1)
                xr = float(input(f"    X real (metros): "))
                yr = float(input(f"    Y real (metros): "))
                conf = input(
                    f"    \033[92m→ ({xr:.3f}m, {yr:.3f}m)\033[0m"
                    f"  ENTER confirmar  |  d apagar: "
                ).strip().lower()
                if conf == "d":
                    log("AVISO", f"Ponto {i+1} apagado. A reintroduzir...")
                    print()
                    continue
                log("DEBUG", f"Ponto {i+1}: real=({xr:.3f}m, {yr:.3f}m)")
                print()
                return [xr, yr]
            except ValueError:
                log("AVISO", "Valor inválido. Insere um número (ex: 1.50)")

    for i in range(n):
        pts_reais[i] = pedir_coordenada(i)

    while True:
        cv2.waitKey(1)
        print()
        print("  \033[96m┌─────────────────────────────────────────────────┐\033[0m")
        print("  \033[96m│            RESUMO DE COORDENADAS                │\033[0m")
        print("  \033[96m├──────┬──────────────┬──────────────────────────┤\033[0m")
        print("  \033[96m│ Pto  │   Pixel (px) │   Real (metros)          │\033[0m")
        print("  \033[96m├──────┼──────────────┼──────────────────────────┤\033[0m")
        for i in range(n):
            px_x, px_y = pts_px[i]
            xr, yr = pts_reais[i]
            print(f"  \033[96m│\033[0m  {i+1:2d}  \033[96m│\033[0m"
                  f" ({px_x:4d},{px_y:4d})  \033[96m│\033[0m"
                  f"  X={xr:7.3f}m   Y={yr:7.3f}m       \033[96m│\033[0m")
        print("  \033[96m└──────┴──────────────┴──────────────────────────┘\033[0m")
        print()
        resp = input("  \033[93m>> Corrigir algum ponto? (número 1–{} ou ENTER para continuar): \033[0m"
                     .format(n)).strip()
        if resp == "":
            break
        try:
            idx_corr = int(resp) - 1
            if 0 <= idx_corr < n:
                log("HUMANO", f"A reescrever coordenadas do ponto {idx_corr + 1}...")
                pts_reais[idx_corr] = pedir_coordenada(idx_corr)
            else:
                log("AVISO", f"Número fora do intervalo (1–{n}). Tenta novamente.")
        except ValueError:
            log("AVISO", "Entrada inválida. Insere o número do ponto ou ENTER.")

    cv2.destroyAllWindows()

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
    # Aplicar undistort nos pontos de origem ANTES de calcular a homografia
    pts_origem_undist = np.array(
        [undistort_ponto(px, py) for (px, py) in pts_px],
        dtype=np.float32
    )

    if n == 4:
        H, mask = cv2.findHomography(pts_origem_undist, pts_destino_px)
        inliers  = 4
    else:
        H, mask = cv2.findHomography(pts_origem_undist, pts_destino_px,
                                  cv2.RANSAC, ransacReprojThreshold=5.0)
        inliers  = int(mask.sum()) if mask is not None else 0

    if H is None:
        log("ERRO", "Não foi possível calcular a homografia. Pontos colineares?")
        sys.exit(1)

    erros = []
    for i in range(n):
        # pts_origem_undist já calculado no passo anterior
        dst_pred = cv2.perspectiveTransform(
            np.array([[[pts_origem_undist[i][0], pts_origem_undist[i][1]]]], dtype=np.float32), H
        )
        dx = float(dst_pred[0][0][0]) - float(pts_destino_px[i][0])
        dy = float(dst_pred[0][0][1]) - float(pts_destino_px[i][1])
        erros.append(np.sqrt(dx**2 + dy**2))

    erro_medio_px = float(np.mean(erros))
    erro_medio_m  = float(erro_medio_px / ppm)
    log("HUMANO", f"Homografia calculada: {inliers}/{n} inliers | "
                  f"erro médio = {erro_medio_m*100:.1f}cm")
    log("DEBUG",  f"erro_medio_px={erro_medio_px:.1f}px  ppm={ppm:.2f}")

    if erro_medio_m > 0.05:
        log("AVISO", f"Erro elevado ({erro_medio_m*100:.1f}cm > 5cm). "
                     "Considera recalibrar com mais pontos.")

    PASTA_CALIB_REF.mkdir(parents=True, exist_ok=True)

    out_w_px = int(round(W_real * ppm))
    out_h_px = int(round(D_real * ppm))

    calib = numpy_para_python({
        "H_mat":           H,
        "ppm":             ppm,
        "x_min":           x_min,
        "y_min":           y_min,
        "W_real_m":        round(float(W_real), 4),
        "D_real_m":        round(float(D_real), 4),
        "output_size_px":  [out_w_px, out_h_px],
        "n_pontos":        n,
        "inliers":         inliers,
        "erro_medio_px":   round(erro_medio_px, 3),
        "erro_medio_m":    round(erro_medio_m,  5),
        "resolucao_calib": [w_img, h_img],
        "data":            datetime.now().isoformat(timespec="seconds"),
    })

    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=4)
    log("HUMANO", f"Calibração guardada: {CALIB_FILE.name}")

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
    log("DEBUG", f"Imagem de referência guardada: {IMG_REF_PATH.name}")

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pontos_calib = {
        "data":       datetime.now().isoformat(timespec="seconds"),
        "n_pontos":   n,
        "ppm":        round(ppm, 3),
        "erro_medio_m": round(erro_medio_m, 5),
        "inliers":    inliers,
        "pontos": [
            {
                "numero":  i + 1,
                "px_x":    pts_px[i][0],
                "px_y":    pts_px[i][1],
                "real_x_m": round(pts_reais[i][0], 4),
                "real_y_m": round(pts_reais[i][1], 4),
            }
            for i in range(n)
        ],
    }
    json_calib_path = PASTA_CALIB_REF / f"pontos_{ts_str}.json"
    with open(json_calib_path, "w") as f:
        json.dump(pontos_calib, f, indent=4, ensure_ascii=False)
    log("DEBUG", f"Registo de pontos guardado: resultados/calibracao/{json_calib_path.name}")

    img_calib_path = PASTA_CALIB_REF / f"imagem_{ts_str}.png"
    ok_enc, buf = cv2.imencode(".png", img_ref)
    if ok_enc:
        img_calib_path.write_bytes(buf.tobytes())
        log("DEBUG", f"Imagem de calibração guardada: resultados/calibracao/{img_calib_path.name}")
    else:
        log("AVISO", "Não foi possível guardar a imagem na pasta calibracao.")

    try:
        img_topdown = cv2.warpPerspective(
            img_ref, H, (out_w_px, out_h_px),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        cv2.putText(img_topdown,
                    f"TOP-DOWN PREVIEW | {W_real:.2f}m × {D_real:.2f}m | ppm={ppm:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        topdown_path = PASTA_CALIB_REF / f"topdown_{ts_str}.png"
        ok_enc, buf = cv2.imencode(".png", img_topdown)
        if ok_enc:
            topdown_path.write_bytes(buf.tobytes())
            log("DEBUG", f"Preview top-down guardado: resultados/calibracao/{topdown_path.name}")
        else:
            log("AVISO", "Não foi possível codificar o preview top-down.")
    except Exception as e:
        log("AVISO", f"Falha ao gerar preview top-down: {e}")

    log("HUMANO", f"Quadra calibrada: {W_real:.2f}m × {D_real:.2f}m")
    log("DEBUG",  f"ppm={ppm:.1f} | saída top-down={out_w_px}×{out_h_px}px")
    sys.exit(0)


# ═════════════════════════════════════════════════════════════════════
#  MODO PRODUÇÃO
# ═════════════════════════════════════════════════════════════════════
def _px_para_metros(cx: float, cy: float,
                    H, ppm: float,
                    x_min: float, y_min: float):
    ux, uy  = undistort_ponto(cx, cy)
    pt_warp = cv2.perspectiveTransform(
        np.array([[[ux, uy]]], dtype=np.float32), H)
    x_metros = float(pt_warp[0][0][0]) / ppm + x_min
    y_metros = float(pt_warp[0][0][1]) / ppm + y_min
    return round(x_metros, 4), round(y_metros, 4)


def servidor_producao(calib: dict):
    H     = np.array(calib["H_mat"])
    ppm   = calib["ppm"]
    x_min = calib.get("x_min", 0.0)
    y_min = calib.get("y_min", 0.0)

    if "output_size_px" in calib and calib["output_size_px"]:
        out_w_px, out_h_px = calib["output_size_px"]
    elif "W_real_m" in calib and "D_real_m" in calib:
        out_w_px = int(round(float(calib["W_real_m"]) * ppm))
        out_h_px = int(round(float(calib["D_real_m"]) * ppm))
    else:
        res = calib.get("resolucao_calib", [1920, 1080])
        out_w_px, out_h_px = int(res[0]), int(res[1])
        log("AVISO", "Calibração antiga sem 'output_size_px' — a usar fallback "
                     f"de {out_w_px}×{out_h_px}px. Recalibra para mais precisão.")
    out_w_px = max(out_w_px, 1)
    out_h_px = max(out_h_px, 1)

    PASTA_POSICOES.mkdir(parents=True, exist_ok=True)
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    PASTA_IMAGENS_TD.mkdir(parents=True, exist_ok=True)
    log("DEBUG", f"Pastas prontas: {PASTA_POSICOES.name} | {PASTA_IMAGENS.name} | "
              f"{PASTA_IMAGENS_TD.name}")
    log("DEBUG", f"Vista top-down: {out_w_px}×{out_h_px}px @ {ppm:.1f}ppm "
                f"({out_w_px/ppm:.2f}m × {out_h_px/ppm:.2f}m)")

    # NOVO: arrancar o servidor de fila para o GraphProcessor
    iniciar_servidor_grafo()

    log("HUMANO", "Retificador pronto. A aguardar pacotes do VisionProcessing...")
    log("DEBUG",  f"Calibração: ppm={ppm:.1f} | erro médio={calib.get('erro_medio_m','?')}m")
    log("DEBUG",  f"Posições  → .../{PASTA_POSICOES.relative_to(BASE_PATH)}")
    log("DEBUG",  f"Imagens   → .../{PASTA_IMAGENS.relative_to(BASE_PATH)}")
    log("DEBUG",  f"Top-down  → .../{PASTA_IMAGENS_TD.relative_to(BASE_PATH)}")

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
                    robo_px  = pacote.get("robo_px", {})
                    frame    = pacote["frame"]

                    # ── Retificar bolas ────────────────────────────────────
                    res_bolas = []
                    for b in bolas_px:
                        cx = (b["x1"] + b["x2"]) / 2.0
                        cy = (b["y1"] + b["y2"]) / 2.0
                        xm, ym = _px_para_metros(cx, cy, H, ppm, x_min, y_min)
                        res_bolas.append({"x": xm, "y": ym})

                    # ── Retificar robô (ArUco) ─────────────────────────────
                    res_robo = {
                        "frontal":          None,
                        "traseiro":         None,
                        "orientacao_graus": robo_px.get("orientacao_graus"),
                    }
                    if robo_px.get("frontal"):
                        xm, ym = _px_para_metros(
                            robo_px["frontal"]["cx"], robo_px["frontal"]["cy"],
                            H, ppm, x_min, y_min)
                        res_robo["frontal"] = {"x": xm, "y": ym}
                    if robo_px.get("traseiro"):
                        xm, ym = _px_para_metros(
                            robo_px["traseiro"]["cx"], robo_px["traseiro"]["cy"],
                            H, ppm, x_min, y_min)
                        res_robo["traseiro"] = {"x": xm, "y": ym}

                    robo_log = "—"
                    if res_robo["frontal"] or res_robo["traseiro"]:
                        partes = []
                        if res_robo["frontal"]:
                            f = res_robo["frontal"]
                            partes.append(f"F({f['x']:.2f},{f['y']:.2f})")
                        if res_robo["traseiro"]:
                            t = res_robo["traseiro"]
                            partes.append(f"T({t['x']:.2f},{t['y']:.2f})")
                        if res_robo["orientacao_graus"] is not None:
                            partes.append(f"{res_robo['orientacao_graus']:.1f}°")
                        robo_log = " ".join(partes)

                    latencia = round(
                        (time.time() - pacote["timestamp_visao"]) * 1000, 2)
                    saida = {
                        "indice":      indice,
                        "latencia_ms": latencia,
                        "n_bolas":     len(res_bolas),
                        "trajetoria":  res_bolas,
                        "robo":        res_robo,
                    }

                    saida_serializavel = numpy_para_python(saida)

                    # ── Guardar posições (Opção B preservada) ──────────────
                    fich_json = PASTA_POSICOES / f"posicao_{indice:04d}.json"
                    with open(fich_json, "w") as f:
                        json.dump(saida_serializavel, f, indent=4)

                    # ── NOVO: empurrar para fila do GraphProcessor (Opção A) ─
                    try:
                        _fila_grafo.put_nowait(saida_serializavel)
                    except queue.Full:
                        # Drop do mais antigo para não estagnar
                        try:
                            _fila_grafo.get_nowait()
                            _fila_grafo.put_nowait(saida_serializavel)
                            log("AVISO", "Fila do grafo cheia — descartado pacote mais antigo.")
                        except Exception:
                            pass

                    # ── Guardar imagem ─────────────────────────────────────
                    fich_img = PASTA_IMAGENS / f"frame_{indice:04d}.jpg"
                    if frame is not None and hasattr(frame, "shape"):
                        try:
                            ok_enc, buf = cv2.imencode(
                                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            if ok_enc:
                                fich_img.write_bytes(buf.tobytes())
                                log("DEBUG", f"Imagem guardada: {fich_img.name}")
                            else:
                                log("ERRO", f"cv2.imencode falhou para {fich_img.name} "
                                            f"(shape={frame.shape}, dtype={frame.dtype})")
                        except Exception as e_img:
                            log("ERRO", f"Erro ao guardar imagem {fich_img.name}: {e_img}")

                        fich_img_td = PASTA_IMAGENS_TD / f"frame_{indice:04d}.jpg"
                        try:
                            frame_td = aplicar_topdown(frame, H, out_w_px, out_h_px)
                            ok_td, buf_td = cv2.imencode(
                                ".jpg", frame_td, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            if ok_td:
                                fich_img_td.write_bytes(buf_td.tobytes())
                                log("DEBUG", f"Top-down guardado: {fich_img_td.name}")
                            else:
                                log("ERRO", f"cv2.imencode falhou para {fich_img_td.name}")
                        except Exception as e_td:
                            log("ERRO", f"Erro ao gerar/guardar top-down "
                                        f"{fich_img_td.name}: {e_td}")
                    else:
                        log("AVISO", f"Frame {indice:04d}: frame ausente ou inválido — imagem não guardada.")

                    conn.send("LIBERADO")

                    total_frames += 1
                    total_bolas  += len(res_bolas)
                    # Spam de cada frame só para debug; o utilizador vê a evolução
                    # no GraphProcessor (janela ao vivo) — não precisa nesta consola.
                    log("DEBUG",
                        f"Frame {indice:04d} | {len(res_bolas)} bola(s) | "
                        f"robô={robo_log} | latência={latencia}ms | total={total_frames} frames")
                    # Mas a cada 50 frames damos um pulso humano para confirmar vida
                    if total_frames % 50 == 0:
                        log("HUMANO",
                            f"{total_frames} frames processados "
                            f"({total_bolas} bolas no total).")

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
            log("HUMANO", "Executa com --calibrar primeiro, ou usa o MasterControl.py")
            sys.exit(1)

        iniciar_health_server()

        try:
            with open(CALIB_FILE) as f:
                calib = json.load(f)
        except json.JSONDecodeError as e:
            log("ERRO", f"Ficheiro de calibração corrompido: {e}")
            log("AVISO", "Apaga o ficheiro e recalibra (MasterControl → opção 's').")
            time.sleep(60)
            sys.exit(1)

        log("HUMANO", f"Calibração de {calib.get('data','data desconhecida')} carregada.")
        log("DEBUG",  f"ppm={calib['ppm']:.1f} | {calib.get('n_pontos','?')} pontos | "
                     f"erro_medio_m={calib.get('erro_medio_m','?')}")
        servidor_producao(calib)
