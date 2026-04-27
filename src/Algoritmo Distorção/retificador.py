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
BASE_PATH         = Path(__file__).parent.resolve()
PASTA_SAIDA       = BASE_PATH / "resultados"
PASTA_POSICOES    = PASTA_SAIDA / "posicoes"
PASTA_IMAGENS     = PASTA_SAIDA / "imagens"            # frame original (anotado)
PASTA_IMAGENS_TD  = PASTA_SAIDA / "imagens_topdown"    # frame corrigido (vista de cima)
PASTA_CALIB_REF   = PASTA_SAIDA / "calibracao"   # pontos + imagem de cada calibração
CALIB_FILE       = PASTA_CALIB_REF / "homografia_calibracao.json"
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
# Cache dos mapas de undistort (calculados uma vez por resolução).
# Evita recomputar cv2.initUndistortRectifyMap em cada frame de produção.
_UNDISTORT_MAPS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

def _obter_maps_undistort(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (map1, map2) para cv2.remap, criando-os se ainda não existirem."""
    chave = (w, h)
    if chave not in _UNDISTORT_MAPS:
        map1, map2 = cv2.initUndistortRectifyMap(
            K_CAM, D_CAM, None, K_CAM, (w, h), cv2.CV_16SC2)
        _UNDISTORT_MAPS[chave] = (map1, map2)
        log("INFO", f"Mapas de undistort pré-calculados para {w}×{h}px.")
    return _UNDISTORT_MAPS[chave]

def aplicar_undistort(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    map1, map2 = _obter_maps_undistort(w, h)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

def undistort_ponto(cx: float, cy: float) -> tuple[float, float]:
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    pt_corr = cv2.undistortPoints(pt, K_CAM, D_CAM, None, K_CAM)
    return float(pt_corr[0][0][0]), float(pt_corr[0][0][1])

def aplicar_topdown(img: np.ndarray, H: np.ndarray,
                    out_w: int, out_h: int) -> np.ndarray:
    """
    Aplica undistort + warpPerspective para obter a vista de cima do plano de jogo.
    O tamanho de saída (out_w × out_h) deve corresponder a (W_real_m × ppm, D_real_m × ppm)
    para que cada píxel da imagem retificada seja um quadrado de 1/ppm metros.
    """
    img_undist = aplicar_undistort(img)
    return cv2.warpPerspective(
        img_undist, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

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
    while True:
        try:
            n = int(input("  >>> "))
            if n < 4:
                log("AVISO", "Mínimo 4 pontos. Insere um valor ≥ 4.")
            else:
                break
        except ValueError:
            log("AVISO", "Entrada inválida. Insere um número inteiro (ex: 6).")

    # ── Recolha de pontos na imagem ────────────────────────
    pts_px   = []
    JANELA   = "CALIBRACAO — Marque os pontos"

    def redesenhar_pontos():
        """Reconstrói img_draw do zero e desenha todos os pontos actuais."""
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
            log("INFO", f"Ponto {len(pts_px)}/{n} marcado em px=({x}, {y})")
            redesenhar_pontos()

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JANELA, min(w_img, 1400), min(h_img, 900))
    redesenhar_pontos()
    cv2.setMouseCallback(JANELA, on_clique)

    log("INFO", f"Janela aberta. Marque os {n} pontos.")
    log("INFO", "  Clique esquerdo — adicionar ponto")
    log("INFO", "  Tecla D         — apagar último ponto")
    log("INFO", "  ENTER           — confirmar (quando todos marcados)")
    log("INFO", "  ESC             — cancelar")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13 and len(pts_px) == n:   # ENTER
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

    # ── Atualiza a mesma janela para modo de consulta (sem abrir nova) ──
    img_draw = redesenhar_pontos()   # captura estado final como img_draw
    cv2.putText(img_draw,
                "Consulta os numeros enquanto inserires as coordenadas no terminal.",
                (20, h_img - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.setWindowTitle(JANELA, "CALIBRACAO — Referência (consulta no terminal)")
    cv2.imshow(JANELA, img_draw)
    cv2.resizeWindow(JANELA, min(w_img, 1400), min(h_img, 900))
    cv2.waitKey(1)

    # ── Recolha de coordenadas reais ───────────────────────
    print()
    log("FASE", "Inserção de coordenadas reais (metros)")
    log("INFO", "A janela com os pontos numerados está ABERTA para consulta.")
    log("INFO", "Alterna entre o terminal e a janela conforme precisares.")
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
                # ── Confirmação imediata ───────────────────
                conf = input(
                    f"    \033[92m→ ({xr:.3f}m, {yr:.3f}m)\033[0m"
                    f"  ENTER confirmar  |  d apagar: "
                ).strip().lower()
                if conf == "d":
                    log("AVISO", f"Ponto {i+1} apagado. A reintroduzir...")
                    print()
                    continue   # volta ao início do while — pede de novo
                log("OK", f"Ponto {i+1}: real=({xr:.3f}m, {yr:.3f}m)")
                print()
                return [xr, yr]
            except ValueError:
                log("AVISO", "Valor inválido. Insere um número (ex: 1.50)")

    # Primeira passagem — pedir todos os pontos em sequência
    for i in range(n):
        pts_reais[i] = pedir_coordenada(i)

    # ── Revisão e correção de coordenadas ─────────────────
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
                log("INFO", f"A reescrever coordenadas do ponto {idx_corr + 1}...")
                pts_reais[idx_corr] = pedir_coordenada(idx_corr)
            else:
                log("AVISO", f"Número fora do intervalo (1–{n}). Tenta novamente.")
        except ValueError:
            log("AVISO", "Entrada inválida. Insere o número do ponto ou ENTER.")

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

    # ── Guardar registo histórico em resultados/calibracao/ ─
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON com todos os pontos (pixel + metro) usados nesta calibração
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
    log("OK", f"Registo de pontos guardado: resultados/calibracao/{json_calib_path.name}")

    # Imagem anotada com as posições dos pontos (cópia na subpasta)
    img_calib_path = PASTA_CALIB_REF / f"imagem_{ts_str}.png"
    ok_enc, buf = cv2.imencode(".png", img_ref)
    if ok_enc:
        img_calib_path.write_bytes(buf.tobytes())
        log("OK", f"Imagem de calibração guardada: resultados/calibracao/{img_calib_path.name}")
    else:
        log("AVISO", "Não foi possível guardar a imagem na pasta calibracao.")

    # ── Pré-visualização top-down (verificação visual da homografia) ─
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
            log("OK", f"Preview top-down guardado: resultados/calibracao/{topdown_path.name}")
        else:
            log("AVISO", "Não foi possível codificar o preview top-down.")
    except Exception as e:
        log("AVISO", f"Falha ao gerar preview top-down: {e}")

    log("OK", f"  ppm={ppm:.1f} | área={W_real:.2f}×{D_real:.2f}m | "
              f"saída top-down={out_w_px}×{out_h_px}px")
    sys.exit(0)

# ─────────────────────────────────────────────
#  MODO PRODUÇÃO
# ─────────────────────────────────────────────
def _px_para_metros(cx: float, cy: float,
                    H: np.ndarray, ppm: float,
                    x_min: float, y_min: float) -> tuple[float, float]:
    """Aplica undistort + homografia e devolve (x_metros, y_metros)."""
    ux, uy  = undistort_ponto(cx, cy)
    pt_warp = cv2.perspectiveTransform(
        np.array([[[ux, uy]]], dtype=np.float32), H)
    x_metros = float(pt_warp[0][0][0]) / ppm + x_min
    y_metros = float(pt_warp[0][0][1]) / ppm + y_min
    return round(x_metros, 4), round(y_metros, 4)


def servidor_producao(calib: dict):
    """
    Servidor de retificação em loop contínuo.
    Recebe pacotes do VisionProcessing com bounding boxes em píxeis
    e posição ArUco do robô, converte tudo para metros e guarda:
      - resultados/posicoes/posicao_NNNN.json  — coordenadas em metros
      - resultados/imagens/frame_NNNN.jpg      — frame anotado
    """
    H     = np.array(calib["H_mat"])
    ppm   = calib["ppm"]
    x_min = calib.get("x_min", 0.0)
    y_min = calib.get("y_min", 0.0)

    # Tamanho do canvas de saída do warpPerspective.
    # Preferimos os campos guardados na calibração; se não existirem
    # (calibrações antigas), caímos para a resolução do frame de calibração.
    if "output_size_px" in calib and calib["output_size_px"]:
        out_w_px, out_h_px = calib["output_size_px"]
    elif "W_real_m" in calib and "D_real_m" in calib:
        out_w_px = int(round(float(calib["W_real_m"]) * ppm))
        out_h_px = int(round(float(calib["D_real_m"]) * ppm))
    else:
        # Fallback: usa a resolução da calibração — funciona porque o ppm
        # foi escolhido para que max(out) ≈ max(resolucao_calib).
        res = calib.get("resolucao_calib", [1920, 1080])
        out_w_px, out_h_px = int(res[0]), int(res[1])
        log("AVISO", "Calibração antiga sem 'output_size_px' — a usar fallback "
                     f"de {out_w_px}×{out_h_px}px. Recalibra para mais precisão.")
    out_w_px = max(out_w_px, 1)
    out_h_px = max(out_h_px, 1)

    PASTA_POSICOES.mkdir(parents=True, exist_ok=True)
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
    PASTA_IMAGENS_TD.mkdir(parents=True, exist_ok=True)
    log("OK", f"Pastas prontas: {PASTA_POSICOES.name} | {PASTA_IMAGENS.name} | "
              f"{PASTA_IMAGENS_TD.name}")
    log("INFO", f"Vista top-down: {out_w_px}×{out_h_px}px @ {ppm:.1f}ppm "
                f"({out_w_px/ppm:.2f}m × {out_h_px/ppm:.2f}m)")

    log("FASE", "Servidor de retificação ativo (porta 6001)")
    log("INFO", f"Calibração: ppm={ppm:.1f} | erro médio={calib.get('erro_medio_m','?')}m")
    log("INFO", f"Posições  → .../{PASTA_POSICOES.relative_to(BASE_PATH)}")
    log("INFO", f"Imagens   → .../{PASTA_IMAGENS.relative_to(BASE_PATH)}")
    log("INFO", f"Top-down  → .../{PASTA_IMAGENS_TD.relative_to(BASE_PATH)}")
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
                    robo_px  = pacote.get("robo_px", {})   # compatível com versões antigas
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

                    # Log do robô
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

                    # ── Construir JSON de saída ────────────────────────────
                    latencia = round(
                        (time.time() - pacote["timestamp_visao"]) * 1000, 2)
                    saida = {
                        "indice":      indice,
                        "latencia_ms": latencia,
                        "n_bolas":     len(res_bolas),
                        "trajetoria":  res_bolas,
                        "robo":        res_robo,
                    }

                    # ── Guardar posições ───────────────────────────────────
                    fich_json = PASTA_POSICOES / f"posicao_{indice:04d}.json"
                    with open(fich_json, "w") as f:
                        json.dump(numpy_para_python(saida), f, indent=4)

                    # ── Guardar imagem ─────────────────────────────────────
                    # Usa imencode + escrita binária em vez de cv2.imwrite,
                    # evitando falhas com caminhos Unicode/acentuados no Windows
                    # (ex.: OneDrive com "4º Ano", espaços, etc.)
                    fich_img = PASTA_IMAGENS / f"frame_{indice:04d}.jpg"
                    if frame is not None and hasattr(frame, "shape"):
                        try:
                            ok_enc, buf = cv2.imencode(
                                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            if ok_enc:
                                fich_img.write_bytes(buf.tobytes())
                                log("OK", f"Imagem guardada: {fich_img.name}")
                            else:
                                log("ERRO", f"cv2.imencode falhou para {fich_img.name} "
                                            f"(shape={frame.shape}, dtype={frame.dtype})")
                        except Exception as e_img:
                            log("ERRO", f"Erro ao guardar imagem {fich_img.name}: {e_img}")

                        # ── Vista de cima (homografia aplicada) ──────────
                        # undistort + warpPerspective sobre o MESMO frame anotado,
                        # para que as bounding boxes e marcadores ArUco também
                        # apareçam na vista corrigida.
                        fich_img_td = PASTA_IMAGENS_TD / f"frame_{indice:04d}.jpg"
                        try:
                            frame_td = aplicar_topdown(frame, H, out_w_px, out_h_px)
                            ok_td, buf_td = cv2.imencode(
                                ".jpg", frame_td, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            if ok_td:
                                fich_img_td.write_bytes(buf_td.tobytes())
                                log("OK", f"Top-down guardado: {fich_img_td.name}")
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
                    log("OK",
                        f"Frame {indice:04d} | {len(res_bolas)} bola(s) | "
                        f"robô={robo_log} | latência={latencia}ms | total={total_frames} frames")

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
