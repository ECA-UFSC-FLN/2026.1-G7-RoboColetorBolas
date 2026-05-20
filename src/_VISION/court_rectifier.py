"""
retificador.py — Calibração e Retificação de Coordenadas UFSC/FEUP
===================================================================
Dois modos de operação:

  --calibrar   Recebe um frame via socket (porta 6001), permite ao utilizador
               marcar pontos de referência com coordenadas reais conhecidas,
               calcula a homografia e guarda o JSON de calibração.

  (sem args)   Servidor de produção. Recebe pacotes do VisionProcessing
               com bounding boxes em píxeis, aplica a homografia e guarda
               as coordenadas em metros em ficheiros JSON.

Perfis de câmara (geridos por parametros.py):
  IPHONE16_1X   — lente principal 1× (f≈5823px, 4032×3024)
  IPHONE16_05X  — ultra-grande angular 0.5× (f≈2912px, 4032×3024)
  EXTERNO        — parâmetros configuráveis pelo utilizador (ext_fx, ext_fy, …)

Correção de paralaxe (nova):
  Ao capturar de cima, o centro 2D detetado de um objeto com altura h
  é deslocado radialmente em relação ao ponto nadir da câmara. A função
  corrigir_altura_px() calcula o deslocamento esperado e move o ponto
  de volta ao solo. Esta correção é aplicada:
    • a CADA bola (altura = raio da bola → parâmetro altura_bola_m)
    • ao centroide FRONTAL e TRASEIRO de cada ArUco (altura = topo do robô
      → parâmetro altura_aruco_m)

NOTA: Uma thread separada fica à escuta na porta 6011 (PORTA_HEALTH)
      exclusivamente para responder a health-checks do MasterControl.

Em produção, o retificador também serve uma FILA de JSONs ao
GraphProcessor na porta 6020 (authkey b"grafo_ufsc").
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from multiprocessing.connection import Listener
from datetime import datetime

from _COMMON.logging_utils import log as _log
import _CONFIG.system_parameters as _params

MOD = "RETIFICADOR"

def log(nivel: str, msg: str):
    _log(MOD, nivel, msg)


# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
BASE_PATH = Path(__file__).resolve().parents[1]
PASTA_SAIDA       = BASE_PATH / "resultados"
PASTA_POSICOES    = PASTA_SAIDA / "posicoes"
PASTA_IMAGENS     = PASTA_SAIDA / "imagens"
PASTA_IMAGENS_TD  = PASTA_SAIDA / "imagens_topdown"
PASTA_CALIB_REF   = PASTA_SAIDA / "calibracao"
CALIB_FILE        = PASTA_CALIB_REF / "homografia_calibracao.json"

PORTA        = 6001
PORTA_HEALTH = 6011
PORTA_GRAFO  = 6020
AUTHKEY      = b"retificador_ufsc"
AUTHKEY_GRAFO = b"grafo_ufsc"


# ─────────────────────────────────────────────
#  INTRÍNSECOS — carregados uma vez ao arrancar
# ─────────────────────────────────────────────
def _carregar_intrinsicos() -> tuple:
    """
    Lê parametros.json e devolve (K, D, resolucao_ref).
    Chamado no arranque de cada modo (calibrar / produção).
    """
    cfg = _params.carregar()
    K, D, res = _params.obter_intrinsics(cfg)
    perfil = cfg.get("perfil_camara", "IPHONE16_1X")
    log("HUMANO", f"Perfil de câmara: {perfil}  (ref. {res[0]}×{res[1]}px)")
    log("DEBUG",  f"K=\n{K}\nD={D.tolist()}")
    return K, D, res, cfg


# Cache dos mapas de undistort (calculados uma vez por (K,D,resolução)).
_UNDISTORT_MAPS: dict = {}

def _obter_maps_undistort(K, D, w: int, h: int):
    chave = (w, h, id(K))   # id(K) diferencia perfis distintos na mesma sessão
    if chave not in _UNDISTORT_MAPS:
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, K, (w, h), cv2.CV_16SC2)
        _UNDISTORT_MAPS[chave] = (map1, map2)
        log("DEBUG", f"Mapas de undistort pré-calculados para {w}×{h}px.")
    return _UNDISTORT_MAPS[chave]


def aplicar_undistort(img, K, D):
    h, w = img.shape[:2]
    map1, map2 = _obter_maps_undistort(K, D, w, h)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)


def undistort_ponto(cx: float, cy: float, K, D):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    pt_corr = cv2.undistortPoints(pt, K, D, None, K)
    return float(pt_corr[0][0][0]), float(pt_corr[0][0][1])


def aplicar_topdown(img, H, out_w: int, out_h: int, K, D):
    img_undist = aplicar_undistort(img, K, D)
    return cv2.warpPerspective(
        img_undist, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


# ─────────────────────────────────────────────
#  CORREÇÃO DE PARALAXE POR ALTURA DO OBJETO
# ─────────────────────────────────────────────
def corrigir_altura_px(cx: float, cy: float,
                       K,
                       altura_objeto_m: float,
                       altura_camara_m: float) -> tuple[float, float]:
    """
    Corrige o deslocamento de paralaxe introduzido pela altura de um objeto
    numa câmara overhead com modelo pinhole.

    Princípio: um ponto no solo à posição (Xw, Yw, 0) aparece na imagem
    em (cx0, cy0). Se o objeto tem altura h, o seu topo (visível) está em
    (Xw, Yw, h) e projeta em (cx, cy) — deslocado radialmente do nadir
    (ponto da imagem correspondente à posição da câmara).

    O fator de correção é:
        escala = (H - h) / H       onde H = altura_camara_m, h = altura_objeto_m
    Aplicando a escala em coordenadas normalizadas (sem distorção):
        x_norm_corr = x_norm * escala
        y_norm_corr = y_norm * escala

    Se a altura do objeto for zero (ou a câmara estiver muito baixa),
    devolve (cx, cy) sem alteração.

    Parâmetros
    ----------
    cx, cy           : coordenadas do ponto detetado em pixéis (após undistort)
    K                : matriz intrínseca 3×3
    altura_objeto_m  : altura do objeto ao solo (m) — raio da bola ou topo do ArUco
    altura_camara_m  : altura da câmara ao solo (m)

    Devolve
    -------
    (cx_corr, cy_corr) — coordenadas corrigidas em pixéis
    """
    if altura_camara_m <= 0 or altura_objeto_m <= 0:
        return cx, cy

    escala = max(0.0, (altura_camara_m - altura_objeto_m) / altura_camara_m)

    fx = K[0, 0];  fy = K[1, 1]
    ox = K[0, 2];  oy = K[1, 2]

    # Converter para coordenadas normalizadas
    xn = (cx - ox) / fx
    yn = (cy - oy) / fy

    # Aplicar escala (move o ponto projetado para o solo)
    xn_c = xn * escala
    yn_c = yn * escala

    # Voltar a pixéis
    cx_c = xn_c * fx + ox
    cy_c = yn_c * fy + oy

    return float(cx_c), float(cy_c)


# ─────────────────────────────────────────────
#  FILA PARA O GRAPH PROCESSOR
# ─────────────────────────────────────────────
_fila_grafo: queue.Queue = queue.Queue(maxsize=500)


# ─────────────────────────────────────────────
#  FILA DE ESCRITA DE FICHEIROS (worker assíncrono)
# ─────────────────────────────────────────────
# As escritas de JSON + JPEG são feitas numa thread daemon separada.
# O LIBERADO é enviado IMEDIATAMENTE após a retificação, sem esperar
# pelo disco — isto elimina o bottleneck de ~1 FPS causado por 3
# escritas síncronas de ficheiros a cada frame.
_fila_escrita: queue.Queue = queue.Queue(maxsize=300)


def _tentar_enfileirar_escrita(tarefa: dict) -> bool:
    """Best effort: se a fila estiver cheia, o ciclo principal segue sem bloquear."""
    try:
        _fila_escrita.put_nowait(tarefa)
        return True
    except queue.Full:
        return False


def _worker_escrita():
    """Thread daemon que drena a fila de escritas de ficheiros."""
    while True:
        tarefa = _fila_escrita.get()
        try:
            tipo = tarefa.get("tipo")

            if tipo == "json":
                with open(tarefa["caminho"], "w") as _f:
                    json.dump(tarefa["dados"], _f, indent=4)

            elif tipo == "jpeg":
                ok_w, buf_w = cv2.imencode(
                    ".jpg", tarefa["frame"],
                    [cv2.IMWRITE_JPEG_QUALITY, tarefa.get("qualidade", 90)])
                if ok_w:
                    Path(tarefa["caminho"]).write_bytes(buf_w.tobytes())

            elif tipo == "topdown":
                frame_td = aplicar_topdown(
                    tarefa["frame"], tarefa["H"],
                    tarefa["out_w"], tarefa["out_h"],
                    tarefa["K"], tarefa["D"])
                ok_w, buf_w = cv2.imencode(
                    ".jpg", frame_td,
                    [cv2.IMWRITE_JPEG_QUALITY, tarefa.get("qualidade", 90)])
                if ok_w:
                    Path(tarefa["caminho"]).write_bytes(buf_w.tobytes())

        except Exception:
            pass   # nunca bloquear o worker por erros de disco
        finally:
            _fila_escrita.task_done()


def _iniciar_worker_escrita():
    t = threading.Thread(target=_worker_escrita, daemon=True, name="worker-escrita")
    t.start()
    return t


# ─────────────────────────────────────────────
#  HEALTH-CHECK SERVER (porta 6011)
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
    log("DEBUG", f"Health-check ativo na porta {porta}")


# ─────────────────────────────────────────────
#  SERVIDOR DA FILA PARA O GRAPHPROCESSOR
# ─────────────────────────────────────────────
def iniciar_servidor_grafo(porta: int = PORTA_GRAFO):
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
#  CONVERSÃO PÍXEIS → METROS (com correção)
# ─────────────────────────────────────────────
def _px_para_metros(cx: float, cy: float,
                    H, ppm: float,
                    x_min: float, y_min: float,
                    K, D,
                    altura_objeto_m: float = 0.0,
                    altura_camara_m: float = 0.0,
                    H_metros=None):
    """
    Pipeline completo:
      1. Undistort do ponto
      2. Correção de paralaxe por altura do objeto (se altura > 0)
      3. Warp perspetiva (homografia)
      4. Conversão para metros
    """
    ux, uy = undistort_ponto(cx, cy, K, D)

    if altura_objeto_m > 0 and altura_camara_m > 0:
        ux, uy = corrigir_altura_px(ux, uy, K, altura_objeto_m, altura_camara_m)

    if H_metros is not None:
        pt_m = cv2.perspectiveTransform(
            np.array([[[ux, uy]]], dtype=np.float32), H_metros)
        return round(float(pt_m[0][0][0]), 4), round(float(pt_m[0][0][1]), 4)

    pt_warp = cv2.perspectiveTransform(
        np.array([[[ux, uy]]], dtype=np.float32), H)
    x_metros = float(pt_warp[0][0][0]) / ppm + x_min
    y_metros = float(pt_warp[0][0][1]) / ppm + y_min
    return round(x_metros, 4), round(y_metros, 4)


# ═════════════════════════════════════════════════════════════════════
#  MODO CALIBRAÇÃO
# ═════════════════════════════════════════════════════════════════════
def calibrar_via_socket():
    iniciar_health_server()
    try:
        cv2.startWindowThread()
    except Exception:
        pass

    K, D, res, cfg = _carregar_intrinsicos()

    log("HUMANO", "Servidor de calibração ativo na porta 6001")
    log("HUMANO", "Aguardando frame do imageStreaming... (captura com tecla C)")
    address = ("localhost", PORTA)
    with Listener(address, authkey=AUTHKEY) as listener:
        with listener.accept() as conn:
            pacote = conn.recv()
            img    = pacote["frame"]
    log("HUMANO", "Frame recebido. A preparar janela de calibração...")

    img_undist = aplicar_undistort(img, K, D)
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
            cv2.drawMarker(
                base, (px, py), (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(base, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(base, str(idx0 + 1),
                        (px + 10, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(base, f"({px},{py})",
                        (px + 10, py + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.imshow(JANELA, base)
        return base

    def on_clique(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_px) < n:
            pts_px.append((x, y))
            log("DEBUG", f"Ponto {len(pts_px)}/{n} marcado em px=({x}, {y})")
            redesenhar_pontos()

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        cv2.setWindowProperty(JANELA, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass
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

    janela_viva = threading.Event()
    janela_viva.set()

    def manter_janela_responsiva():
        while janela_viva.is_set():
            try:
                cv2.waitKey(50)
            except Exception:
                break

    t_janela = threading.Thread(
        target=manter_janela_responsiva,
        daemon=True,
        name="calibracao-janela-responsiva",
    )
    t_janela.start()

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

    janela_viva.clear()
    try:
        cv2.waitKey(1)
    except Exception:
        pass
    cv2.destroyAllWindows()

    xs     = [p[0] for p in pts_reais]
    ys     = [p[1] for p in pts_reais]
    x_min  = min(xs);  y_min = min(ys)
    W_real = max(xs) - x_min
    D_real = max(ys) - y_min
    if W_real <= 0 or D_real <= 0:
        log("ERRO", "As coordenadas reais não formam uma área válida.")
        sys.exit(1)

    pts_np = np.array(pts_px, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts_np[i] - pts_np[j]) < 8:
                log("ERRO", f"Pontos {i+1} e {j+1} estão praticamente sobrepostos.")
                sys.exit(1)

    # ── PPM calculado a partir da homografia, não da resolução da imagem ──────
    # Estratégia em 2 passos:
    #   1. Calcular homografia provisória H_tmp com destino em metros (escala 1:1)
    #   2. Medir a escala local do jacobiano de H_tmp no centro dos pontos de
    #      calibração → ppm = 1 / (metros_por_pixel)
    # Isto é muito mais preciso do que max(w_img,h_img)/max(W_real,D_real), que
    # assume que a quadra preenche toda a imagem — raramente verdadeiro.
    pts_reais_m = np.array(
        [((p[0] - x_min), (p[1] - y_min)) for p in pts_reais],
        dtype=np.float32
    )
    pts_origem_px = np.array(pts_px, dtype=np.float32)

    # Homografia provisória (metros)
    if n == 4:
        H_tmp, _ = cv2.findHomography(pts_origem_px, pts_reais_m)
    else:
        H_tmp, _ = cv2.findHomography(pts_origem_px, pts_reais_m,
                                       cv2.RANSAC, ransacReprojThreshold=0.005)

    if H_tmp is None:
        log("AVISO", "Homografia provisória falhou — ppm fallback heurístico.")
        ppm = (max(w_img, h_img) / max(W_real, D_real)
               if max(W_real, D_real) > 0 else 200.0)
    else:
        # Jacobiano numérico de H_tmp no centro dos pontos de calibração
        cx_pts = float(np.mean([p[0] for p in pts_px]))
        cy_pts = float(np.mean([p[1] for p in pts_px]))
        eps    = 1.0   # 1 pixel — deslocamento para derivada numérica
        def _warp1(x, y, Hmat):
            r = cv2.perspectiveTransform(
                np.array([[[x, y]]], dtype=np.float32), Hmat)
            return float(r[0,0,0]), float(r[0,0,1])
        pc  = _warp1(cx_pts,       cy_pts,       H_tmp)
        pdx = _warp1(cx_pts + eps, cy_pts,       H_tmp)
        pdy = _warp1(cx_pts,       cy_pts + eps, H_tmp)
        # metros/pixel de imagem (escala local do warp)
        sx = abs(pdx[0] - pc[0])
        sy = abs(pdy[1] - pc[1])
        m_por_px = (sx + sy) / 2.0
        if m_por_px < 1e-9:
            log("AVISO", "Escala do warp nula — ppm fallback heurístico.")
            ppm = (max(w_img, h_img) / max(W_real, D_real)
                   if max(W_real, D_real) > 0 else 200.0)
        else:
            ppm = 1.0 / m_por_px
            log("DEBUG",
                f"ppm calculado pelo jacobiano de H: {ppm:.1f} px/m "
                f"(escala local: {m_por_px*100:.3f} cm/px  |  "
                f"resolução imagem {w_img}×{h_img}px  |  "
                f"quadra {W_real:.2f}m×{D_real:.2f}m)")
            ppm_heuristico = (max(w_img, h_img) / max(W_real, D_real)
                              if max(W_real, D_real) > 0 else 200.0)
            if abs(ppm / ppm_heuristico - 1.0) > 0.05:
                log("AVISO",
                    f"ppm heurístico era {ppm_heuristico:.1f} (diferença de "
                    f"{abs(ppm/ppm_heuristico-1)*100:.1f}%) — "
                    f"o cálculo pelo jacobiano é mais preciso e foi adoptado.")

    pts_destino_px = np.array(
        [((p[0] - x_min) * ppm, (p[1] - y_min) * ppm) for p in pts_reais],
        dtype=np.float32
    )

    if n == 4:
        H, mask = cv2.findHomography(pts_origem_px, pts_destino_px)
        H_metros, _ = cv2.findHomography(pts_origem_px, np.array(pts_reais, dtype=np.float32))
        inliers  = 4
    else:
        H, mask = cv2.findHomography(pts_origem_px, pts_destino_px,
                                      cv2.RANSAC, ransacReprojThreshold=2.0)
        H_metros, _ = cv2.findHomography(
            pts_origem_px,
            np.array(pts_reais, dtype=np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=0.03,
        )
        inliers  = int(mask.sum()) if mask is not None else 0

    if H is None or H_metros is None:
        log("ERRO", "Não foi possível calcular a homografia. Pontos colineares?")
        sys.exit(1)

    erros = []
    for i in range(n):
        dst_pred = cv2.perspectiveTransform(
            np.array([[[pts_origem_px[i][0], pts_origem_px[i][1]]]], dtype=np.float32), H
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

    # Guardar perfil de câmara usado na calibração para referência futura
    perfil_calib = cfg.get("perfil_camara", "IPHONE16_1X")

    calib = numpy_para_python({
        "H_mat":           H,
        "H_metros_mat":    H_metros,
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
        "perfil_camara":   perfil_calib,
        "homografia_usa_frame_undistort": True,
        "data":            datetime.now().isoformat(timespec="seconds"),
    })

    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=4)
    log("HUMANO", f"Calibração guardada: {CALIB_FILE.name}")

    # ── Imagem de referência anotada ──
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
                f"perfil={perfil_calib} | {datetime.now().strftime('%H:%M:%S')}",
                (20, h_img - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)

    cv2.imwrite(str(IMG_REF_PATH), img_ref)
    log("DEBUG", f"Imagem de referência guardada: {IMG_REF_PATH.name}")

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pontos_calib = {
        "data":         datetime.now().isoformat(timespec="seconds"),
        "n_pontos":     n,
        "ppm":          round(ppm, 3),
        "erro_medio_m": round(erro_medio_m, 5),
        "inliers":      inliers,
        "perfil_camara": perfil_calib,
        "pontos": [
            {
                "numero":   i + 1,
                "px_x":     pts_px[i][0],
                "px_y":     pts_px[i][1],
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
def servidor_producao(calib: dict):
    K, D, res, cfg = _carregar_intrinsicos()

    H     = np.array(calib["H_mat"])
    H_metros = (np.array(calib["H_metros_mat"])
                if calib.get("H_metros_mat") is not None else None)
    ppm   = calib["ppm"]
    x_min = calib.get("x_min", 0.0)
    y_min = calib.get("y_min", 0.0)

    altura_camara_m = float(cfg.get("altura_camara_m", 0.0))
    altura_bola_m   = float(cfg.get("altura_bola_m",   0.0))
    altura_aruco_m  = float(cfg.get("altura_aruco_m",  0.0))
    guardar_disco   = bool(int(cfg.get("guardar_resultados_disco", 0)))
    guardar_imagens = guardar_disco and bool(int(cfg.get("guardar_imagens_debug", 0)))
    intervalo_guardar_imagens_s = float(cfg.get("intervalo_guardar_imagens_s", 5.0))

    log("DEBUG", f"Correção de paralaxe: câmara={altura_camara_m:.2f}m | "
                 f"bola={altura_bola_m*100:.1f}cm | ArUco={altura_aruco_m*100:.1f}cm")

    # Aviso se os parâmetros de câmara mudaram desde a calibração
    perfil_calib = calib.get("perfil_camara")
    perfil_atual = cfg.get("perfil_camara", "IPHONE16_1X")
    if perfil_calib and perfil_calib != perfil_atual:
        log("AVISO", f"ATENÇÃO: calibração foi feita com perfil '{perfil_calib}' "
                     f"mas o perfil atual é '{perfil_atual}'. Recalibra antes de produção!")

    if "output_size_px" in calib and calib["output_size_px"]:
        out_w_px, out_h_px = calib["output_size_px"]
    elif "W_real_m" in calib and "D_real_m" in calib:
        out_w_px = int(round(float(calib["W_real_m"]) * ppm))
        out_h_px = int(round(float(calib["D_real_m"]) * ppm))
    else:
        res_c = calib.get("resolucao_calib", [1920, 1080])
        out_w_px, out_h_px = int(res_c[0]), int(res_c[1])
        log("AVISO", "Calibração antiga sem 'output_size_px' — a usar fallback "
                     f"de {out_w_px}×{out_h_px}px. Recalibra para mais precisão.")
    out_w_px = max(out_w_px, 1)
    out_h_px = max(out_h_px, 1)

    if guardar_disco:
        PASTA_POSICOES.mkdir(parents=True, exist_ok=True)
    if guardar_imagens:
        PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)
        PASTA_IMAGENS_TD.mkdir(parents=True, exist_ok=True)
    log("DEBUG", f"Escrita em disco={'ON' if guardar_disco else 'OFF'} | "
                 f"imagens debug={'ON' if guardar_imagens else 'OFF'}")
    log("DEBUG", f"Vista top-down: {out_w_px}×{out_h_px}px @ {ppm:.1f}ppm "
                f"({out_w_px/ppm:.2f}m × {out_h_px/ppm:.2f}m)")

    iniciar_servidor_grafo()
    if guardar_disco:
        _iniciar_worker_escrita()
        log("DEBUG", "Worker de escrita assíncrona iniciado.")

    log("HUMANO", "Retificador pronto. A aguardar pacotes do VisionProcessing...")
    log("DEBUG",  f"Calibração: ppm={ppm:.1f} | erro médio={calib.get('erro_medio_m','?')}m")
    if guardar_disco:
        log("DEBUG",  f"Posições  → .../{PASTA_POSICOES.relative_to(BASE_PATH)}")
    if guardar_imagens:
        log("DEBUG",  f"Imagens   → .../{PASTA_IMAGENS.relative_to(BASE_PATH)}")
        log("DEBUG",  f"Top-down  → .../{PASTA_IMAGENS_TD.relative_to(BASE_PATH)}")

    total_frames = 0
    total_bolas  = 0
    proxima_imagem_debug = 0.0

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

                    # ── Retificar bolas (com correção de paralaxe) ─────────
                    res_bolas = []
                    for b in bolas_px:
                        cx = (b["x1"] + b["x2"]) / 2.0
                        cy = (b["y1"] + b["y2"]) / 2.0
                        xm, ym = _px_para_metros(
                            cx, cy, H, ppm, x_min, y_min, K, D,
                            altura_objeto_m=altura_bola_m,
                            altura_camara_m=altura_camara_m,
                            H_metros=H_metros,
                        )
                        res_bolas.append({"x": xm, "y": ym})

                    # ── Retificar robô/ArUco (com correção de paralaxe) ────
                    res_robo = {
                        "frontal":          None,
                        "traseiro":         None,
                        "orientacao_graus": robo_px.get("orientacao_graus"),
                    }
                    if robo_px.get("frontal"):
                        xm, ym = _px_para_metros(
                            robo_px["frontal"]["cx"], robo_px["frontal"]["cy"],
                            H, ppm, x_min, y_min, K, D,
                            altura_objeto_m=altura_aruco_m,
                            altura_camara_m=altura_camara_m,
                            H_metros=H_metros,
                        )
                        res_robo["frontal"] = {"x": xm, "y": ym}
                    if robo_px.get("traseiro"):
                        xm, ym = _px_para_metros(
                            robo_px["traseiro"]["cx"], robo_px["traseiro"]["cy"],
                            H, ppm, x_min, y_min, K, D,
                            altura_objeto_m=altura_aruco_m,
                            altura_camara_m=altura_camara_m,
                            H_metros=H_metros,
                        )
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

                    # ── Empurrar para fila do GraphProcessor ───────────────
                    try:
                        _fila_grafo.put_nowait(saida_serializavel)
                    except queue.Full:
                        try:
                            _fila_grafo.get_nowait()
                            _fila_grafo.put_nowait(saida_serializavel)
                            log("AVISO", "Fila do grafo cheia — descartado pacote mais antigo.")
                        except Exception:
                            pass

                    # ── LIBERADO imediatamente — não espera pelas escritas ──
                    # As escritas de disco são assíncronas (worker thread).
                    # Isto elimina o bottleneck de ~1 FPS causado pelas 3
                    # escritas síncronas de ficheiros por frame.
                    conn.send("LIBERADO")

                    if guardar_disco:
                        fich_json = PASTA_POSICOES / f"posicao_{indice:04d}.json"
                        _tentar_enfileirar_escrita({
                            "tipo":    "json",
                            "caminho": str(fich_json),
                            "dados":   saida_serializavel,
                        })

                    agora_img = time.time()
                    deve_guardar_imagem = (
                        guardar_imagens
                        and frame is not None
                        and hasattr(frame, "shape")
                        and agora_img >= proxima_imagem_debug
                    )
                    if deve_guardar_imagem:
                        proxima_imagem_debug = agora_img + intervalo_guardar_imagens_s
                        fich_img = PASTA_IMAGENS / f"frame_{indice:04d}.jpg"
                        _tentar_enfileirar_escrita({
                            "tipo":      "jpeg",
                            "caminho":   str(fich_img),
                            "frame":     frame,
                            "qualidade": 90,
                        })

                        fich_img_td = PASTA_IMAGENS_TD / f"frame_{indice:04d}.jpg"
                        _tentar_enfileirar_escrita({
                            "tipo":      "topdown",
                            "caminho":   str(fich_img_td),
                            "frame":     frame,
                            "H":         H,
                            "out_w":     out_w_px,
                            "out_h":     out_h_px,
                            "K":         K,
                            "D":         D,
                            "qualidade": 90,
                        })

                    total_frames += 1
                    total_bolas  += len(res_bolas)
                    log("DEBUG",
                        f"Frame {indice:04d} | {len(res_bolas)} bola(s) | "
                        f"robô={robo_log} | latência={latencia}ms | total={total_frames} frames")
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
            log("AVISO", "Apaga o ficheiro e recalibra (MasterControl → opção 2).")
            time.sleep(60)
            sys.exit(1)

        log("HUMANO", f"Calibração de {calib.get('data','data desconhecida')} carregada.")
        log("DEBUG",  f"ppm={calib['ppm']:.1f} | {calib.get('n_pontos','?')} pontos | "
                     f"erro_medio_m={calib.get('erro_medio_m','?')} | "
                     f"perfil={calib.get('perfil_camara','(não registado)')}")
        servidor_producao(calib)



