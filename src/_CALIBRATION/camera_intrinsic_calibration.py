"""
calibracao_camera.py — Calibração Intrínseca por Tabuleiro de Xadrez UFSC/FEUP
=================================================================================
Calcula os parâmetros intrínsecos reais da câmara (K, D) a partir de
múltiplas fotos de um tabuleiro de xadrez — o método correto para eliminar
erros de distorção sistemáticos que afetam diretamente o erro de homografia.

Dois modos:
  --capturar    Liga-se ao imageStreaming (modo CALIBRACAO) e captura frames
                com a tecla C. Recomendado: 15–25 posições variadas do
                tabuleiro (diferentes ângulos, distâncias, regiões da imagem).

  --ficheiros   Lê frames já capturados de uma pasta (útil se já tens fotos
                do tabuleiro).

  (sem args)    Modo interativo: captura ao vivo + preview do detetor.

Saída:
  resultados/calibracao/intrinsicos_camera.json
    → K (3×3), D (1×5), resolucao, rms, n_frames, data
    → Compatível com parametros.py perfil "CALIBRADO"

Uso do resultado:
  No parametros.json, define "perfil_camara": "CALIBRADO" — o retificador
  e o VisionProcessing passam a usar os intrínsecos calculados aqui em vez
  dos perfis fixos do iPhone.

Configuração do tabuleiro:
  COLS × ROWS = cantos interiores (não quadrados!).
  Exemplo: tabuleiro 10×7 quadrados → COLS=9, ROWS=6
  TAMANHO_QUADRADO_M: comprimento real de cada quadrado em metros.

Teclas durante captura ao vivo:
  C       — capturar frame atual (se detetor encontrar tabuleiro)
  SPACE   — capturar frame (mesmo que C)
  D       — apagar último frame capturado
  ENTER   — terminar captura e calcular
  ESC/Q   — cancelar
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
import time
import ctypes
import subprocess
from pathlib import Path
from datetime import datetime
from multiprocessing.connection import Client

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO DO TABULEIRO
# ─────────────────────────────────────────────
# Cantos INTERIORES do tabuleiro (quadrados - 1 em cada direção)
COLS = 9          # cantos interiores na horizontal
ROWS = 6          # cantos interiores na vertical
TAMANHO_QUADRADO_M = 0.025   # 2.5 cm por quadrado (ajustar ao tabuleiro real)

# Mínimo de frames com deteção bem-sucedida para aceitar calibração
MIN_FRAMES_CALIBRACAO = 10

# Configuração da ligação ao imageStreaming
PORTA_RET   = 6001
AUTHKEY_RET = b"retificador_ufsc"

# ─────────────────────────────────────────────
#  CAMINHOS
# ─────────────────────────────────────────────
BASE_PATH = Path(__file__).resolve().parents[1]
PASTA_CALIB = BASE_PATH / "resultados" / "calibracao"
SAIDA_JSON  = PASTA_CALIB / "intrinsicos_camera.json"
PASTA_FRAMES_CAPTURADOS = PASTA_CALIB / "frames_tabuleiro"

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"


# ─────────────────────────────────────────────
#  LOGGING SIMPLES (independente de bolas_log)
# ─────────────────────────────────────────────
def log(nivel: str, msg: str):
    cores = {
        "HUMANO": "\033[97m",
        "DEBUG":  "\033[90m",
        "AVISO":  "\033[93m",
        "ERRO":   "\033[91m",
        "OK":     "\033[92m",
    }
    reset = "\033[0m"
    cor   = cores.get(nivel, "")
    ts    = datetime.now().strftime("%H:%M:%S")
    print(f"{cor}[{ts}] [{nivel:6s}] {msg}{reset}", flush=True)


# ─────────────────────────────────────────────
#  CRITÉRIOS DO DETETOR (subpixel)
# ─────────────────────────────────────────────
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def preparar_pontos_objeto(cols: int, rows: int, tamanho_m: float) -> np.ndarray:
    """
    Gera as coordenadas 3D dos cantos do tabuleiro no referencial do objeto.
    (0,0,0), (tamanho,0,0), (2*tamanho,0,0), …, (cols*tamanho, rows*tamanho, 0)
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= tamanho_m
    return objp


def detetar_tabuleiro(frame: np.ndarray,
                      cols: int, rows: int,
                      refinar: bool = True) -> tuple[bool, np.ndarray | None]:
    """
    Deteta cantos do tabuleiro num frame. Devolve (encontrado, corners).
    Aplica equalização adaptativa para robustez a iluminação desigual.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE para robustez a iluminação
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    found, corners = cv2.findChessboardCorners(
        gray_eq, (cols, rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )

    if found and refinar:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

    return found, corners


def desenhar_overlay_captura(frame: np.ndarray,
                              found: bool,
                              corners,
                              cols: int, rows: int,
                              n_capturados: int) -> np.ndarray:
    """Desenha overlay informativo durante captura ao vivo."""
    vis = frame.copy()

    if found and corners is not None:
        cv2.drawChessboardCorners(vis, (cols, rows), corners, found)

    h, w = vis.shape[:2]
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    estado_txt = "TABULEIRO DETECTADO ✓" if found else "Aponta para o tabuleiro..."
    cor_estado  = (0, 255, 80) if found else (0, 120, 255)
    cv2.putText(vis, estado_txt,
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor_estado, 2)

    info = (f"Capturados: {n_capturados}  |  Mínimo: {MIN_FRAMES_CALIBRACAO}  |  "
            f"C/SPACE: capturar  D: apagar  ENTER: calcular  ESC: cancelar")
    cv2.putText(vis, info,
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # Barra de progresso
    prog = min(n_capturados / MIN_FRAMES_CALIBRACAO, 1.0)
    bw = int((w - 30) * prog)
    cv2.rectangle(vis, (15, 65), (w - 15, 72), (60, 60, 60), -1)
    cv2.rectangle(vis, (15, 65), (15 + bw, 72),
                  (0, 255, 100) if n_capturados >= MIN_FRAMES_CALIBRACAO else (0, 180, 255), -1)

    return vis


# ─────────────────────────────────────────────
#  RECEBER FRAME DO IMAGESTREAMING
# ─────────────────────────────────────────────
def receber_frame_streaming() -> np.ndarray | None:
    """
    Liga-se ao retificador de calibração (porta 6001) e recebe UM frame
    do imageStreaming (após o utilizador premir C nesse processo).
    """
    log("HUMANO", f"A aguardar frame do imageStreaming na porta {PORTA_RET}...")
    log("HUMANO", "Garante que o imageStreaming está em modo CALIBRACAO e prime C.")
    try:
        # O imageStreaming liga-se a NÓS — somos o servidor neste caso.
        # (igual ao retificador --calibrar)
        from multiprocessing.connection import Listener
        address = ("localhost", PORTA_RET)
        with Listener(address, authkey=AUTHKEY_RET) as listener:
            with listener.accept() as conn:
                pacote = conn.recv()
                frame  = pacote["frame"]
        log("OK", "Frame recebido do imageStreaming.")
        return frame
    except Exception as e:
        log("ERRO", f"Erro ao receber frame: {e}")
        return None


# ─────────────────────────────────────────────
#  CAPTURA AO VIVO (câmara direta via OpenCV)
# ─────────────────────────────────────────────
def modo_captura_ao_vivo(cols: int, rows: int) -> list[np.ndarray]:
    """
    Abre a câmara diretamente (sem depender do imageStreaming) e permite
    capturar frames ao vivo com preview do detetor. Mais prático para
    a fase de calibração intrínseca.
    Devolve lista de frames aceites.
    """
    log("HUMANO", "A abrir câmara para captura ao vivo...")

    # Tentar abrir câmara (mesma lógica do imageStreaming — Camo primeiro)
    cap = None
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "ANY"),
    ]
    for idx in range(5):
        for backend, nome in backends:
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                log("DEBUG", f"Câmara aberta: índice={idx} backend={nome}")
                cap = c
                break
        if cap is not None:
            break

    if cap is None:
        log("ERRO", "Nenhuma câmara disponível.")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log("HUMANO", f"Câmara pronta ({w}×{h}px).")
    log("HUMANO", "Coloca o tabuleiro em várias posições/ângulos e prime C para capturar.")

    JANELA = "Calibração Intrínseca — Tabuleiro de Xadrez"
    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JANELA, min(w, 1280), min(h, 720))

    frames_aceites  = []
    ultimo_found    = False
    ultimo_corners  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            log("AVISO", "Frame inválido — câmera desligada?")
            time.sleep(0.05)
            continue

        found, corners = detetar_tabuleiro(frame, cols, rows, refinar=True)
        ultimo_found   = found
        ultimo_corners = corners

        vis = desenhar_overlay_captura(frame, found, corners, cols, rows,
                                       len(frames_aceites))
        cv2.imshow(JANELA, vis)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla in (ord("c"), ord("C"), 32):   # C ou SPACE
            if found:
                frames_aceites.append(frame.copy())
                log("OK", f"Frame {len(frames_aceites)} capturado "
                           f"(tabuleiro detetado em {len(corners)} cantos).")
                # Flash visual
                vis_flash = vis.copy()
                cv2.rectangle(vis_flash, (0, 0), (w, h), (0, 255, 100), 8)
                cv2.imshow(JANELA, vis_flash)
                cv2.waitKey(150)
            else:
                log("AVISO", "Tabuleiro não detetado neste frame — tenta novamente.")

        elif tecla in (ord("d"), ord("D")):
            if frames_aceites:
                frames_aceites.pop()
                log("AVISO", f"Último frame removido. Restam {len(frames_aceites)}.")
            else:
                log("AVISO", "Nenhum frame para remover.")

        elif tecla == 13:   # ENTER
            if len(frames_aceites) < MIN_FRAMES_CALIBRACAO:
                log("AVISO",
                    f"Só {len(frames_aceites)} frames — mínimo {MIN_FRAMES_CALIBRACAO}. "
                    f"Captura mais antes de continuar.")
            else:
                log("HUMANO", f"ENTER confirmado com {len(frames_aceites)} frames. "
                               f"A calcular calibração...")
                break

        elif tecla in (27, ord("q"), ord("Q")):   # ESC ou Q
            log("AVISO", "Calibração cancelada.")
            frames_aceites = []
            break

    cap.release()
    cv2.destroyAllWindows()
    return frames_aceites


# ─────────────────────────────────────────────
#  CARREGAR FRAMES DE FICHEIROS
# ─────────────────────────────────────────────
def carregar_frames_de_pasta(pasta: Path, cols: int, rows: int) -> list[np.ndarray]:
    """
    Lê imagens de uma pasta e deteta o tabuleiro em cada uma.
    Aceita .jpg, .jpeg, .png, .bmp.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    fichs = sorted(p for p in pasta.iterdir() if p.suffix.lower() in exts)
    if not fichs:
        log("ERRO", f"Nenhuma imagem encontrada em {pasta}")
        return []

    log("HUMANO", f"A processar {len(fichs)} ficheiro(s) em {pasta.name}...")
    frames_aceites = []
    for fich in fichs:
        frame = cv2.imread(str(fich))
        if frame is None:
            log("AVISO", f"Não foi possível ler: {fich.name}")
            continue
        found, _ = detetar_tabuleiro(frame, cols, rows)
        if found:
            frames_aceites.append(frame)
            log("OK", f"  ✓  {fich.name}")
        else:
            log("AVISO", f"  ✗  {fich.name}  (tabuleiro não encontrado)")

    log("HUMANO",
        f"{len(frames_aceites)}/{len(fichs)} frames aceites com tabuleiro detetado.")
    return frames_aceites


# ─────────────────────────────────────────────
#  CALIBRAÇÃO INTRÍNSECA
# ─────────────────────────────────────────────
def calibrar_intrinsicos(frames: list[np.ndarray],
                          cols: int, rows: int,
                          tamanho_m: float) -> dict | None:
    """
    Executa cv2.calibrateCamera nos frames fornecidos.
    Devolve dicionário com K, D, rms e metadados — ou None em caso de erro.
    """
    if len(frames) < MIN_FRAMES_CALIBRACAO:
        log("ERRO",
            f"Frames insuficientes ({len(frames)} < {MIN_FRAMES_CALIBRACAO}). "
            f"Captura mais imagens com o tabuleiro.")
        return None

    objp = preparar_pontos_objeto(cols, rows, tamanho_m)

    obj_points = []   # coordenadas 3D no mundo
    img_points = []   # coordenadas 2D na imagem
    resolucao  = None

    log("HUMANO", f"A detetar cantos em {len(frames)} frames...")
    n_ok = 0
    for i, frame in enumerate(frames):
        found, corners = detetar_tabuleiro(frame, cols, rows, refinar=True)
        if found:
            obj_points.append(objp)
            img_points.append(corners)
            n_ok += 1
            if resolucao is None:
                resolucao = (frame.shape[1], frame.shape[0])  # (w, h)
        else:
            log("AVISO", f"  Frame {i+1}: tabuleiro perdido na re-deteção (ignorado).")

    if n_ok < MIN_FRAMES_CALIBRACAO:
        log("ERRO",
            f"Só {n_ok} frames com deteção válida — mínimo {MIN_FRAMES_CALIBRACAO}.")
        return None

    log("HUMANO", f"A calibrar com {n_ok} frames ({resolucao[0]}×{resolucao[1]}px)...")
    t0 = time.time()

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, resolucao,
        None, None,
        flags=(
            cv2.CALIB_RATIONAL_MODEL |     # 8 coefs distorção
            cv2.CALIB_THIN_PRISM_MODEL     # + prism (útil para lentes grande-angular)
        )
    )
    t_cal = time.time() - t0

    log("OK",
        f"Calibração concluída em {t_cal:.1f}s | RMS = {rms:.4f}px "
        f"(< 0.5 = excelente, < 1.0 = aceitável)")

    if rms > 2.0:
        log("AVISO",
            f"RMS elevado ({rms:.2f}px). Considera recapturar com mais poses variadas "
            f"e garantir que o tabuleiro não está parcialmente fora de frame.")

    # Calcular erro por frame para diagnóstico
    erros_por_frame = []
    for i in range(len(obj_points)):
        pts_proj, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(img_points[i], pts_proj, cv2.NORM_L2) / len(pts_proj)
        erros_por_frame.append(round(float(err), 4))

    resultado = {
        "K":              K.tolist(),
        "D":              D.tolist(),
        "resolucao":      list(resolucao),
        "rms_px":         round(float(rms), 5),
        "n_frames":       n_ok,
        "tamanho_quadrado_m": tamanho_m,
        "cols_cantos":    cols,
        "rows_cantos":    rows,
        "erro_por_frame": erros_por_frame,
        "data":           datetime.now().isoformat(timespec="seconds"),
        "perfil_camara":  "CALIBRADO",
    }

    return resultado


# ─────────────────────────────────────────────
#  GUARDAR E MOSTRAR RESULTADO
# ─────────────────────────────────────────────
def guardar_resultado(resultado: dict, caminho: Path):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w") as f:
        json.dump(resultado, f, indent=4)
    log("OK", f"Intrínsecos guardados: {caminho}")


def mostrar_resultado(resultado: dict):
    K = np.array(resultado["K"])
    D = np.array(resultado["D"])
    print()
    print("\033[1;96m" + "═" * 60 + "\033[0m")
    print("\033[1;97m  RESULTADO DA CALIBRAÇÃO INTRÍNSECA\033[0m")
    print("\033[1;96m" + "═" * 60 + "\033[0m")
    print(f"  RMS:             {resultado['rms_px']:.4f} px")
    print(f"  Frames usados:   {resultado['n_frames']}")
    print(f"  Resolução:       {resultado['resolucao'][0]}×{resultado['resolucao'][1]} px")
    print(f"  fx = {K[0,0]:.1f} px    fy = {K[1,1]:.1f} px")
    print(f"  cx = {K[0,2]:.1f} px    cy = {K[1,2]:.1f} px")
    print(f"  D  = {np.round(D.flatten()[:5], 5).tolist()}")
    print()
    print("  Para usar no pipeline, adiciona ao parametros.json:")
    print('    "perfil_camara": "CALIBRADO"')
    print("  E garante que parametros.py lê 'intrinsicos_camera.json'")
    print("  quando o perfil for 'CALIBRADO'.")
    print("\033[1;96m" + "═" * 60 + "\033[0m")
    print()


# ─────────────────────────────────────────────
#  PREVIEW DE UNDISTORT (verificação visual)
# ─────────────────────────────────────────────
def preview_undistort(frames: list[np.ndarray], K, D):
    """
    Mostra side-by-side: imagem original vs corrigida de distorção.
    Permite verificar visualmente se a calibração faz sentido.
    Tecla Q/ESC fecha; SPACE avança para o próximo frame.
    """
    if not frames:
        return
    log("HUMANO", "Preview de undistort — SPACE: próximo  ESC/Q: fechar")
    JANELA = "Undistort Preview  (original | corrigido)"
    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)

    for frame in frames:
        h, w = frame.shape[:2]
        K_np = np.array(K)
        D_np = np.array(D)
        newK, roi = cv2.getOptimalNewCameraMatrix(K_np, D_np, (w, h), 1, (w, h))
        undist = cv2.undistort(frame, K_np, D_np, None, newK)

        # Linha de separação
        sep = np.zeros((h, 4, 3), dtype=np.uint8)
        sep[:] = (0, 200, 255)

        comp = np.concatenate([frame, sep, undist], axis=1)
        scale = min(1.0, 1400 / comp.shape[1])
        comp_small = cv2.resize(comp,
                                (int(comp.shape[1]*scale), int(comp.shape[0]*scale)))
        cv2.imshow(JANELA, comp_small)

        tecla = cv2.waitKey(0) & 0xFF
        if tecla in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  PATCH AO parametros.py — perfil CALIBRADO
# ─────────────────────────────────────────────
PATCH_PARAMETROS = '''
    # ── Perfil CALIBRADO (gerado por calibracao_camera.py) ──────────
    elif perfil == "CALIBRADO":
        import json as _json
        from pathlib import Path as _Path
        _calib_path = _Path(__file__).parent / "resultados" / "calibracao" / "intrinsicos_camera.json"
        if not _calib_path.exists():
            raise FileNotFoundError(
                f"Intrínsecos calibrados não encontrados: {_calib_path}\\n"
                "Executa calibracao_camera.py primeiro."
            )
        with open(_calib_path) as _f:
            _ci = _json.load(_f)
        K = np.array(_ci["K"])
        D = np.array(_ci["D"])
        res = tuple(_ci["resolucao"])
'''


def verificar_patch_parametros():
    """
    Verifica se parametros.py já tem suporte ao perfil CALIBRADO.
    Se não, mostra as instruções de patch.
    """
    fich = BASE_PATH / "_CONFIG" / "system_parameters.py"
    if not fich.exists():
        log("AVISO", "_CONFIG/system_parameters.py não encontrado.")
        return

    conteudo = fich.read_text(encoding="utf-8")
    if "CALIBRADO" in conteudo:
        log("OK", "parametros.py já tem suporte ao perfil CALIBRADO.")
        return

    log("AVISO", "parametros.py NÃO tem suporte ao perfil CALIBRADO ainda.")
    log("HUMANO", "Adiciona o seguinte bloco no elif da função obter_intrinsics():")
    print(PATCH_PARAMETROS)


# ─────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Calibração intrínseca da câmara por tabuleiro de xadrez",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--capturar", action="store_true",
                        help="Captura frames ao vivo com a câmara")
    parser.add_argument("--ficheiros", type=str, metavar="PASTA",
                        help="Lê frames de uma pasta de ficheiros de imagem")
    parser.add_argument("--cols",    type=int, default=COLS,
                        help=f"Cantos interiores horizontais (default={COLS})")
    parser.add_argument("--rows",    type=int, default=ROWS,
                        help=f"Cantos interiores verticais (default={ROWS})")
    parser.add_argument("--quadrado", type=float, default=TAMANHO_QUADRADO_M,
                        help=f"Tamanho do quadrado em metros (default={TAMANHO_QUADRADO_M})")
    parser.add_argument("--preview", action="store_true",
                        help="Mostra preview de undistort após calibração")
    parser.add_argument("--saida", type=str, default=str(SAIDA_JSON),
                        help=f"Caminho do JSON de saída (default={SAIDA_JSON})")
    args = parser.parse_args()

    cols      = args.cols
    rows      = args.rows
    tamanho_m = args.quadrado
    saida     = Path(args.saida)

    print()
    print("\033[1;96m" + "═" * 60 + "\033[0m")
    print("\033[1;97m  CALIBRAÇÃO INTRÍNSECA — TABULEIRO DE XADREZ\033[0m")
    print(f"  Tabuleiro: {cols}×{rows} cantos interiores")
    print(f"  Quadrado:  {tamanho_m*100:.1f} cm")
    print(f"  Mínimo:    {MIN_FRAMES_CALIBRACAO} frames com deteção")
    print("\033[1;96m" + "═" * 60 + "\033[0m")
    print()

    # ── Adquirir frames ──────────────────────────────────────────
    frames = []

    if args.ficheiros:
        pasta = Path(args.ficheiros)
        if not pasta.exists():
            log("ERRO", f"Pasta não encontrada: {pasta}")
            sys.exit(1)
        frames = carregar_frames_de_pasta(pasta, cols, rows)

    else:
        # Modo padrão: câmara ao vivo
        frames = modo_captura_ao_vivo(cols, rows)

    if len(frames) < MIN_FRAMES_CALIBRACAO:
        log("ERRO",
            f"Frames insuficientes ({len(frames)}). "
            f"São necessários pelo menos {MIN_FRAMES_CALIBRACAO}.")
        sys.exit(1)

    # ── Guardar frames capturados para referência ────────────────
    PASTA_FRAMES_CAPTURADOS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, fr in enumerate(frames):
        cv2.imwrite(str(PASTA_FRAMES_CAPTURADOS / f"frame_{ts}_{i+1:02d}.jpg"), fr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    log("DEBUG", f"Frames guardados em {PASTA_FRAMES_CAPTURADOS.name}/")

    # ── Calibrar ─────────────────────────────────────────────────
    resultado = calibrar_intrinsicos(frames, cols, rows, tamanho_m)

    if resultado is None:
        log("ERRO", "Calibração falhou. Verifica os frames e tenta novamente.")
        sys.exit(1)

    # ── Guardar e mostrar ────────────────────────────────────────
    guardar_resultado(resultado, saida)
    mostrar_resultado(resultado)

    # ── Preview de undistort (opcional) ─────────────────────────
    if args.preview:
        preview_undistort(frames[:5], resultado["K"], resultado["D"])

    # ── Verificar patch do parametros.py ────────────────────────
    verificar_patch_parametros()

    log("OK", "Calibração intrínseca concluída!")
    log("HUMANO",
        "Próximo passo: define 'perfil_camara': 'CALIBRADO' no parametros.json "
        "e recalibra a homografia (MasterControl opção 2).")


if __name__ == "__main__":
    main()





