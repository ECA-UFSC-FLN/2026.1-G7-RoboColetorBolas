"""
retificador.py  –  Servidor de Retificação por Homografia
==========================================================
Fluxo geral:
  1. Na primeira execução (ou via --calibrar), o utilizador calibra a homografia:
       - Seleciona a câmara
       - Abre uma imagem de referência
       - Clica N pontos e introduz as coordenadas reais em metros
       - A matriz H e os parâmetros são guardados em 'homografia_calibracao.json'
  2. Em modo servidor (fluxo normal):
       - Aguarda ligação do VisionProcessing via socket IPC (porta 6001)
       - Recebe pacote { frame, bolas_px, indice, timestamp_visao }
       - Aplica undistort + homografia a cada coordenada de bola
       - Guarda coordenadas reais em JSON  →  Homografia/Coordenadas retificadas/
       - Guarda imagem retificada          →  Homografia/Imagem Retificada/
       - Envia "LIBERADO" de volta ao VisionProcessing
"""

import cv2
import numpy as np
import os
import sys
import json
import time
import argparse
from pathlib import Path
from multiprocessing.connection import Listener

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DE PASTAS
# ---------------------------------------------------------------------------
BASE_DIR         = Path(__file__).parent
PASTA_COORDS     = BASE_DIR / "Homografia" / "Coordenadas retificadas"
PASTA_IMAGENS    = BASE_DIR / "Homografia" / "Imagem Retificada"
FICHEIRO_CALIB   = BASE_DIR / "homografia_calibracao.json"

# ---------------------------------------------------------------------------
# PARÂMETROS INTRÍNSECOS – iPhone 16
# ---------------------------------------------------------------------------
IPHONE16_W = 4032
IPHONE16_H = 3024

IPHONE16_INTRINSICS = {
    "fx": 5823.0, "fy": 5823.0,
    "cx": 2016.0, "cy": 1512.0,
    "k1":  0.1220, "k2": -0.2460, "k3": 0.1760,
    "p1":  0.0001, "p2": -0.0002,
}

# ---------------------------------------------------------------------------
# UTILITÁRIOS DE INPUT
# ---------------------------------------------------------------------------
def pedir_float(mensagem: str, padrao=None) -> float:
    sufixo = f" [{padrao}]: " if padrao is not None else ": "
    while True:
        texto = input(mensagem + sufixo).strip()
        if texto == "" and padrao is not None:
            return float(padrao)
        try:
            return float(texto)
        except ValueError:
            print("  ✗ Valor inválido. Introduza um número.")


# ---------------------------------------------------------------------------
# CÂMARA / UNDISTORT
# ---------------------------------------------------------------------------
def construir_camera_matrix(p: dict):
    K = np.array([
        [p["fx"],    0.0,   p["cx"]],
        [   0.0,  p["fy"], p["cy"]],
        [   0.0,     0.0,     1.0 ],
    ], dtype=np.float64)
    D = np.array([p["k1"], p["k2"], p["p1"], p["p2"], p["k3"]], dtype=np.float64)
    return K, D


def aplicar_undistort(img: np.ndarray, p: dict) -> np.ndarray:
    K, D = construir_camera_matrix(p)
    return cv2.undistort(img, K, D)


def recolher_params_camara() -> dict:
    print("\n  Escolha o modo de câmara:")
    print("  [1]  iPhone 16 – parâmetros automáticos")
    print("  [2]  Câmara configurável – introdução manual")
    while True:
        opc = input("\n  Opção (1 ou 2): ").strip()
        if opc == "1":
            p = dict(IPHONE16_INTRINSICS)
            p["img_w"], p["img_h"] = IPHONE16_W, IPHONE16_H
            return p
        if opc == "2":
            img_w = int(pedir_float("  Largura da imagem (px)"))
            img_h = int(pedir_float("  Altura da imagem (px)"))
            fx    = pedir_float("  fx (px)")
            fy    = pedir_float("  fy (px)", padrao=fx)
            cx    = pedir_float("  cx (px)", padrao=img_w / 2)
            cy    = pedir_float("  cy (px)", padrao=img_h / 2)
            k1    = pedir_float("  k1", padrao=0.0)
            k2    = pedir_float("  k2", padrao=0.0)
            k3    = pedir_float("  k3", padrao=0.0)
            p1    = pedir_float("  p1", padrao=0.0)
            p2    = pedir_float("  p2", padrao=0.0)
            return {"img_w": img_w, "img_h": img_h,
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2}
        print("  ✗ Introduza 1 ou 2.")


# ---------------------------------------------------------------------------
# SELEÇÃO INTERATIVA DE PONTOS (mantida do original)
# ---------------------------------------------------------------------------
CORES  = [(0,255,0),(0,128,255),(255,0,255),(0,255,255),
          (255,0,0),(0,0,255),(255,255,0),(128,0,255),(0,200,100),(200,100,0)]
LABELS = ["A","B","C","D","E","F","G","H","I","J"]
_estado = {}


def _redesenhar_pontos():
    img_base = _estado["img_display_limpa"].copy()
    for i, (px, py) in enumerate(_estado["pontos_img"]):
        escala     = _estado["escala"]
        barra_h_px = _estado["barra_h_px"]
        sx = int(px * escala)
        sy = int(py * escala) + barra_h_px
        cor   = CORES[i % len(CORES)]
        label = LABELS[i]
        cv2.circle(img_base, (sx, sy), 9, (255,255,255), 2)
        cv2.circle(img_base, (sx, sy), 7, cor, -1)
        cv2.putText(img_base, label, (sx+12, sy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor, 2, cv2.LINE_AA)
    _estado["img_display"] = img_base


def _atualizar_barra(mensagem: str):
    barra_h_px = _estado["barra_h_px"]
    largura    = _estado["img_display"].shape[1]
    barra = np.full((barra_h_px, largura, 3), 30, dtype=np.uint8)
    cv2.putText(barra, mensagem, (10,26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1, cv2.LINE_AA)
    _estado["img_display"][:barra_h_px] = barra


def _callback_rato(evento, x, y, flags, param):
    if _estado.get("concluido"):
        return
    if evento == cv2.EVENT_LBUTTONDOWN:
        n_atual = len(_estado["pontos_img"])
        if n_atual >= _estado["n_pontos"]:
            return
        escala     = _estado["escala"]
        barra_h_px = _estado["barra_h_px"]
        x_orig = int(x / escala)
        y_orig = max(0, int((y - barra_h_px) / escala))
        _estado["pontos_img"].append((x_orig, y_orig))
        cor   = CORES[n_atual % len(CORES)]
        label = LABELS[n_atual]
        cv2.circle(_estado["img_display"], (x,y), 9, (255,255,255), 2)
        cv2.circle(_estado["img_display"], (x,y), 7, cor, -1)
        cv2.putText(_estado["img_display"], label, (x+12, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor, 2, cv2.LINE_AA)
        n_atual += 1
        restam = _estado["n_pontos"] - n_atual
        if restam > 0:
            msg = (f"Ponto {label} OK  |  Clique {LABELS[n_atual]} "
                   f"({n_atual}/{_estado['n_pontos']})  |  Z=desfazer  R=recomeçar  ESC=cancelar")
            _atualizar_barra(msg)
            cv2.imshow("Selecionar Pontos", _estado["img_display"])
            print(f"  ✓ Ponto {label} ({x_orig},{y_orig})px → clique {LABELS[n_atual]}")
        else:
            msg = (f"Todos os {_estado['n_pontos']} pontos OK  |  "
                   f"ENTER=confirmar  Z=desfazer  R=recomeçar  ESC=cancelar")
            _atualizar_barra(msg)
            cv2.imshow("Selecionar Pontos", _estado["img_display"])
            print(f"  ✓ Ponto {label} ({x_orig},{y_orig})px")
            print("\n  Todos os pontos selecionados. Prima ENTER para continuar.")
            _estado["concluido"] = True


def selecionar_pontos(img_undist: np.ndarray, n_pontos: int) -> list:
    MAX_DIM = 1100
    h, w    = img_undist.shape[:2]
    escala  = min(1.0, MAX_DIM / max(h, w))
    img_vis = cv2.resize(img_undist, (int(w*escala), int(h*escala)))
    BARRA_H = 38
    barra   = np.full((BARRA_H, img_vis.shape[1], 3), 30, dtype=np.uint8)
    msg0    = f"Clique o ponto A  (0/{n_pontos})  |  Z=desfazer  R=recomeçar  ESC=cancelar"
    cv2.putText(barra, msg0, (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1, cv2.LINE_AA)
    img_display = np.vstack([barra, img_vis])
    _estado.clear()
    _estado.update({
        "pontos_img": [], "n_pontos": n_pontos,
        "img_display": img_display,
        "img_display_limpa": img_display.copy(),
        "escala": escala, "barra_h_px": BARRA_H, "concluido": False,
    })
    cv2.namedWindow("Selecionar Pontos", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Selecionar Pontos", _callback_rato)
    cv2.imshow("Selecionar Pontos", img_display)
    print(f"  Clique o ponto A...   Z=desfazer | R=recomeçar | ESC=cancelar")
    while True:
        tecla = cv2.waitKey(50) & 0xFF
        if _estado["concluido"] and tecla == 13:
            break
        if tecla == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
        if tecla in (ord('z'), ord('Z')) and _estado["pontos_img"]:
            _estado["pontos_img"].pop()
            _estado["concluido"] = False
            _redesenhar_pontos()
            n = len(_estado["pontos_img"])
            _atualizar_barra(f"Desfez  |  Clique {LABELS[n]} ({n}/{n_pontos})")
            cv2.imshow("Selecionar Pontos", _estado["img_display"])
        if tecla in (ord('r'), ord('R')) and _estado["pontos_img"]:
            _estado["pontos_img"].clear()
            _estado["concluido"] = False
            _estado["img_display"] = _estado["img_display_limpa"].copy()
            _atualizar_barra(msg0)
            cv2.imshow("Selecionar Pontos", _estado["img_display"])
    cv2.destroyAllWindows()
    return list(_estado["pontos_img"])


def recolher_coordenadas_reais(n_pontos: int) -> list:
    print("\n" + "─"*60)
    print("  COORDENADAS REAIS  |  Origem (0,0) = canto sup. esquerdo")
    print("  X → direita        |  Y → profundidade (para baixo)  [metros]")
    print("─"*60)
    coords = []
    i = 0
    while i < n_pontos:
        label  = LABELS[i]
        voltou = False
        print(f"\n  Ponto {label}  ({i+1}/{n_pontos}):")
        while True:
            t = input("    x real (m)  [ou 'v' para voltar]: ").strip()
            if t.lower() in ("v","voltar"):
                if i > 0:
                    coords.pop(); i -= 1
                    print(f"  ↩  Voltou ao ponto {LABELS[i]}.")
                else:
                    print("  ✗ Já está no primeiro ponto.")
                voltou = True; break
            try:
                x_r = float(t); break
            except ValueError:
                print("  ✗ Valor inválido.")
        if voltou:
            continue
        while True:
            t2 = input("    y real (m)  [ou 'v' para voltar]: ").strip()
            if t2.lower() in ("v","voltar"):
                print(f"  ↩  Reintroduz o ponto {label}.")
                voltou = True; break
            try:
                y_r = float(t2); break
            except ValueError:
                print("  ✗ Valor inválido.")
        if voltou:
            continue
        coords.append((x_r, y_r))
        i += 1
    return coords


# ---------------------------------------------------------------------------
# CALIBRAÇÃO – calcula H e guarda em JSON
# ---------------------------------------------------------------------------
def calibrar_e_guardar() -> dict:
    """Fluxo interativo de calibração. Devolve o dict de calibração."""
    print("\n" + "═"*60)
    print("  CALIBRAÇÃO DA HOMOGRAFIA")
    print("  Selecione uma imagem de referência da quadra,")
    print("  clique os pontos de referência e introduza as")
    print("  respetivas coordenadas reais em metros.")
    print("═"*60)

    params = recolher_params_camara()

    while True:
        caminho = input("\nCaminho da imagem de referência: ").strip().strip('"').strip("'")
        if os.path.isfile(caminho):
            break
        print(f"  ✗ Ficheiro não encontrado: '{caminho}'")

    img = cv2.imread(caminho)
    if img is None:
        print("  ✗ Não foi possível ler a imagem.")
        sys.exit(1)

    h_img, w_img = img.shape[:2]
    if w_img != params["img_w"] or h_img != params["img_h"]:
        print(f"  ⚠  Redimensionando imagem de ({w_img}x{h_img}) "
              f"para ({params['img_w']}x{params['img_h']}).")
        img = cv2.resize(img, (params["img_w"], params["img_h"]))

    print("\n  A corrigir distorção da lente...")
    img_undist = aplicar_undistort(img, params)

    # Número de pontos
    while True:
        try:
            n = int(input("\n  Número de pontos de referência (4/6/8/10): ").strip())
            if n in (4,6,8,10):
                break
        except ValueError:
            pass
        print("  ✗ Introduza 4, 6, 8 ou 10.")

    print(f"\n  Vai abrir a imagem. Clique {n} pontos de referência.")
    input("  Prima ENTER para continuar...")
    pts_img   = selecionar_pontos(img_undist, n)
    pts_reais = recolher_coordenadas_reais(n)

    # Calcular H para guardar
    xs, ys  = [p[0] for p in pts_reais], [p[1] for p in pts_reais]
    x_min, y_min = min(xs), min(ys)
    W_real = max(xs) - x_min
    D_real = max(ys) - y_min

    h_in, w_in = img_undist.shape[:2]
    ppm = max(w_in, h_in) / max(W_real, D_real) if max(W_real, D_real) > 0 else 200.0
    pts_dst = np.array([((p[0]-x_min)*ppm, (p[1]-y_min)*ppm) for p in pts_reais], dtype=np.float32)
    pts_src = np.array(pts_img, dtype=np.float32)

    if len(pts_img) == 4:
        H_mat, _ = cv2.findHomography(pts_src, pts_dst)
    else:
        H_mat, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransacReprojThreshold=5.0)

    if H_mat is None:
        print("  ✗ Não foi possível calcular a homografia. Pontos colineares?")
        sys.exit(1)

    calib = {
        "params_camara": params,
        "pts_img":        pts_img,
        "pts_reais":      pts_reais,
        "H_mat":          H_mat.tolist(),
        "ppm":            ppm,
        "x_min":          x_min,
        "y_min":          y_min,
        "W_real":         W_real,
        "D_real":         D_real,
        "out_w":          int(W_real * ppm),
        "out_h":          int(D_real * ppm),
    }

    with open(FICHEIRO_CALIB, "w") as f:
        json.dump(calib, f, indent=4)

    print(f"\n  ✓ Calibração guardada em: {FICHEIRO_CALIB}")
    print(f"  ✓ Área calibrada: {W_real:.2f} m × {D_real:.2f} m")
    print(f"  ✓ Resolução: {ppm:.0f} px/m")
    return calib


def carregar_calibracao() -> dict:
    """Carrega calibração de ficheiro JSON ou executa o fluxo de calibração."""
    if not FICHEIRO_CALIB.exists():
        print(f"  ⚠  Ficheiro de calibração não encontrado: {FICHEIRO_CALIB}")
        print("  A iniciar fluxo de calibração...")
        return calibrar_e_guardar()

    with open(FICHEIRO_CALIB) as f:
        calib = json.load(f)

    calib["H_mat"] = np.array(calib["H_mat"], dtype=np.float64)
    print(f"  ✓ Calibração carregada de: {FICHEIRO_CALIB}")
    print(f"  ✓ Área: {calib['W_real']:.2f} m × {calib['D_real']:.2f} m  "
          f"| {calib['ppm']:.0f} px/m")
    return calib


# ---------------------------------------------------------------------------
# TRANSFORMAÇÃO DE COORDENADAS  pixel → metros
# ---------------------------------------------------------------------------
def pixel_para_real(px: float, py: float, H_mat: np.ndarray,
                    ppm: float, x_min: float, y_min: float) -> tuple:
    """
    Aplica a homografia H a um ponto (px, py) em pixels e devolve
    (x_m, y_m) em metros no referencial real.
    """
    pt = np.array([[[px, py]]], dtype=np.float32)
    pt_ret = cv2.perspectiveTransform(pt, H_mat)
    x_ret_px = float(pt_ret[0][0][0])
    y_ret_px = float(pt_ret[0][0][1])
    x_m = x_ret_px / ppm + x_min
    y_m = y_ret_px / ppm + y_min
    return round(x_m, 4), round(y_m, 4)


# ---------------------------------------------------------------------------
# SERVIDOR IPC  (porta 6001)
# ---------------------------------------------------------------------------
def servidor_retificacao(calib: dict):
    """Loop principal: aguarda pacotes do VisionProcessing e retifica."""
    PASTA_COORDS.mkdir(parents=True, exist_ok=True)
    PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)

    H_mat  = calib["H_mat"]
    ppm    = calib["ppm"]
    x_min  = calib["x_min"]
    y_min  = calib["y_min"]
    out_w  = calib["out_w"]
    out_h  = calib["out_h"]
    params = calib["params_camara"]

    address  = ('localhost', 6001)
    listener = Listener(address, authkey=b'retificador_ufsc')

    print("\n" + "═"*60)
    print("  SERVIDOR DE RETIFICAÇÃO ATIVO  –  porta 6001")
    print("  Aguardando ligação do VisionProcessing...")
    print("═"*60)

    indice = 0
    while True:
        conn = listener.accept()
        print(f"\n[ETAPA] Ligação estabelecida com o VisionProcessing.")

        try:
            # ── Receber pacote ─────────────────────────────────────────────
            t_recv = time.time()
            pacote = conn.recv()

            timestamp_visao = pacote["timestamp_visao"]
            frame           = pacote["frame"]
            bolas_px        = pacote["bolas_px"]   # lista de {"x1","y1","x2","y2"}
            idx_visao       = pacote["indice"]

            print(f"[ETAPA] Pacote #{idx_visao} recebido  "
                  f"| {len(bolas_px)} bola(s) detetada(s).")

            # ── Timer: início do processamento ─────────────────────────────
            t_proc_inicio = time.time()

            # ── Undistort do frame ─────────────────────────────────────────
            img_undist = aplicar_undistort(frame, params)

            # ── Warp da imagem completa ─────────────────────────────────────
            img_ret = cv2.warpPerspective(img_undist, H_mat, (out_w, out_h))

            # ── Converter coordenadas pixel → metros ───────────────────────
            bolas_reais = []
            for b in bolas_px:
                # Centro da bounding box em pixels (coordenadas undistorted)
                cx_px = (b["x1"] + b["x2"]) / 2.0
                cy_px = (b["y1"] + b["y2"]) / 2.0
                x_m, y_m = pixel_para_real(cx_px, cy_px, H_mat, ppm, x_min, y_min)

                # Centro projetado na imagem retificada (para visualização)
                cx_ret = (cx_px * H_mat[0,0] + cy_px * H_mat[0,1] + H_mat[0,2])
                cy_ret = (cx_px * H_mat[1,0] + cy_px * H_mat[1,1] + H_mat[1,2])
                w_ret  = (cx_px * H_mat[2,0] + cy_px * H_mat[2,1] + H_mat[2,2])
                if w_ret != 0:
                    cx_ret = int(cx_ret / w_ret)
                    cy_ret = int(cy_ret / w_ret)
                else:
                    cx_ret, cy_ret = 0, 0

                bolas_reais.append({
                    "centro_px_original": {"x": round(cx_px, 1), "y": round(cy_px, 1)},
                    "centro_real_m":      {"x": x_m, "y": y_m},
                    "bbox_px":            b,
                })

                # Desenha ponto na imagem retificada
                cv2.circle(img_ret, (cx_ret, cy_ret), 8, (0,255,0), -1)
                cv2.circle(img_ret, (cx_ret, cy_ret), 10, (255,255,255), 2)
                cv2.putText(img_ret,
                            f"({x_m:.2f}m, {y_m:.2f}m)",
                            (cx_ret + 12, cy_ret - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)

            # ── Timers ─────────────────────────────────────────────────────
            t_proc_fim      = time.time()
            latencia_total  = (t_proc_fim - timestamp_visao) * 1000   # ms (ponta-a-ponta)
            latencia_ret    = (t_proc_fim - t_proc_inicio)   * 1000   # ms (só retificação)

            # ── Guardar JSON ────────────────────────────────────────────────
            nome_json = f"coordenadas_ret_{indice:04d}.json"
            saida_json = {
                "indice":              indice,
                "indice_visao":        idx_visao,
                "latencia_total_ms":   round(latencia_total, 2),
                "latencia_ret_ms":     round(latencia_ret, 2),
                "area_m":              {"largura": calib["W_real"], "profundidade": calib["D_real"]},
                "bolas":               bolas_reais,
            }
            with open(PASTA_COORDS / nome_json, "w") as f:
                json.dump(saida_json, f, indent=4)

            # ── Guardar imagem retificada ───────────────────────────────────
            nome_img = f"imagem_ret_{indice:04d}.jpg"
            cv2.imwrite(str(PASTA_IMAGENS / nome_img), img_ret,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            print(f"[ETAPA]  JSON  → {PASTA_COORDS / nome_json}")
            print(f"[ETAPA]  IMG   → {PASTA_IMAGENS / nome_img}")
            print(f"[SUCESSO] Latência total: {latencia_total:.1f} ms  "
                  f"| Retificação: {latencia_ret:.1f} ms  "
                  f"| Bolas: {len(bolas_reais)}")

            # ── Sinal de libertação ─────────────────────────────────────────
            conn.send("LIBERADO")
            indice += 1

        except Exception as e:
            print(f"[ERRO] Falha no processamento: {e}")
            try:
                conn.send("LIBERADO")   # liberta mesmo em caso de erro
            except Exception:
                pass
        finally:
            conn.close()
            print("[ESTADO] Retificador resetado. Aguardando próxima captura...")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Servidor de Retificação por Homografia")
    parser.add_argument("--calibrar", action="store_true",
                        help="Forçar nova calibração mesmo que já exista uma guardada.")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  RETIFICADOR – Conversão Pixel → Coordenadas Reais")
    print("═"*60)

    if args.calibrar:
        calib = calibrar_e_guardar()
    else:
        calib = carregar_calibracao()

    servidor_retificacao(calib)


if __name__ == "__main__":
    main()
