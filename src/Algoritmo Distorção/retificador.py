"""

Fluxo:
  1. Escolha do modo de câmara (iPhone 16 ou configurável)
  2. Carregamento da imagem
  3. Undistort (correção de lente)
  4. Utilizador clica N pontos na imagem (N ∈ {4, 6, 8, 10})
  5. Para cada ponto, introduz a coordenada real (x, y) em metros
  6. Homografia calculada empiricamente → warpPerspective
  7. Imagem retificada mostrada e guardada
"""

import cv2
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# PARÂMETROS INTRÍNSECOS – iPhone 16
# ---------------------------------------------------------------------------
IPHONE16_W = 4032
IPHONE16_H = 3024

IPHONE16_INTRINSICS = {
    "fx": 5823.0,
    "fy": 5823.0,
    "cx": 2016.0,
    "cy": 1512.0,
    "k1":  0.1220,
    "k2": -0.2460,
    "k3":  0.1760,
    "p1":  0.0001,
    "p2": -0.0002,
}


# ===========================================================================
# UTILIDADES DE INPUT
# ===========================================================================

def pedir_float(mensagem: str, padrao: float | None = None) -> float:
    sufixo = f" [{padrao}]: " if padrao is not None else ": "
    while True:
        texto = input(mensagem + sufixo).strip()
        if texto == "" and padrao is not None:
            return float(padrao)
        try:
            return float(texto)
        except ValueError:
            print("  ✗ Valor inválido. Introduza um número.")


def pedir_int_opcoes(mensagem: str, opcoes: list) -> int:
    opts_str = "/".join(str(o) for o in opcoes)
    while True:
        texto = input(f"{mensagem} ({opts_str}): ").strip()
        try:
            val = int(texto)
            if val in opcoes:
                return val
        except ValueError:
            pass
        print(f"  ✗ Introduza um dos seguintes valores: {opts_str}")


def pedir_caminho_imagem() -> str:
    while True:
        caminho = input("\nCaminho da imagem: ").strip().strip('"').strip("'")
        if os.path.isfile(caminho):
            return caminho
        print(f"  ✗ Ficheiro não encontrado: '{caminho}'. Tente novamente.")


# ===========================================================================
# RECOLHA DE PARÂMETROS DA CÂMARA
# ===========================================================================

def recolher_parametros_iphone16() -> dict:
    print("\n" + "─" * 60)
    print("  MODO: iPhone 16 (câmara principal)")
    print("─" * 60)
    print("  Intrínsecos carregados automaticamente.\n")
    params = dict(IPHONE16_INTRINSICS)
    params["img_w"] = IPHONE16_W
    params["img_h"] = IPHONE16_H
    return params


def recolher_parametros_configuravel() -> dict:
    print("\n" + "─" * 60)
    print("  MODO: Câmara Configurável")
    print("─" * 60)

    print("\n  [ Resolução do sensor ]")
    img_w = int(pedir_float("  Largura da imagem (px)"))
    img_h = int(pedir_float("  Altura da imagem (px)"))

    print("\n  [ Parâmetros intrínsecos ]")
    fx = pedir_float("  fx – distância focal horizontal (px)")
    fy = pedir_float("  fy – distância focal vertical   (px)", padrao=fx)
    cx = pedir_float("  cx – ponto principal X (px)", padrao=img_w / 2)
    cy = pedir_float("  cy – ponto principal Y (px)", padrao=img_h / 2)

    print("\n  [ Coeficientes de distorção ]")
    k1 = pedir_float("  k1 (distorção radial 1)",     padrao=0.0)
    k2 = pedir_float("  k2 (distorção radial 2)",     padrao=0.0)
    k3 = pedir_float("  k3 (distorção radial 3)",     padrao=0.0)
    p1 = pedir_float("  p1 (distorção tangencial 1)", padrao=0.0)
    p2 = pedir_float("  p2 (distorção tangencial 2)", padrao=0.0)

    return {
        "img_w": img_w, "img_h": img_h,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2,
    }


# ===========================================================================
# CORREÇÃO DE DISTORÇÃO DE LENTE
# ===========================================================================

def construir_camera_matrix(p: dict) -> tuple:
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


# ===========================================================================
# SELEÇÃO INTERATIVA DE PONTOS
# ===========================================================================

CORES = [
    (  0, 255,   0),  # verde
    (  0, 128, 255),  # laranja
    (255,   0, 255),  # magenta
    (  0, 255, 255),  # amarelo
    (255,   0,   0),  # azul
    (  0,   0, 255),  # vermelho
    (255, 255,   0),  # ciano
    (128,   0, 255),  # roxo
    (  0, 200, 100),
    (200, 100,   0),
]
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Estado global para o callback do rato
_estado = {}


def _redesenhar_pontos():
    """Reconstrói img_display com os pontos atuais a partir da imagem limpa."""
    img_base = _estado["img_display_limpa"].copy()
    for i, (px, py) in enumerate(_estado["pontos_img"]):
        # Converte de volta para coordenadas de ecrã
        escala     = _estado["escala"]
        barra_h_px = _estado["barra_h_px"]
        sx = int(px * escala)
        sy = int(py * escala) + barra_h_px
        cor   = CORES[i % len(CORES)]
        label = LABELS[i]
        cv2.circle(img_base, (sx, sy), 9, (255, 255, 255), 2)
        cv2.circle(img_base, (sx, sy), 7, cor, -1)
        cv2.putText(
            img_base, label,
            (sx + 12, sy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor, 2, cv2.LINE_AA,
        )
    _estado["img_display"] = img_base


def _atualizar_barra(mensagem: str):
    """Atualiza o texto da barra de instrução no topo."""
    barra_h_px = _estado["barra_h_px"]
    largura    = _estado["img_display"].shape[1]
    barra = np.full((barra_h_px, largura, 3), 30, dtype=np.uint8)
    cv2.putText(
        barra, mensagem,
        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA,
    )
    _estado["img_display"][:barra_h_px] = barra


def _callback_rato(evento, x, y, flags, param):
    if _estado.get("concluido"):
        return

    if evento == cv2.EVENT_LBUTTONDOWN:
        n_atual = len(_estado["pontos_img"])
        if n_atual >= _estado["n_pontos"]:
            return

        # Guarda ponto em coordenadas da imagem original (desfaz escala + offset barra)
        escala     = _estado["escala"]
        barra_h_px = _estado["barra_h_px"]
        x_orig = int(x / escala)
        y_orig = int((y - barra_h_px) / escala)
        y_orig = max(0, y_orig)
        _estado["pontos_img"].append((x_orig, y_orig))

        # Desenha marcador colorido
        cor   = CORES[n_atual % len(CORES)]
        label = LABELS[n_atual]
        cv2.circle(_estado["img_display"], (x, y), 9, (255, 255, 255), 2)
        cv2.circle(_estado["img_display"], (x, y), 7, cor, -1)
        cv2.putText(
            _estado["img_display"], label,
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor, 2, cv2.LINE_AA,
        )

        n_atual += 1
        restam   = _estado["n_pontos"] - n_atual
        if restam > 0:
            proximo_label = LABELS[n_atual]
            msg = (f"Ponto {label} OK  |  Clique {proximo_label} "
                   f"({n_atual}/{_estado['n_pontos']})  |  Z=desfazer  R=recomeçar  ESC=cancelar")
            _atualizar_barra(msg)
            cv2.imshow("Selecionar Pontos", _estado["img_display"])
            print(f"  ✓ Ponto {label} ({x_orig}, {y_orig}) px  "
                  f"→  clique o ponto {proximo_label}  (faltam {restam})")
        else:
            msg = (f"Todos os {_estado['n_pontos']} pontos OK  |  "
                   f"ENTER=confirmar  Z=desfazer último  R=recomeçar  ESC=cancelar")
            _atualizar_barra(msg)
            cv2.imshow("Selecionar Pontos", _estado["img_display"])
            print(f"  ✓ Ponto {label} ({x_orig}, {y_orig}) px")
            print("\n  Todos os pontos selecionados. Prima ENTER para continuar.")
            _estado["concluido"] = True


def selecionar_pontos(img_undist: np.ndarray, n_pontos: int) -> list:
    """Abre janela interativa; devolve lista de (x,y) em coords da imagem original.

    Controlos:
      Clique esquerdo - adiciona ponto
      Z               - desfaz o ultimo ponto
      R               - recomeça (apaga todos os pontos)
      ENTER           - confirma (so apos todos os pontos selecionados)
      ESC             - cancela e sai
    """
    MAX_DIM  = 1100
    h, w     = img_undist.shape[:2]
    escala   = min(1.0, MAX_DIM / max(h, w))
    img_vis  = cv2.resize(img_undist, (int(w * escala), int(h * escala)))

    # Barra de instrucao no topo
    BARRA_H  = 38
    barra    = np.full((BARRA_H, img_vis.shape[1], 3), 30, dtype=np.uint8)
    msg_inicial = f"Clique o ponto A  (0/{n_pontos})  |  Z=desfazer  R=recomecar  ESC=cancelar"
    cv2.putText(
        barra, msg_inicial,
        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA,
    )
    img_display = np.vstack([barra, img_vis])
    # Copia limpa sem marcadores para poder redesenhar ao desfazer
    img_display_limpa = img_display.copy()

    _estado.clear()
    _estado.update({
        "pontos_img":        [],
        "n_pontos":          n_pontos,
        "img_display":       img_display,
        "img_display_limpa": img_display_limpa,
        "escala":            escala,
        "barra_h_px":        BARRA_H,
        "concluido":         False,
    })

    cv2.namedWindow("Selecionar Pontos", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Selecionar Pontos", _callback_rato)
    cv2.imshow("Selecionar Pontos", img_display)
    print(f"  Clique o ponto A na janela da imagem...")
    print(f"  Z = desfazer ultimo ponto  |  R = recomecar  |  ESC = cancelar")

    while True:
        tecla = cv2.waitKey(50) & 0xFF

        if _estado["concluido"] and tecla == 13:    # ENTER – confirmar
            break

        if tecla == 27:                              # ESC – cancelar
            print("\n  ✗ Operacao cancelada.")
            cv2.destroyAllWindows()
            sys.exit(0)

        if tecla in (ord('z'), ord('Z')):            # Z – desfazer ultimo ponto
            if _estado["pontos_img"]:
                removido = _estado["pontos_img"].pop()
                _estado["concluido"] = False
                _redesenhar_pontos()
                n_atual = len(_estado["pontos_img"])
                proximo_label = LABELS[n_atual]
                msg = (f"Ponto {LABELS[n_atual]} removido  |  Clique {proximo_label} "
                       f"({n_atual}/{n_pontos})  |  Z=desfazer  R=recomecar  ESC=cancelar")
                _atualizar_barra(msg)
                cv2.imshow("Selecionar Pontos", _estado["img_display"])
                print(f"  ↩  Ponto {LABELS[n_atual]} removido {removido}. "
                      f"Clique novamente o ponto {proximo_label}.")
            else:
                print("  ✗ Nao ha pontos para desfazer.")

        if tecla in (ord('r'), ord('R')):            # R – recomecar do zero
            if _estado["pontos_img"]:
                _estado["pontos_img"].clear()
                _estado["concluido"] = False
                _estado["img_display"] = _estado["img_display_limpa"].copy()
                _atualizar_barra(msg_inicial)
                cv2.imshow("Selecionar Pontos", _estado["img_display"])
                print("  ↩  Todos os pontos apagados. Recomeçe a clicar o ponto A.")
            else:
                print("  ✗ Ainda nao ha pontos para apagar.")

    cv2.destroyAllWindows()
    return list(_estado["pontos_img"])


# ===========================================================================
# RECOLHA DE COORDENADAS REAIS
# ===========================================================================

def recolher_coordenadas_reais(n_pontos: int) -> list:
    print("\n" + "─" * 60)
    print("  COORDENADAS REAIS DOS PONTOS")
    print("  Origem (0.0, 0.0) = canto superior esquerdo da area")
    print("  X → direita   |   Y → profundidade (para baixo)")
    print("  Escreva \'v\' ou \'voltar\' para anular o ponto anterior.")
    print("─" * 60)

    coords = []
    i = 0
    while i < n_pontos:
        label = LABELS[i]
        print(f"\n  Ponto {label}  ({i+1}/{n_pontos}):")

        # Pede x real
        voltou = False
        while True:
            texto = input("    x real (m)  [ou \'v\' para voltar]: ").strip()
            if texto.lower() in ("v", "voltar"):
                if i > 0:
                    coords.pop()
                    i -= 1
                    print(f"  ↩  Ponto {LABELS[i]} anulado. Reintroduz o ponto {LABELS[i]}.")
                else:
                    print("  ✗ Ja esta no primeiro ponto, nao ha nada para anular.")
                voltou = True
                break
            try:
                x_r = float(texto)
                break
            except ValueError:
                print("  ✗ Valor invalido. Introduza um numero.")

        if voltou:
            continue

        # Pede y real
        while True:
            texto2 = input("    y real (m)  [ou \'v\' para voltar]: ").strip()
            if texto2.lower() in ("v", "voltar"):
                print(f"  ↩  Reintroduz o ponto {label} desde o inicio.")
                voltou = True
                break
            try:
                y_r = float(texto2)
                break
            except ValueError:
                print("  ✗ Valor invalido. Introduza um numero.")

        if voltou:
            continue

        coords.append((x_r, y_r))
        i += 1

    return coords


# ===========================================================================
# HOMOGRAFIA E WARP
# ===========================================================================

def aplicar_homografia(img_undist: np.ndarray,
                        pts_img: list,
                        pts_reais: list) -> tuple:
    """
    Calcula homografia entre pts_img (píxeis) e pts_reais (metros),
    aplica warpPerspective e devolve (imagem_retificada, px_por_metro).
    """
    xs    = [p[0] for p in pts_reais]
    ys    = [p[1] for p in pts_reais]
    x_min = min(xs)
    y_min = min(ys)
    W_real = max(xs) - x_min
    D_real = max(ys) - y_min

    # Resolução: adapta ao tamanho da imagem de entrada
    h_in, w_in = img_undist.shape[:2]
    ppm = max(w_in, h_in) / max(W_real, D_real) if max(W_real, D_real) > 0 else 200.0
    out_w = int(W_real * ppm)
    out_h = int(D_real * ppm)

    # Destino em píxeis
    pts_destino = np.array(
        [((p[0] - x_min) * ppm, (p[1] - y_min) * ppm) for p in pts_reais],
        dtype=np.float32,
    )
    pts_origem = np.array(pts_img, dtype=np.float32)

    if len(pts_img) == 4:
        H_mat, _ = cv2.findHomography(pts_origem, pts_destino)
    else:
        H_mat, _ = cv2.findHomography(
            pts_origem, pts_destino, cv2.RANSAC, ransacReprojThreshold=5.0
        )

    if H_mat is None:
        print("\n  ✗ Não foi possível calcular a homografia.")
        print("  Verifique se os pontos não são colineares.")
        sys.exit(1)

    img_ret = cv2.warpPerspective(img_undist, H_mat, (out_w, out_h))
    return img_ret, ppm, W_real, D_real


# ===========================================================================
# GUARDAR E MOSTRAR
# ===========================================================================

def guardar_imagem(img: np.ndarray, caminho_original: str) -> str:
    pasta = os.path.dirname(os.path.abspath(__file__))
    nome  = os.path.splitext(os.path.basename(caminho_original))[0]
    saida = os.path.join(pasta, f"{nome}_retificada.jpg")
    cv2.imwrite(saida, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return saida


def mostrar_imagem(img: np.ndarray, titulo: str = "Imagem Retificada") -> None:
    MAX_DIM = 1100
    h, w    = img.shape[:2]
    if max(h, w) > MAX_DIM:
        escala  = MAX_DIM / max(h, w)
        img_vis = cv2.resize(img, (int(w * escala), int(h * escala)))
    else:
        img_vis = img

    cv2.imshow(titulo, img_vis)
    print("\n  Prima qualquer tecla na janela da imagem para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===========================================================================
# MAIN
# ===========================================================================

def _pedir_camara() -> dict:
    """Pede o modo de camara e devolve os parametros. Suporta 'v' para voltar ao menu."""
    while True:
        print("\n  Escolha o modo de camara:")
        print("  [1]  iPhone 16 (camara principal) – parametros padrao")
        print("  [2]  Camara configuravel           – introducao manual")
        escolha = input("\n  Opcao (1 ou 2): ").strip()
        if escolha == "1":
            return recolher_parametros_iphone16()
        if escolha == "2":
            return recolher_parametros_configuravel()
        print("  ✗ Opcao invalida. Introduza 1 ou 2.")


def _pedir_imagem_e_params(params: dict) -> tuple:
    """Pede caminho da imagem e devolve (img, img_undist, caminho).
    Devolve None se o utilizador quiser voltar atras (escreve 'v')."""
    while True:
        caminho = input("\nCaminho da imagem  [ou 'v' para voltar a escolha da camara]: ").strip().strip('"').strip("'")
        if caminho.lower() in ("v", "voltar"):
            return None
        if not os.path.isfile(caminho):
            print(f"  ✗ Ficheiro nao encontrado: '{caminho}'. Tente novamente.")
            continue

        img = cv2.imread(caminho)
        if img is None:
            print(f"  ✗ Nao foi possivel ler a imagem: '{caminho}'")
            continue

        h_img, w_img = img.shape[:2]
        if w_img != params["img_w"] or h_img != params["img_h"]:
            print(f"\n  ⚠  Imagem ({w_img}x{h_img}) redimensionada para "
                  f"{params['img_w']}x{params['img_h']}.")
            img = cv2.resize(img, (params["img_w"], params["img_h"]))

        return img, caminho


def main() -> None:

    # ── Aviso inicial ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ⚠  ATENCAO ANTES DE CONTINUAR:")
    print("  Garanta que todo o campo de recolha desejado")
    print("  e completamente visivel na imagem capturada.")
    print("  Areas fora do enquadramento nao serao recuperaveis.")
    print("═" * 60)

    # ── Passo 1: Modo de camara (com possibilidade de repetir) ─────────────
    PASSO = 1
    params       = None
    caminho_img  = None
    img          = None
    img_undist   = None
    n_pontos     = None
    pts_img      = None

    while PASSO <= 5:

        # ── 1. Escolha da camara ───────────────────────────────────────────
        if PASSO == 1:
            params = _pedir_camara()
            PASSO = 2

        # ── 2. Carregar imagem ─────────────────────────────────────────────
        elif PASSO == 2:
            resultado = _pedir_imagem_e_params(params)
            if resultado is None:          # utilizador quer voltar atras
                PASSO = 1
                continue
            img, caminho_img = resultado
            print("\n  A corrigir distorcao da lente...")
            img_undist = aplicar_undistort(img, params)
            PASSO = 3

        # ── 3. Numero de pontos ────────────────────────────────────────────
        elif PASSO == 3:
            print("\n" + "─" * 60)
            print("  CALIBRACAO POR PONTOS DE REFERENCIA")
            print("─" * 60)
            print("  Vai clicar N pontos na imagem cujas posicoes reais conhece.")
            print("  Recomendacao: use os cantos da area de interesse (4 pontos),")
            print("  ou pontos extra para maior robustez (6, 8, 10).\n")
            print("  Escreva 'v' para voltar a escolha da imagem.")
            texto_n = input("  Numero de pontos (4/6/8/10): ").strip()
            if texto_n.lower() in ("v", "voltar"):
                PASSO = 2
                continue
            try:
                n_pontos = int(texto_n)
                if n_pontos not in (4, 6, 8, 10):
                    raise ValueError
            except ValueError:
                print("  ✗ Introduza 4, 6, 8 ou 10.")
                continue
            PASSO = 4

        # ── 4. Selecao interativa de pontos ────────────────────────────────
        elif PASSO == 4:
            print(f"\n  Vai abrir a imagem. Clique {n_pontos} pontos.")
            print("  Na janela: Z=desfazer ultimo  R=recomecar  ESC=cancelar")
            print("  De seguida, introduz as coordenadas reais de cada ponto.\n")
            resp = input("  Prima ENTER para abrir a imagem  [ou 'v' para voltar]: ").strip()
            if resp.lower() in ("v", "voltar"):
                PASSO = 3
                continue
            pts_img = selecionar_pontos(img_undist, n_pontos)
            PASSO = 5

        # ── 5. Coordenadas reais ────────────────────────────────────────────
        elif PASSO == 5:
            pts_reais = recolher_coordenadas_reais(n_pontos)
            # Nao ha volta atras daqui – mas o proprio recolher suporta 'v' ponto a ponto
            break

    # ── Homografia + Warp ──────────────────────────────────────────────────
    print("\n  A calcular homografia e retificar imagem...")
    img_ret, ppm, W_real, D_real = aplicar_homografia(img_undist, pts_img, pts_reais)

    # ── Guardar ────────────────────────────────────────────────────────────
    caminho_saida = guardar_imagem(img_ret, caminho_img)
    print(f"\n  ✓ Area retificada:   {W_real:.2f} m x {D_real:.2f} m")
    print(f"  ✓ Resolucao saida:   {ppm:.0f} px/m")
    print(f"  ✓ Dimensoes saida:   {img_ret.shape[1]} x {img_ret.shape[0]} px")
    print(f"  ✓ Ficheiro guardado: {caminho_saida}")

    # ── Mostrar ────────────────────────────────────────────────────────────
    mostrar_imagem(img_ret)


if __name__ == "__main__":
    main()
