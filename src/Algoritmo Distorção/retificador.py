import cv2
import numpy as np
import json
import sys
import argparse
import time
from pathlib import Path
from multiprocessing.connection import Listener

# Caminhos de sistema
BASE_PATH = Path(r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC")
PASTA_SAIDA = BASE_PATH / "Prontinho para os Grafões do Pedrão fazer a trajetória das ilegalidades"
FICHEIRO_CALIB = BASE_PATH / "homografia_calibracao.json"

# Parâmetros intrínsecos iPhone 16
K_CAM = np.array([[5823, 0, 2016], [0, 5823, 1512], [0, 0, 1]], dtype=np.float64)
D_CAM = np.array([0.122, -0.246, 0.0001, -0.0002, 0.176], dtype=np.float64)


def aplicar_undistort(img):
    return cv2.undistort(img, K_CAM, D_CAM)


def calibrar_via_socket():
    print("\n[CALIBRACAO] Servidor ativo na porta 6001. Envie o frame no Stream...")
    address = ('localhost', 6001)
    with Listener(address, authkey=b'retificador_ufsc') as listener:
        with listener.accept() as conn:
            pacote = conn.recv()
            img = pacote['frame']

    img_undist = aplicar_undistort(img)

    try:
        n = int(input("\n[CONFIG] Quantos pontos de referencia vai marcar? (min 4): "))
    except:
        n = 4

    pts_px = []

    def clique(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_px) < n:
            pts_px.append((x, y))
            cv2.circle(img_draw, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(img_draw, str(len(pts_px)), (x + 15, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("CALIBRACAO - Selecione os pontos", img_draw)

    img_draw = img_undist.copy()
    cv2.namedWindow("CALIBRACAO - Selecione os pontos")
    cv2.imshow("CALIBRACAO - Selecione os pontos", img_draw)
    cv2.setMouseCallback("CALIBRACAO - Selecione os pontos", clique)

    print(f"[INSTRUCAO] Marque os {n} pontos na imagem. Ao terminar, pressione ENTER na imagem.")

    while len(pts_px) < n:
        cv2.waitKey(1)

    cv2.waitKey(0)

    # --- RECOLHA DE COORDENADAS ---
    pts_reais = []
    print("\n[COORDENADAS REAIS] Consulte a imagem numerada para inserir os metros:")
    for i in range(n):
        print(f"\n--- Ponto {i+1} ---")
        cv2.waitKey(1)
        xr = float(input(f"  X real (metros): "))
        yr = float(input(f"  Y real (metros): "))
        pts_reais.append([xr, yr])

    cv2.destroyAllWindows()

    # ---------------------------------------------------------------
    # HOMOGRAFIA CORRIGIDA: px → px (com ppm para converter depois)
    # ---------------------------------------------------------------
    xs    = [p[0] for p in pts_reais]
    ys    = [p[1] for p in pts_reais]
    x_min = min(xs)
    y_min = min(ys)
    W_real = max(xs) - x_min
    D_real = max(ys) - y_min

    h_in, w_in = img_undist.shape[:2]
    ppm = max(w_in, h_in) / max(W_real, D_real) if max(W_real, D_real) > 0 else 200.0

    # Pontos de destino em píxeis (escala da imagem de saída)
    pts_destino_px = np.array(
        [((p[0] - x_min) * ppm, (p[1] - y_min) * ppm) for p in pts_reais],
        dtype=np.float32
    )
    pts_origem_px = np.array(pts_px, dtype=np.float32)

    if n == 4:
        H, _ = cv2.findHomography(pts_origem_px, pts_destino_px)
    else:
        H, _ = cv2.findHomography(pts_origem_px, pts_destino_px,
                                   cv2.RANSAC, ransacReprojThreshold=5.0)

    with open(FICHEIRO_CALIB, "w") as f:
        json.dump({
            "H_mat": H.tolist(),
            "ppm":   ppm,
            "x_min": x_min,
            "y_min": y_min,
        }, f, indent=4)

    print("\n[SUCESSO] Calibracao guardada. O sistema vai avancar.")
    sys.exit(0)


def servidor_producao(calib):
    H     = np.array(calib["H_mat"])
    ppm   = calib["ppm"]
    x_min = calib.get("x_min", 0.0)
    y_min = calib.get("y_min", 0.0)

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    print(f"\n[SISTEMA] Servidor de Retificacao ATIVO (6001).")
    print(f"[INFO] Resultados: {PASTA_SAIDA.name}")

    address = ('localhost', 6001)
    with Listener(address, authkey=b'retificador_ufsc') as listener:
        while True:
            with listener.accept() as conn:
                try:
                    pacote = conn.recv()
                    res_bolas = []

                    for b in pacote["bolas_px"]:
                        cx = (b["x1"] + b["x2"]) / 2.0
                        cy = (b["y1"] + b["y2"]) / 2.0

                        # 1. Corrigir distorção da lente no ponto (consistente com calibração)
                        pt_raw = np.array([[[cx, cy]]], dtype=np.float32)
                        pt_undist = cv2.undistortPoints(pt_raw, K_CAM, D_CAM, P=K_CAM)
                        ux = float(pt_undist[0][0][0])
                        uy = float(pt_undist[0][0][1])

                        # 2. Aplicar homografia px → px
                        pt_warp = cv2.perspectiveTransform(
                            np.array([[[ux, uy]]], dtype=np.float32), H
                        )
                        wx = float(pt_warp[0][0][0])
                        wy = float(pt_warp[0][0][1])

                        # 3. Converter píxeis → metros
                        x_metros = wx / ppm + x_min
                        y_metros = wy / ppm + y_min

                        res_bolas.append({
                            "x": round(x_metros, 4),
                            "y": round(y_metros, 4),
                        })

                    saida = {
                        "indice":      pacote["indice"],
                        "latencia_ms": round((time.time() - pacote["timestamp_visao"]) * 1000, 2),
                        "trajetoria":  res_bolas,
                    }

                    with open(PASTA_SAIDA / f"trajetoria_{pacote['indice']}.json", "w") as f:
                        json.dump(saida, f, indent=4)

                    conn.send("LIBERADO")

                except Exception as e:
                    print(f"[ERRO] {e}")
                    conn.send("LIBERADO")


if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrar", action="store_true")
    args = parser.parse_args()
    if args.calibrar:
        calibrar_via_socket()
    else:
        with open(FICHEIRO_CALIB) as f:
            servidor_producao(json.load(f))
