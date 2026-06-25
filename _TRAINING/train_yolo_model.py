from pathlib import Path

import torch
from ultralytics import YOLO


BASE_PATH = Path(__file__).resolve().parents[1]


def train():
    data_path = BASE_PATH / "DataSet" / "data.yaml"

    print("--- VERIFICACAO ---")
    if not data_path.exists():
        print(f"ERRO: o ficheiro nao esta em: {data_path}")
        return
    print("Ficheiro YAML encontrado.")

    if not torch.cuda.is_available():
        print("GPU nao encontrada. A parar para nao queimar o CPU.")
        return

    print(f"GPU ativa: {torch.cuda.get_device_name(0)}")
    model = YOLO(str(BASE_PATH / "models" / "yolov8s.pt"))
    model.train(
        data=str(data_path),
        epochs=100,
        imgsz=640,
        device=0,
        workers=2,
        name="treino_bolas_v2",
    )


if __name__ == "__main__":
    train()


