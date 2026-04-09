from ultralytics import YOLO
import torch
import os

def train():
    # Caminho exato que extraímos do teu erro
    data_path = r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC\DataSet\data.yaml"

    print(f"--- VERIFICAÇÃO ---")
    if os.path.exists(data_path):
        print(f"✅ Ficheiro YAML encontrado!")
    else:
        print(f"❌ ERRO: O ficheiro não está em: {data_path}")
        return

    # Diagnóstico de GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU Ativa: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU não encontrada. A parar para não queimar o CPU.")
        return

    # Carregar Modelo (Small é melhor que Nano para o teu caso)
    model = YOLO('yolov8s.pt') 

    # Iniciar Treino
    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        device=0,
        workers=2,
        name='treino_bolas_v2'
    )

if __name__ == '__main__':
    train()