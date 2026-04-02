from ultralytics import YOLO
import os

# Caminho exato para o ficheiro de configuração que o Roboflow gerou
# (Assumindo que o ficheiro se chama data.yaml dentro dessa pasta)
path_to_data = r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC\Dataset_Inicial.v2i.yolov8\data.yaml"

# Carregar o modelo (YOLO26n é a versão mais atual e leve de 2026)
model = YOLO("yolo26n.pt") 

if __name__ == '__main__':
    # Iniciar o treino
    model.train(
        data=path_to_data,
        epochs=50,
        imgsz=640,
        device='cpu', # Se tiveres uma GPU NVIDIA, muda para 0
        workers=0      # Importante no Windows dentro do OneDrive para evitar erros de multiprocessamento
    )