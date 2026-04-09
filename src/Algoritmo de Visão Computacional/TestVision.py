from ultralytics import YOLO
import cv2

# 1. Carrega o teu modelo treinado
model = YOLO(r'C:\Users\andre\runs\detect\train\weights\best.pt')

# 2. Faz a previsão numa imagem nova (mete o caminho de uma foto de teste aqui)
results = model.predict(
    source=r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC\Fotografias_Recolhidas\WhatsApp Image 2026-04-02 at 10.36.37.jpeg",
    conf=0.5,
    save=True
)
# 3. Mostrar os resultados
for r in results:
    print(f"Detetadas {len(r.boxes)} bolas de ténis!")