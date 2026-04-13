import os
import time
import json
from pathlib import Path
from ultralytics import YOLO
from multiprocessing.connection import Listener

# --- CONFIGURACAO ---
MODELO_PATH = Path(r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC\runs\detect\treino_bolas_v24\weights\best.pt")
PATH_VALUES = Path(r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC\Coordinates\Values")
PATH_IMAGES = Path(r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC\Coordinates\Images (para visualizaçao)")

def processador_visao():
    PATH_VALUES.mkdir(parents=True, exist_ok=True)
    PATH_IMAGES.mkdir(parents=True, exist_ok=True)

    print("[ETAPA] Carregando modelo YOLOv8 na GPU...")
    model = YOLO(MODELO_PATH)

    # Criar o "Porto" de escuta (Socket IPC)
    address = ('localhost', 6000) 
    listener = Listener(address, authkey=b'bolas_ufsc')
    
    print("[ETAPA] Mutex Binario Ativo: Servidor aguardando conexao direta...")
    
    indice = 0
    while True:
        conn = listener.accept() # Bloqueia aqui ate o Streaming ligar
        print(f"\n[ETAPA] Conexao estabelecida com o Script de Captura.")
        
        try:
            # Receber o pacote de dados (Imagem + Timestamp)
            pacote = conn.recv()
            tempo_captura = pacote['timestamp']
            frame = pacote['frame']
            nome_origem = pacote['nome']

            print(f"[ETAPA] Imagem '{nome_origem}' recebida via Mutex/Socket.")

            # Inferência YOLO
            t_inicio_proc = time.time()
            results = model.predict(source=frame, conf=0.5, device=0, verbose=False)
            
            dados_bolas = []
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    dados_bolas.append({"x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3])})
                
                # Guardar visualizacao
                r.save(filename=str(PATH_IMAGES / f"visualizacao{indice}.jpg"))

            # Calculo de Latencia de Ponta-a-Ponta
            tempo_final = time.time()
            latencia_ms = (tempo_final - tempo_captura) * 1000

            # Salvar JSON final
            nome_json = f"coordinates{indice}.json"
            saida = {
                "indice": indice,
                "latencia_total_ms": round(latencia_ms, 2),
                "bolas": dados_bolas
            }
            with open(PATH_VALUES / nome_json, 'w') as f:
                json.dump(saida, f, indent=4)

            print(f"[ETAPA] Processamento concluido. Latencia: {latencia_ms:.2f}ms")
            
            # Enviar sinal de "OK" de volta para libertar o Streaming
            conn.send("LIBERADO")
            indice += 1

        except Exception as e:
            print(f"[ERRO] Falha no processamento: {e}")
        finally:
            conn.close()
            print("[ESTADO] Mutex resetado. Aguardando proxima captura...")

if __name__ == "__main__":
    processador_visao()