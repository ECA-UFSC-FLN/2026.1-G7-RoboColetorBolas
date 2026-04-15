import time
from ultralytics import YOLO
from multiprocessing.connection import Listener, Client
from pathlib import Path

MODELO = Path(r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC\runs\detect\treino_bolas_v24\weights\best.pt")

def iniciar_visao():
    print("[VISAO] Carregando Modelo...")
    model = YOLO(MODELO)
    
    # Servidor na porta 6000
    listener = Listener(('localhost', 6000), authkey=b'bolas_ufsc')
    print("[VISAO] Aguardando dados na porta 6000...")
    
    indice = 0
    while True:
        with listener.accept() as conn:
            pacote = conn.recv()
            print(f"[PROCESSO] Analisando pacote {indice}...")
            
            results = model.predict(source=pacote['frame'], conf=0.5, device=0, verbose=False)
            bolas = []
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    bolas.append({"x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3])})
            
            # Enviar para o Retificador (6001)
            try:
                with Client(('localhost', 6001), authkey=b'retificador_ufsc') as c_ret:
                    print("[VISAO] Passando para o Retificador...")
                    c_ret.send({
                        'frame': pacote['frame'], 
                        'bolas_px': bolas, 
                        'indice': indice, 
                        'timestamp_visao': pacote['timestamp']
                    })
                    # Espera o Retificador acabar
                    c_ret.recv()
                
                # Só agora responde ao capturador
                conn.send("LIBERADO")
                print(f"[SUCESSO] Pacote {indice} finalizado.")
            except Exception as e:
                print(f"[ERRO] O Retificador (6001) falhou: {e}")
                conn.send("ERRO")
            
            indice += 1

if __name__ == "__main__":
    iniciar_visao()