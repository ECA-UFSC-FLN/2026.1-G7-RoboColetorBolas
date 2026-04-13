import cv2
import time
from pathlib import Path
from multiprocessing.connection import Client

# --- CONFIGURACAO ---
# Pasta inicial apenas para backup/registo se desejares (opcional)
PASTA_BACKUP = Path(r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC\Captured_Pictures")

def enviar_para_visao(frame, nome):
    try:
        address = ('localhost', 6000)
        # Tenta ligar ao Vision (isto atua como o Mutex)
        with Client(address, authkey=b'bolas_ufsc') as conn:
            print(f"\n[ETAPA] Enviando dados diretamente para o Cérebro...")
            
            pacote = {
                'frame': frame,
                'timestamp': time.time(),
                'nome': nome
            }
            
            conn.send(pacote)
            
            # Bloqueia e espera o sinal de retorno (Semaforo de confirmacao)
            print("[ETAPA] Aguardando processamento do Cérebro...")
            resposta = conn.recv()
            
            if resposta == "LIBERADO":
                print("[SUCESSO] Cerebro processou e libertou o sistema.")
                return True
    except ConnectionRefusedError:
        print("[BLOQUEIO] Erro: O Script de Visao nao esta a correr!")
        return False
    except Exception as e:
        print(f"[ERRO] Falha na comunicacao: {e}")
        return False

# --- MÉTODOS DE CAPTURA (Exemplo iPhone) ---
def modo_iphone():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    print("\n--- SISTEMA DE CAPTURA DIRETA ATIVO ---")
    print("Comandos: C - Capturar e Enviar | E - Sair")

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Stream", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla in [ord('c'), ord('C')]:
            # Enviar diretamente via Socket/Mutex
            sucesso = enviar_para_visao(frame.copy(), "iphone_capture")
            if sucesso:
                # Opcional: Guardar na pasta inicial apenas como registo
                ts = time.time()
                cv2.imwrite(str(PASTA_BACKUP / f"backup_{ts}.jpg"), frame)

        elif tecla in [ord('e'), ord('E')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    PASTA_BACKUP.mkdir(parents=True, exist_ok=True)
    modo_iphone() # Ou modo_android()