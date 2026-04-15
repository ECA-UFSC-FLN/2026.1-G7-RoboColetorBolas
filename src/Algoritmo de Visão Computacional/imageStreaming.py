import cv2
import time
from multiprocessing.connection import Client

def enviar_pacote(frame, dispositivo):
    # 1. Tentar Módulo de Visão (Porta 6000)
    try:
        address = ('localhost', 6000)
        with Client(address, authkey=b'bolas_ufsc') as conn:
            print("[INFO] Enviando para Visão (Porta 6000)...")
            conn.send({'frame': frame, 'timestamp': time.time(), 'nome': dispositivo})
            
            # Aguarda o "LIBERADO" que vem via Vision -> Retificador
            resposta = conn.recv()
            if resposta == "LIBERADO":
                print("[SUCESSO] Pipeline completa concluída.")
                return True
    except ConnectionRefusedError:
        # 2. Se a Visão falhar, tentar Retificador direto (Modo Calibração - Porta 6001)
        try:
            address_calib = ('localhost', 6001)
            with Client(address_calib, authkey=b'retificador_ufsc') as conn:
                print("[SISTEMA] Modo Calibração detetado. Enviando para Porta 6001...")
                conn.send({'frame': frame})
                return True
        except ConnectionRefusedError:
            print("[ERRO] Ninguém está a ouvir! (Portas 6000 e 6001 fechadas)")
            return False
    except Exception as e:
        print(f"[ERRO] Falha na comunicação: {e}")
        return False

def stream():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    print("\n[STREAM] Comandos: C (Capturar) | E (Sair)")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Monitor de Captura", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('c'):
            enviar_pacote(frame.copy(), "cam_principal")
        elif tecla == ord('e'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream()