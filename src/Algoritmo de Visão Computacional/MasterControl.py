import subprocess
import time
import sys
import os
from pathlib import Path

PYTHON_EXE = r"C:\Users\andre\venv_bolas\Scripts\python.exe"
BASE_PATH = Path(r"C:\Users\andre\OneDrive\Ambiente de Trabalho\FEUP\4º Ano - Mestrado\Projeto Integrador - UFSC")

def matar_processos_antigos():
    """Limpa execuções pendentes para libertar sockets e ficheiros."""
    print("[SISTEMA] Verificando processos pendentes...")
    os.system('taskkill /f /im python.exe /t >nul 2>&1')
    time.sleep(1)

def executar_modulo(script_name, args=None):
    comando = [PYTHON_EXE, str(BASE_PATH / script_name)] + (args if args else [])
    return subprocess.Popen(comando)

def gerir_sistema():
    print("\n=== CONSOLA MASTER UFSC/FEUP ===")
    calib_file = BASE_PATH / "homografia_calibracao.json"

    quer_calibrar = False
    if not calib_file.exists():
        quer_calibrar = True
    else:
        print(f"[ESTADO] Calibracao existente detetada.")
        resp = input(">> Deseja recalibrar o sistema? (s/N): ").lower()
        if resp == 's':
            quer_calibrar = True
            try:
                os.remove(calib_file)
            except PermissionError:
                matar_processos_antigos()
                os.remove(calib_file)

    if quer_calibrar:
        print("\n[FASE] Iniciando Calibracao Dinamica...")
        p_ret = executar_modulo("retificador.py", ["--calibrar"])
        p_cap = executar_modulo("imageStreaming.py")
        p_ret.wait()
        p_cap.terminate()
        
        if not calib_file.exists():
            print("[ERRO] Calibracao nao concluida.")
            sys.exit(1)

    print("\n[FASE] Ativando Pipeline de Producao...")
    p_ret = executar_modulo("retificador.py")
    time.sleep(4)
    p_vis = executar_modulo("VisionProcessing.py")
    time.sleep(6)
    
    try:
        subprocess.run([PYTHON_EXE, str(BASE_PATH / "imageStreaming.py")])
    finally:
        p_ret.terminate()
        p_vis.terminate()
        matar_processos_antigos()

if __name__ == "__main__":
    gerir_sistema()