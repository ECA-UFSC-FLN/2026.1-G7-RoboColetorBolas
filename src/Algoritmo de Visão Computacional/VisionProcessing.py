"""
MasterControl.py — Orquestrador Central do Sistema UFSC/FEUP
============================================================
NOTA: Os health-checks usam portas dedicadas:
      - Retificador  → porta 6011
      - VisionProcessing → porta 6002
      As portas 6001 e 6000 são exclusivas dos Listeners autenticados.
      Ligar a elas sem authkey causa ConnectionAbortedError 10053.
"""

import subprocess
import time
import sys
import os
import socket
from pathlib import Path
from datetime import datetime

PYTHON_EXE = r"C:\Users\andre\venv_bolas\Scripts\python.exe"
BASE_PATH  = Path(__file__).parent.resolve()

CALIB_FILE           = BASE_PATH / "resultados" / "calibracao" / "homografia_calibracao.json"
PASTA_CALIB_REF      = BASE_PATH / "resultados" / "calibracao"
PORTA_RET_HEALTH     = 6011   # health-check do retificador
PORTA_VIS_HEALTH     = 6002   # health-check do VisionProcessing
PORTA_VIS            = 6000   # comunicação autenticada VisionProcessing
PORTA_RET            = 6001   # comunicação autenticada retificador

TIMEOUT_ARRANQUE = 60
INTERVALO_POLL   = 0.5

ICONS = {"INFO": "·", "OK": "✓", "ERRO": "✗", "AVISO": "!", "FASE": "▶️"}

def log(modulo, nivel, msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    icon = ICONS.get(nivel, "·")
    cor  = {"OK": "\033[92m","ERRO": "\033[91m","AVISO": "\033[93m","FASE": "\033[96m","INFO": "\033[0m"}.get(nivel, "\033[0m")
    print(f"{cor}[{ts}] [{modulo:12s}] {icon} {msg}\033[0m", flush=True)

def separador(titulo=""):
    if titulo:
        print(f"\033[96m{'─'*20} {titulo} {'─'*20}\033[0m", flush=True)
    else:
        print(f"\033[90m{'─'*60}\033[0m", flush=True)

def porta_aberta(porta):
    try:
        with socket.create_connection(("localhost", porta), timeout=0.3):
            return True
    except OSError:
        return False

def aguardar_porta(porta, servico, timeout=TIMEOUT_ARRANQUE):
    log("MASTER", "INFO", f"Aguardando '{servico}' na porta {porta}...")
    tentativas = int(timeout / INTERVALO_POLL)
    for i in range(tentativas):
        if porta_aberta(porta):
            log("MASTER", "OK", f"'{servico}' pronto! (porta {porta})")
            return True
        if i > 0 and i % int(5 / INTERVALO_POLL) == 0:
            log("MASTER", "INFO", f"  ... ainda a aguardar '{servico}' ({int(i*INTERVALO_POLL)}s/{timeout}s)")
        time.sleep(INTERVALO_POLL)
    log("MASTER", "ERRO", f"TIMEOUT: '{servico}' não respondeu em {timeout}s na porta {porta}.")
    return False

def executar_modulo(script_name, args=None):
    cmd = [PYTHON_EXE, str(BASE_PATH / script_name)] + (args or [])
    log("MASTER", "INFO", f"Lançando: {script_name} {' '.join(args or [])}")
    return subprocess.Popen(cmd)

def matar_processos_pendentes():
    log("MASTER", "AVISO", "Encerrando processos Python pendentes...")
    os.system("taskkill /f /im python.exe /t >nul 2>&1")
    time.sleep(1.5)
    log("MASTER", "OK", "Processos encerrados.")

def encerrar_pipeline(processos, motivo="sinal do utilizador"):
    separador("ENCERRAMENTO")
    log("MASTER", "AVISO", f"A encerrar pipeline ({motivo})...")
    for p in processos:
        if p and p.poll() is None:
            try:
                p.terminate(); p.wait(timeout=3)
            except Exception:
                p.kill()
    log("MASTER", "OK", "Pipeline encerrado. Até à próxima!")
    sys.exit(0)

def fase_calibracao():
    separador("FASE 1 — CALIBRAÇÃO DINÂMICA")

    if porta_aberta(PORTA_RET_HEALTH):
        log("CALIB", "AVISO", "Porta de health do retificador já ocupada! A limpar...")
        matar_processos_pendentes()

    log("CALIB", "FASE", "Iniciando servidor de calibração (porta 6001)...")
    p_ret = executar_modulo("retificador.py", ["--calibrar"])

    if not aguardar_porta(PORTA_RET_HEALTH, "retificador (calib) health", timeout=30):
        log("CALIB", "ERRO", "Retificador de calibração não arrancou. Abortando.")
        p_ret.terminate()
        return False

    log("CALIB", "FASE", "Lançando câmera para capturar frame de referência...")
    p_cap = executar_modulo("imageStreaming.py", ["--modo", "calibracao"])

    log("CALIB", "INFO", "Aguarda: captura um frame (tecla C) e depois marca os pontos na janela.")
    log("CALIB", "INFO", "O sistema avança automaticamente após a calibração ser guardada.")

    p_ret.wait()
    p_cap.terminate()

    if CALIB_FILE.exists():
        log("CALIB", "OK", f"Calibração guardada em: {CALIB_FILE.name}")
        return True
    else:
        log("CALIB", "ERRO", "Ficheiro de calibração não foi criado. Verifica os pontos marcados.")
        return False

def fase_producao():
    separador("FASE 2 — PIPELINE DE PRODUÇÃO")

    if porta_aberta(PORTA_RET_HEALTH) or porta_aberta(PORTA_VIS_HEALTH):
        log("PROD", "AVISO", "Portas já ocupadas! A matar processos pendentes...")
        matar_processos_pendentes()

    processos = []

    log("PROD", "FASE", "Lançando retificador.py (health → 6011, socket → 6001)...")
    p_ret = executar_modulo("retificador.py")
    processos.append(p_ret)
    if not aguardar_porta(PORTA_RET_HEALTH, "retificador health"):
        log("PROD", "ERRO", "Retificador não arrancou. Abortando pipeline.")
        encerrar_pipeline(processos, "retificador não respondeu")

    log("PROD", "FASE", "Lançando VisionProcessing.py (health → 6002, socket → 6000)...")
    p_vis = executar_modulo("VisionProcessing.py")
    processos.append(p_vis)
    # Aguarda pela porta de health dedicada — NÃO toca na porta autenticada 6000
    if not aguardar_porta(PORTA_VIS_HEALTH, "VisionProcessing health"):
        log("PROD", "ERRO", "Módulo de visão não arrancou. Abortando pipeline.")
        encerrar_pipeline(processos, "VisionProcessing não respondeu")

    separador()
    log("PROD", "OK", "Pipeline ativo! Iniciando captura contínua de imagem...")
    log("PROD", "INFO", "Tecla P → pausar | Tecla E → encerrar")
    separador()

    try:
        subprocess.run([PYTHON_EXE, str(BASE_PATH / "imageStreaming.py"), "--modo", "producao"])
    except KeyboardInterrupt:
        pass
    finally:
        encerrar_pipeline(processos, "imageStreaming encerrado")

def gerir_sistema():
    os.system("cls")
    print("\033[96m")
    print("╔══════════════════════════════════════════════════╗")
    print("║      SISTEMA INTEGRADO UFSC/FEUP — Bolas v3      ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\033[0m")
    log("MASTER", "INFO", f"Pasta do projeto: {BASE_PATH.name}")

    quer_calibrar = False
    fichs_calib = list(PASTA_CALIB_REF.glob("homografia*.json")) if PASTA_CALIB_REF.exists() else []

    if not fichs_calib:
        log("MASTER", "AVISO", "Nenhuma calibração encontrada em resultados/calibracao/. Calibração obrigatória.")
        quer_calibrar = True
    else:
        # Usa sempre o homografia_calibracao.json fixo; os outros são histórico
        log("MASTER", "OK", f"Calibração existente: {CALIB_FILE.name} "
                            f"({len(fichs_calib)} ficheiro(s) na pasta)")
        separador()
        resp = input("\033[93m>> Recalibrar o sistema? (s/N): \033[0m").strip().lower()
        if resp == "s":
            quer_calibrar = True
            try:
                CALIB_FILE.unlink()
                log("MASTER", "OK", "Calibração anterior removida.")
            except PermissionError:
                matar_processos_pendentes()
                CALIB_FILE.unlink()

    if quer_calibrar:
        ok = fase_calibracao()
        if not ok:
            log("MASTER", "ERRO", "Calibração falhou. Encerra e tenta novamente.")
            sys.exit(1)
        separador()
        input("\033[92m>> Calibração concluída! Prima ENTER para iniciar a produção...\033[0m")

    fase_producao()

if __name__ == "__main__":
    gerir_sistema()