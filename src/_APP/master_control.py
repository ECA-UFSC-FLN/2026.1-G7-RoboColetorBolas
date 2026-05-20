"""
MasterControl.py — Orquestrador Central UFSC/FEUP
==================================================
Pontos de entrada após arrancar:
  [1] Iniciar produção
  [2] Recalibrar (homografia)
  [3] Configurar parâmetros
  [4] Sair

Argumentos de linha de comando:
  --debug   Abre uma segunda consola onde aparecem todas as mensagens
            de DEBUG dos módulos do pipeline. Sem este argumento, o
            utilizador vê só mensagens humanas, eventos, avisos e erros.

Câmara suportada:
  Camo Studio (Reincubate) via USB. O iPhone deve estar ligado via USB
  com a app Camo ativa antes de iniciar. O perfil da câmara
  (IPHONE16_1X / IPHONE16_05X / EXTERNO) é configurado nos parâmetros
  (opção 3) e determina os intrínsecos e a correção de paralaxe.

Health-checks (cada módulo abre o seu numa porta dedicada):
  6011  retificador
  6002  VisionProcessing
  6013  GraphProcessor
  6014  RobotController

Portas autenticadas (não tocar sem authkey):
  6000  VisionProcessing ↔ imageStreaming
  6001  retificador      ↔ imageStreaming/calibração
  6020  retificador       → GraphProcessor (fila de JSONs)
  6021  GraphProcessor    → RobotController (broadcast de estado)
  6030  consola de debug  ← qualquer módulo (TCP texto)
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import _CONFIG.system_parameters as parametros
from _COMMON.logging_utils import log, pedir_input

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
PYTHON_EXE = r"C:\Users\andre\venv_bolas\Scripts\python.exe"
BASE_PATH = PROJECT_ROOT
SCRIPTS = {
    "debug_console": BASE_PATH / "_TOOLS" / "debug_console.py",
    "camera_stream": BASE_PATH / "_VISION" / "camera_stream.py",
    "vision_processor": BASE_PATH / "_VISION" / "vision_processor.py",
    "court_rectifier": BASE_PATH / "_VISION" / "court_rectifier.py",
    "court_graph_processor": BASE_PATH / "_PLANNING" / "court_graph_processor.py",
    "robot_controller": BASE_PATH / "_CONTROL" / "robot_controller.py",
}

CALIB_FILE          = BASE_PATH / "resultados" / "calibracao" / "homografia_calibracao.json"
PASTA_CALIB_REF     = BASE_PATH / "resultados" / "calibracao"

# Portas de health
PORTA_RET_HEALTH      = 6011
PORTA_VIS_HEALTH      = 6002
PORTA_GRAFO_HEALTH    = 6013
PORTA_CONTROL_HEALTH  = 6014

# Portas autenticadas
PORTA_VIS         = 6000
PORTA_RET         = 6001
PORTA_RET_GRAFO   = 6020
PORTA_BROADCAST   = 6021

# Consola de debug
PORTA_DEBUG       = 6030

TIMEOUT_ARRANQUE = 60
INTERVALO_POLL   = 0.5

MOD = "MASTER"


# ─────────────────────────────────────────────
#  ARGUMENTOS DE LINHA DE COMANDO
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Orquestrador UFSC/FEUP")
parser.add_argument("--debug", action="store_true",
                    help="Ativa consola de debug separada (porta 6030)")
ARGS = parser.parse_args()

if ARGS.debug:
    os.environ["BOLAS_DEBUG"] = "1"


# ─────────────────────────────────────────────
#  AUXILIARES VISUAIS
# ─────────────────────────────────────────────
def separador(titulo: str = ""):
    if titulo:
        print(f"\033[1;96m{'─'*22} {titulo} {'─'*22}\033[0m", flush=True)
    else:
        print(f"\033[90m{'─'*70}\033[0m", flush=True)


def cabecalho_inicial():
    os.system("cls")
    print("\033[1;96m")
    print("╔══════════════════════════════════════════════════════╗")
    print("║       SISTEMA INTEGRADO UFSC/FEUP — Bolas v3         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("\033[0m")


# ─────────────────────────────────────────────
#  CONSOLA DE DEBUG
# ─────────────────────────────────────────────
def lancar_consola_debug():
    log(MOD, "HUMANO", "A abrir consola de debug em janela separada...")
    log(MOD, "DEBUG",  f"PYTHON_EXE={PYTHON_EXE} | porta={PORTA_DEBUG}")
    cmd = (
        f'start "SISTEMA UFSC/FEUP — Debug" cmd /k '
        f'chcp 65001 > nul ^&^& '
        f'"{PYTHON_EXE}" "{SCRIPTS["debug_console"]}"'
    )
    try:
        subprocess.Popen(cmd, shell=True)
        time.sleep(1.5)
        log(MOD, "HUMANO", "Consola de debug pronta.")
    except Exception as e:
        log(MOD, "ERRO", f"Não consegui abrir consola de debug: {e}")


# ─────────────────────────────────────────────
#  GESTÃO DE PORTAS E PROCESSOS
# ─────────────────────────────────────────────
def porta_aberta(porta: int) -> bool:
    try:
        with socket.create_connection(("localhost", porta), timeout=0.3):
            return True
    except OSError:
        return False


def aguardar_porta(porta: int, servico: str, timeout: int = TIMEOUT_ARRANQUE) -> bool:
    log(MOD, "HUMANO", f"A aguardar arranque de '{servico}'...")
    log(MOD, "DEBUG",  f"polling porta {porta} timeout={timeout}s")
    tentativas = int(timeout / INTERVALO_POLL)
    for i in range(tentativas):
        if porta_aberta(porta):
            log(MOD, "HUMANO", f"'{servico}' pronto.")
            log(MOD, "DEBUG",  f"porta {porta} respondeu após "
                               f"{i*INTERVALO_POLL:.1f}s")
            return True
        if i > 0 and i % int(5 / INTERVALO_POLL) == 0:
            log(MOD, "DEBUG", f"  ... ainda a aguardar '{servico}' "
                              f"({int(i*INTERVALO_POLL)}s/{timeout}s)")
        time.sleep(INTERVALO_POLL)
    log(MOD, "ERRO", f"TIMEOUT: '{servico}' não respondeu em {timeout}s.")
    log(MOD, "DEBUG", f"porta {porta} continua fechada após {timeout}s")
    return False


def executar_modulo(script_key: str, args: list | None = None) -> subprocess.Popen:
    script_path = SCRIPTS[script_key]
    cmd = [PYTHON_EXE, str(script_path)] + (args or [])
    log(MOD, "DEBUG", f"a lançar: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=os.environ.copy())


def _pids_a_ocupar_portas(portas: list[int]) -> set[int]:
    pids: set[int] = set()
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "TCP"],
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
        )
    except Exception as e:
        log(MOD, "DEBUG", f"netstat falhou: {e}")
        return pids

    for linha in out.splitlines():
        partes = linha.split()
        if len(partes) < 5:
            continue
        if "LISTENING" not in partes[3]:
            continue
        local = partes[1]
        if ":" not in local:
            continue
        try:
            porta = int(local.rsplit(":", 1)[1])
        except ValueError:
            continue
        if porta in portas:
            try:
                pids.add(int(partes[4]))
            except ValueError:
                pass
    return pids


def matar_processos_pendentes():
    portas_alvo = [
        PORTA_RET_HEALTH, PORTA_VIS_HEALTH,
        PORTA_GRAFO_HEALTH, PORTA_CONTROL_HEALTH,
        PORTA_VIS, PORTA_RET, PORTA_RET_GRAFO, PORTA_BROADCAST,
        PORTA_DEBUG,
    ]
    log(MOD, "AVISO", "A verificar processos pendentes nas portas do pipeline...")
    log(MOD, "DEBUG",  f"portas alvo: {portas_alvo}")

    pid_master = os.getpid()
    pids = _pids_a_ocupar_portas(portas_alvo)
    pids.discard(pid_master)

    if not pids:
        log(MOD, "HUMANO", "Não havia processos pendentes — tudo limpo.")
        return

    log(MOD, "DEBUG", f"PIDs a matar: {sorted(pids)} (Master={pid_master} blindado)")
    for pid in pids:
        ret = os.system(f"taskkill /f /pid {pid} /t >nul 2>&1")
        log(MOD, "DEBUG", f"taskkill /pid {pid} → exit={ret}")
    time.sleep(1.5)
    log(MOD, "HUMANO", f"Processos pendentes encerrados ({len(pids)} no total).")


def encerrar_pipeline(processos: list, motivo: str = "sinal do utilizador"):
    separador("ENCERRAMENTO")
    log(MOD, "AVISO", f"A encerrar pipeline ({motivo})...")
    for p in processos:
        if p and p.poll() is None:
            log(MOD, "DEBUG", f"a terminar PID {p.pid}")
            try:
                p.terminate(); p.wait(timeout=3)
            except Exception:
                log(MOD, "DEBUG", f"terminate falhou, a forçar kill no PID {p.pid}")
                p.kill()
    log(MOD, "HUMANO", "Pipeline encerrado. Até à próxima!")
    sys.exit(0)


# ─────────────────────────────────────────────
#  FASE 1 — CALIBRAÇÃO DINÂMICA
# ─────────────────────────────────────────────
def fase_calibracao() -> bool:
    separador("FASE 1 — CALIBRAÇÃO DINÂMICA")

    # Mostrar perfil de câmara activo antes de calibrar
    cfg = parametros.carregar()
    perfil = cfg.get("perfil_camara", "IPHONE16_1X")
    log(MOD, "HUMANO", f"Perfil de câmara para esta calibração: {perfil}")
    log(MOD, "HUMANO",
        "Para alterar o perfil cancela e vai a [3] Configurar parâmetros → CAMARA.")

    if porta_aberta(PORTA_RET_HEALTH):
        log(MOD, "AVISO", "Porta de health do retificador já ocupada. A limpar...")
        matar_processos_pendentes()

    log(MOD, "HUMANO", "A iniciar servidor de calibração...")
    p_ret = executar_modulo("court_rectifier", ["--calibrar"])

    if not aguardar_porta(PORTA_RET_HEALTH, "retificador (calib)", timeout=30):
        log(MOD, "ERRO", "Retificador de calibração não arrancou. A abortar.")
        p_ret.terminate()
        return False

    log(MOD, "HUMANO", "A lançar câmara Camo para captura do frame de referência...")
    p_cap = executar_modulo("camera_stream", ["--modo", "calibracao"])

    log(MOD, "HUMANO", "Captura um frame com a tecla 'C' e marca os pontos na janela.")
    log(MOD, "HUMANO", "O sistema avança automaticamente quando a calibração for guardada.")

    p_ret.wait()
    p_cap.terminate()

    if CALIB_FILE.exists():
        log(MOD, "HUMANO", "Calibração guardada com sucesso!")
        log(MOD, "DEBUG",  f"ficheiro: {CALIB_FILE}")
        return True
    else:
        log(MOD, "ERRO", "Ficheiro de calibração não foi criado. Verifica os pontos.")
        return False


# ─────────────────────────────────────────────
#  FASE 2 — PRODUÇÃO
# ─────────────────────────────────────────────
def fase_producao():
    separador("FASE 2 — PIPELINE DE PRODUÇÃO")

    if (porta_aberta(PORTA_RET_HEALTH) or porta_aberta(PORTA_VIS_HEALTH)
        or porta_aberta(PORTA_GRAFO_HEALTH) or porta_aberta(PORTA_CONTROL_HEALTH)):
        log(MOD, "AVISO", "Portas já ocupadas. A matar processos pendentes...")
        matar_processos_pendentes()

    processos: list[subprocess.Popen] = []

    log(MOD, "HUMANO", "A lançar retificador...")
    log(MOD, "DEBUG",  "health=6011 | sockets autenticados=6001 (calib) e 6020 (grafo)")
    p_ret = executar_modulo("court_rectifier")
    processos.append(p_ret)
    if not aguardar_porta(PORTA_RET_HEALTH, "retificador"):
        encerrar_pipeline(processos, "retificador não respondeu")

    log(MOD, "HUMANO", "A lançar processamento de visão...")
    log(MOD, "DEBUG",  "health=6002 | socket autenticado=6000")
    p_vis = executar_modulo("vision_processor")
    processos.append(p_vis)
    if not aguardar_porta(PORTA_VIS_HEALTH, "VisionProcessing"):
        encerrar_pipeline(processos, "VisionProcessing não respondeu")

    log(MOD, "HUMANO", "A lançar GraphProcessor...")
    log(MOD, "DEBUG",  "health=6013 | cliente de 6020 | broadcaster em 6021")
    p_grafo = executar_modulo("court_graph_processor")
    processos.append(p_grafo)
    if not aguardar_porta(PORTA_GRAFO_HEALTH, "GraphProcessor"):
        encerrar_pipeline(processos, "GraphProcessor não respondeu")

    log(MOD, "HUMANO", "A lançar controlador do robô...")
    log(MOD, "DEBUG",  "health=6014 | cliente de 6021")
    p_ctrl = executar_modulo("robot_controller")
    processos.append(p_ctrl)
    if not aguardar_porta(PORTA_CONTROL_HEALTH, "RobotController"):
        encerrar_pipeline(processos, "RobotController não respondeu")

    separador()
    log(MOD, "HUMANO", "Pipeline ativo!")
    log(MOD, "HUMANO", "Tecla P → pausar  |  E → encerrar  |  I → info no terminal")
    separador()

    try:
        subprocess.run([PYTHON_EXE, str(SCRIPTS["camera_stream"]),
                        "--modo", "producao"], env=os.environ.copy())
    except KeyboardInterrupt:
        pass
    finally:
        encerrar_pipeline(processos, "imageStreaming encerrado")


# ─────────────────────────────────────────────
#  CONFIGURAÇÃO DE PARÂMETROS
# ─────────────────────────────────────────────
def menu_parametros():
    separador("CONFIGURAR PARÂMETROS")
    cfg = parametros.carregar()
    log(MOD, "HUMANO", f"Ficheiro: {parametros.FICH_PARAMS}")
    log(MOD, "HUMANO", f"Total de {len(parametros.ESQUEMA)} parâmetros configuráveis.")
    log(MOD, "HUMANO",
        "NOTA: parâmetros [EXTERNO] só têm efeito quando perfil_camara = EXTERNO.")

    while True:
        print()
        cat_atual = None
        for i, entry in enumerate(parametros.ESQUEMA, 1):
            if entry["categoria"] != cat_atual:
                cat_atual = entry["categoria"]
                print(f"\n  \033[1;96m── {cat_atual} ──\033[0m")
            valor_fmt = parametros.formatar_valor(entry, cfg)
            print(f"  \033[90m[{i:2d}]\033[0m  "
                  f"{entry['chave']:28s}  "
                  f"\033[1;92m{valor_fmt:18s}\033[0m  "
                  f"\033[90m{entry['descricao']}\033[0m")
        print()

        resp = pedir_input(MOD,
            "Número do parâmetro a editar (ou 'g' guardar e sair, 'c' cancelar):")

        if resp.lower() == "c":
            log(MOD, "AVISO", "Configuração cancelada — alterações não guardadas.")
            return
        if resp.lower() == "g":
            parametros.guardar(cfg)
            log(MOD, "HUMANO", "Parâmetros guardados.")
            log(MOD, "DEBUG",  f"escrito em {parametros.FICH_PARAMS}")
            return

        try:
            idx = int(resp) - 1
            if not (0 <= idx < len(parametros.ESQUEMA)):
                raise ValueError
        except ValueError:
            log(MOD, "AVISO", f"Entrada inválida: '{resp}'. Tenta de novo.")
            continue

        entry = parametros.ESQUEMA[idx]
        valor_atual = parametros.formatar_valor(entry, cfg)
        intervalo   = parametros.formatar_intervalo(entry)

        print()
        print(f"  \033[1;96m{entry['chave']}\033[0m  ({entry['categoria']})")
        print(f"  {entry['descricao']}")
        print(f"  Valor atual:       \033[1;92m{valor_atual}\033[0m")
        print(f"  Intervalo válido:  {intervalo}")
        novo = pedir_input(MOD, "Novo valor (ENTER mantém atual):")

        if not novo:
            log(MOD, "DEBUG", f"sem alteração em '{entry['chave']}'")
            continue

        ok, valor_conv, erro = parametros.validar_valor(entry, novo)
        if not ok:
            log(MOD, "AVISO", f"Valor rejeitado: {erro}")
            continue

        cfg[entry["chave"]] = valor_conv
        log(MOD, "HUMANO",
            f"'{entry['chave']}' alterado para "
            f"{parametros.formatar_valor(entry, cfg)}")


# ─────────────────────────────────────────────
#  MENU PRINCIPAL
# ─────────────────────────────────────────────
def menu_principal():
    while True:
        fichs_calib = (list(PASTA_CALIB_REF.glob("homografia*.json"))
                       if PASTA_CALIB_REF.exists() else [])
        tem_calib = bool(fichs_calib)

        if tem_calib:
            log(MOD, "HUMANO", f"Calibração existente: {CALIB_FILE.name}")
            log(MOD, "DEBUG",  f"{len(fichs_calib)} ficheiro(s) na pasta de calibração")
        else:
            log(MOD, "AVISO",
                "Nenhuma calibração encontrada. Será obrigatória antes da produção.")

        cfg = parametros.carregar()
        perfil = cfg.get("perfil_camara", "IPHONE16_1X")
        log(MOD, "DEBUG",
            f"parametros.json com {len(cfg)} chave(s) | perfil={perfil}")

        separador()
        print()
        print("  \033[1;96m[1]\033[0m  Iniciar produção")
        print("  \033[1;96m[2]\033[0m  Recalibrar (homografia)")
        print("  \033[1;96m[3]\033[0m  Configurar parâmetros")
        print("  \033[1;96m[4]\033[0m  Sair")
        print()
        resp = pedir_input(MOD, "Escolhe uma opção [1-4]:")

        if resp == "1":
            if not tem_calib:
                log(MOD, "AVISO",
                    "Não há calibração. A iniciar calibração obrigatória primeiro.")
                if not fase_calibracao():
                    log(MOD, "ERRO", "Calibração falhou. A voltar ao menu.")
                    continue
            fase_producao()
            return

        elif resp == "2":
            if tem_calib:
                conf = pedir_input(MOD,
                    "Tens a certeza? Calibração antiga será apagada (s/N):")
                if conf.lower() != "s":
                    log(MOD, "DEBUG", "recalibração cancelada pelo utilizador")
                    continue
                try:
                    CALIB_FILE.unlink()
                    log(MOD, "HUMANO", "Calibração anterior removida.")
                    log(MOD, "DEBUG",  f"unlink {CALIB_FILE}")
                except PermissionError:
                    log(MOD, "AVISO",
                        "Não consegui apagar — a matar processos pendentes...")
                    matar_processos_pendentes()
                    CALIB_FILE.unlink()
            if fase_calibracao():
                input("\033[1;92m  Calibração concluída! "
                      "Prima ENTER para voltar ao menu...\033[0m")

        elif resp == "3":
            menu_parametros()

        elif resp == "4":
            log(MOD, "HUMANO", "Até à próxima!")
            sys.exit(0)

        else:
            log(MOD, "AVISO", f"Opção inválida: '{resp}'")


# ─────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────
def main():
    cabecalho_inicial()
    log(MOD, "HUMANO", f"Pasta do projeto: {BASE_PATH.name}")
    log(MOD, "DEBUG",  f"BASE_PATH={BASE_PATH}")
    log(MOD, "DEBUG",  f"PYTHON_EXE={PYTHON_EXE}")
    log(MOD, "DEBUG",  f"BOLAS_DEBUG={'1' if ARGS.debug else '0'}")

    if ARGS.debug:
        lancar_consola_debug()

    menu_principal()


if __name__ == "__main__":
    main()




