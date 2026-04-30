"""
debug_console.py — Consola Secundária de Debug UFSC/FEUP
=========================================================
Pequeno servidor TCP que aceita ligações de qualquer módulo do pipeline
e imprime as suas mensagens de DEBUG nesta janela. As mensagens já vêm
formatadas com cores ANSI; este script só as escreve no stdout.

Como funciona:
  - Escuta em localhost:6030
  - Aceita múltiplas ligações simultâneas (uma por módulo)
  - Cada linha recebida é impressa imediatamente
  - Ctrl+C ou fechar a janela termina o servidor

Esta consola é lançada automaticamente pelo MasterControl.py quando este
é executado com --debug. Para abrir manualmente:
    python debug_console.py
"""

import socket
import threading
import sys
import os
from datetime import datetime

PORTA_DEBUG = 6030

COR_HEADER  = "\033[1;96m"   # ciano brilhante
COR_DIM     = "\033[90m"
COR_RESET   = "\033[0m"


def header():
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{COR_HEADER}╔══════════════════════════════════════════════════════════════════╗")
    print(f"║         CONSOLA DE DEBUG — SISTEMA UFSC/FEUP — Bolas v3          ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝{COR_RESET}")
    print(f"{COR_DIM}  A escutar na porta {PORTA_DEBUG}. Mensagens de DEBUG dos vários módulos")
    print(f"  do pipeline aparecerão aqui em tempo real.")
    print(f"  Ctrl+C ou fecha a janela para terminar.{COR_RESET}")
    print()


def lidar_com_cliente(conn: socket.socket, addr):
    """Lê linhas da ligação e imprime cada uma."""
    try:
        ficheiro = conn.makefile("r", encoding="utf-8", errors="replace")
        for linha in ficheiro:
            linha = linha.rstrip("\r\n")
            if linha:
                print(linha, flush=True)
    except Exception:
        pass
    finally:
        try: conn.close()
        except Exception: pass


def main():
    header()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("localhost", PORTA_DEBUG))
    except OSError as e:
        print(f"{COR_DIM}[FATAL]{COR_RESET} Não consegui abrir a porta {PORTA_DEBUG}: {e}")
        print(f"{COR_DIM}        Outra instância da consola de debug já está aberta?{COR_RESET}")
        input("Prima ENTER para sair...")
        sys.exit(1)

    srv.listen(10)

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=lidar_com_cliente, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print()
        print(f"{COR_DIM}A encerrar consola de debug...{COR_RESET}")
    finally:
        try: srv.close()
        except Exception: pass


if __name__ == "__main__":
    main()
