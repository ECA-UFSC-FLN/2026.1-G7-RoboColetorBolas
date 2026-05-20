"""
bolas_log.py — Logging Centralizado UFSC/FEUP
==============================================
Sistema de logging com seis níveis e cores ANSI distintas, com a
particularidade de poder enviar mensagens de DEBUG para uma segunda
consola separada (escutando na porta 6030).

NÍVEIS:
  HUMANO  (verde brilhante)  — "o quê" e "quando" do fluxo
  EVENTO  (magenta)          — marcos importantes (DISPARO, etc.)
  AVISO   (amarelo)          — algo está mal mas continua
  ERRO    (vermelho)         — falha crítica
  INPUT   (azul brilhante)   — solicitação de entrada do utilizador
  DEBUG   (cinzento)         — detalhes técnicos para diagnóstico

REGRA:
  Mensagens HUMANO/EVENTO/AVISO/ERRO/INPUT vão para STDOUT da consola atual.
  Mensagens DEBUG vão para STDOUT *só* se DEBUG_LOCAL=True (modo legado).
  Mensagens DEBUG vão para PORTA_DEBUG via TCP se a flag --debug estiver ativa
  (controlada pela variável de ambiente BOLAS_DEBUG=1).

Uso típico em qualquer módulo:

    from bolas_log import log

    log("RETIFICADOR", "HUMANO", "Calibração concluída!")
    log("RETIFICADOR", "DEBUG",  "ppm=200.4, x_min=-1.234, y_min=0.0")
    log("RETIFICADOR", "ERRO",   "Não consegui abrir a câmara.")
"""

import os
import sys
import socket
import threading
from datetime import datetime
from queue import Queue, Empty


# ─────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────
PORTA_DEBUG          = 6030
DEBUG_ATIVO          = os.environ.get("BOLAS_DEBUG", "0") == "1"
DEBUG_LOCAL          = False    # se True, debug também sai na consola normal

# Cores ANSI
COR_HUMANO  = "\033[1;92m"   # verde brilhante
COR_EVENTO  = "\033[1;95m"   # magenta brilhante
COR_AVISO   = "\033[93m"     # amarelo
COR_ERRO    = "\033[1;91m"   # vermelho brilhante
COR_INPUT   = "\033[1;94m"   # azul brilhante
COR_DEBUG   = "\033[90m"     # cinzento
COR_RESET   = "\033[0m"

# Ícones
ICONS = {
    "HUMANO": "✓",
    "EVENTO": "★",
    "AVISO":  "!",
    "ERRO":   "✗",
    "INPUT":  "?",
    "DEBUG":  "·",
}

CORES = {
    "HUMANO": COR_HUMANO,
    "EVENTO": COR_EVENTO,
    "AVISO":  COR_AVISO,
    "ERRO":   COR_ERRO,
    "INPUT":  COR_INPUT,
    "DEBUG":  COR_DEBUG,
}


# ─────────────────────────────────────────────
#  CLIENTE TCP PARA A CONSOLA DE DEBUG
# ─────────────────────────────────────────────
# Mantemos uma fila local e um thread que tenta enviar tudo. Se a consola
# de debug não estiver aberta ou cair, simplesmente perdemos as mensagens
# de debug — nunca bloqueamos o emissor, e nunca propagamos exceções para
# o código que chamou log().

_fila_debug: Queue = Queue(maxsize=2000)
_thread_iniciada = False
_thread_lock = threading.Lock()


def _ligar_a_consola() -> socket.socket | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(("localhost", PORTA_DEBUG))
        s.settimeout(None)
        return s
    except (ConnectionRefusedError, OSError):
        return None


def _loop_envio_debug():
    """Thread em background que tenta sempre estar ligado à consola."""
    sock: socket.socket | None = None
    backoff = 0.5
    while True:
        if sock is None:
            sock = _ligar_a_consola()
            if sock is None:
                # Consola de debug não está aberta. Esperamos antes de tentar de novo.
                # Entretanto vamos drenando a fila para não estagnar memória.
                try:
                    _fila_debug.get(timeout=backoff)
                except Empty:
                    pass
                backoff = min(backoff * 1.5, 5.0)
                continue
            backoff = 0.5

        try:
            msg = _fila_debug.get(timeout=1.0)
        except Empty:
            continue

        try:
            sock.sendall((msg + "\n").encode("utf-8", errors="replace"))
        except (BrokenPipeError, ConnectionResetError, OSError):
            try: sock.close()
            except Exception: pass
            sock = None


def _garantir_thread():
    global _thread_iniciada
    with _thread_lock:
        if not _thread_iniciada and DEBUG_ATIVO:
            t = threading.Thread(target=_loop_envio_debug, daemon=True)
            t.start()
            _thread_iniciada = True


# ─────────────────────────────────────────────
#  FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────
def log(modulo: str, nivel: str, msg: str):
    """
    Imprime uma mensagem com timestamp, módulo, nível e cor adequada.
    Mensagens de DEBUG são adicionalmente reenviadas para a consola
    separada se BOLAS_DEBUG=1.

    Parâmetros:
      modulo  — nome curto do módulo emissor (ex: "RETIFICADOR")
      nivel   — um de HUMANO / EVENTO / AVISO / ERRO / INPUT / DEBUG
      msg     — texto a imprimir
    """
    nivel = nivel.upper()
    if nivel not in CORES:
        nivel = "DEBUG"

    ts   = datetime.now().strftime("%H:%M:%S")
    icon = ICONS.get(nivel, "·")
    cor  = CORES.get(nivel, COR_DEBUG)
    linha_visivel = f"{cor}[{ts}] [{modulo:13s}] {icon} {msg}{COR_RESET}"

    # ── Caso DEBUG ──────────────────────────────────────────
    if nivel == "DEBUG":
        if DEBUG_LOCAL:
            print(linha_visivel, flush=True)
        if DEBUG_ATIVO:
            _garantir_thread()
            linha_remota = f"{cor}[{ts}] [{modulo:13s}] {icon} {msg}{COR_RESET}"
            try:
                _fila_debug.put_nowait(linha_remota)
            except Exception:
                # Fila cheia — descartamos. Debug não é crítico.
                pass
        return

    # ── Restantes níveis vão sempre para o stdout local ─────
    print(linha_visivel, flush=True)


# ─────────────────────────────────────────────
#  HELPER PARA INPUT COLORIDO
# ─────────────────────────────────────────────
def pedir_input(modulo: str, prompt: str) -> str:
    """
    Mostra um prompt em azul e devolve o texto introduzido pelo utilizador.
    Equivalente a um log(INPUT) seguido de input(), mas numa só linha.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    cor_prompt = f"{COR_INPUT}[{ts}] [{modulo:13s}] ?{COR_RESET} {COR_INPUT}{prompt}{COR_RESET} "
    return input(cor_prompt).strip()


# ─────────────────────────────────────────────
#  BANNER OPCIONAL — para usar no início de cada módulo
# ─────────────────────────────────────────────
def imprimir_banner(modulo: str, descricao: str = ""):
    """Imprime uma linha-cabeçalho ao arrancar um módulo."""
    if descricao:
        log(modulo, "HUMANO", f"━━━ {descricao} ━━━")
    if DEBUG_ATIVO:
        log(modulo, "DEBUG", f"BOLAS_DEBUG=1 — mensagens de debug a "
                             f"enviar para porta {PORTA_DEBUG}")


if __name__ == "__main__":
    # Demo: corre `python bolas_log.py` para veres as cores
    log("DEMO", "HUMANO", "Esta é uma mensagem humana (verde brilhante)")
    log("DEMO", "EVENTO", "Este é um evento importante (magenta)")
    log("DEMO", "AVISO",  "Este é um aviso (amarelo)")
    log("DEMO", "ERRO",   "Este é um erro (vermelho)")
    log("DEMO", "DEBUG",  "Esta é uma mensagem de debug (cinzento)")
    log("DEMO", "INPUT",  "Este é um pedido de input (azul)")
    print()
    print("Para ativar o reencaminhamento de DEBUG para a segunda consola:")
    print("  set BOLAS_DEBUG=1 && python bolas_log.py")





