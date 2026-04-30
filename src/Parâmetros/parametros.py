"""
parametros.py — Gestão dos Parâmetros Configuráveis UFSC/FEUP
==============================================================
Os parâmetros do sistema vivem em resultados/configuracao/parametros.json.
Cada parâmetro tem nome interno, descrição em PT, unidade, valor atual,
intervalo válido e tipo (int ou float). Se o ficheiro não existir, é
criado automaticamente com os defaults.

Categorias e respetivos parâmetros (14 no total):

  FAIXAS
    n_faixas                  número de faixas horizontais (1–26)
    raio_dedup_cm             raio de deduplicação espacial em cm

  DISPARO
    threshold_pct             percentagem mínima para disparar (5–100)
    k_min                     mínimo absoluto de bolas
    timeout_varrimento_s      segundos para libertar faixa por segurança

  TRAJETÓRIA
    tolerancia_distancia_cm   distância para considerar "chegou" (cm)
    tolerancia_angulo_graus   tolerância de alinhamento (graus)

  CONTROLADOR
    v_max                     velocidade linear máxima (m/s)
    omega_max                 velocidade angular máxima (rad/s)
    k_ang                     ganho proporcional do erro angular
    thr_ang_grosso_graus      acima disto só roda no sítio (graus)
    d_lim                     distância de saturação linear (m)

  REDE
    ip_robo                   IP do ESP32
    porta_udp                 porta UDP do robô

Uso típico em qualquer módulo:

    from parametros import carregar
    cfg = carregar()
    n = cfg["n_faixas"]
    raio_m = cfg["raio_dedup_cm"] / 100.0
"""

import json
from pathlib import Path
from typing import Any


BASE_PATH      = Path(__file__).parent.resolve()
PASTA_CONFIG   = BASE_PATH / "resultados" / "configuracao"
FICH_PARAMS    = PASTA_CONFIG / "parametros.json"


# ─────────────────────────────────────────────
#  ESQUEMA DOS PARÂMETROS
# ─────────────────────────────────────────────
# Cada entrada: (chave_interna, categoria, descricao, unidade, default,
#                tipo, min_val, max_val)
# Lista numerada: a ordem aqui é a que aparece ao utilizador no menu.
ESQUEMA: list[dict] = [
    # ── FAIXAS ─────────────────────────────────────────────────────
    {
        "chave":     "n_faixas",
        "categoria": "FAIXAS",
        "descricao": "Número de faixas horizontais em que a quadra é dividida",
        "unidade":   "",
        "default":   10,
        "tipo":      "int",
        "min":       1,
        "max":       26,
    },
    {
        "chave":     "raio_dedup_cm",
        "categoria": "FAIXAS",
        "descricao": "Raio espacial para considerar deteções da mesma bola",
        "unidade":   "cm",
        "default":   8.0,
        "tipo":      "float",
        "min":       1.0,
        "max":       30.0,
    },
    # ── DISPARO ────────────────────────────────────────────────────
    {
        "chave":     "threshold_pct",
        "categoria": "DISPARO",
        "descricao": "Percentagem mínima do total para uma faixa disparar",
        "unidade":   "%",
        "default":   25.0,
        "tipo":      "float",
        "min":       5.0,
        "max":       100.0,
    },
    {
        "chave":     "k_min",
        "categoria": "DISPARO",
        "descricao": "Número mínimo absoluto de bolas para uma faixa disparar",
        "unidade":   "bolas",
        "default":   4,
        "tipo":      "int",
        "min":       1,
        "max":       50,
    },
    {
        "chave":     "n_obs_min_estavel",
        "categoria": "DISPARO",
        "descricao": "Mínimo de observações consecutivas para uma bola ser considerada parada e contar",
        "unidade":   "frames",
        "default":   3,
        "tipo":      "int",
        "min":       1,
        "max":       30,
    },
    {
        "chave":     "timeout_varrimento_s",
        "categoria": "DISPARO",
        "descricao": "Tempo máximo de varrimento antes de libertar por segurança",
        "unidade":   "s",
        "default":   90.0,
        "tipo":      "float",
        "min":       10.0,
        "max":       600.0,
    },
    # ── TRAJETÓRIA ─────────────────────────────────────────────────
    {
        "chave":     "tolerancia_distancia_cm",
        "categoria": "TRAJETORIA",
        "descricao": "Quão perto o robô tem de estar do ponto-alvo para considerar 'chegou'",
        "unidade":   "cm",
        "default":   20.0,
        "tipo":      "float",
        "min":       2.0,
        "max":       100.0,
    },
    {
        "chave":     "tolerancia_angulo_graus",
        "categoria": "TRAJETORIA",
        "descricao": "Tolerância de alinhamento angular no ponto inicial",
        "unidade":   "°",
        "default":   15.0,
        "tipo":      "float",
        "min":       1.0,
        "max":       90.0,
    },
    # ── CONTROLADOR ────────────────────────────────────────────────
    {
        "chave":     "v_max",
        "categoria": "CONTROLADOR",
        "descricao": "Velocidade linear máxima do robô",
        "unidade":   "m/s",
        "default":   0.15,
        "tipo":      "float",
        "min":       0.05,
        "max":       2.0,
    },
    {
        "chave":     "omega_max",
        "categoria": "CONTROLADOR",
        "descricao": "Velocidade angular máxima do robô",
        "unidade":   "rad/s",
        "default":   1.0,
        "tipo":      "float",
        "min":       0.1,
        "max":       5.0,
    },
    {
        "chave":     "k_ang",
        "categoria": "CONTROLADOR",
        "descricao": "Ganho proporcional do erro angular",
        "unidade":   "rad/s por rad",
        "default":   1.5,
        "tipo":      "float",
        "min":       0.1,
        "max":       10.0,
    },
    {
        "chave":     "thr_ang_grosso_graus",
        "categoria": "CONTROLADOR",
        "descricao": "Acima deste erro angular o robô só roda (não anda)",
        "unidade":   "°",
        "default":   20.0,
        "tipo":      "float",
        "min":       1.0,
        "max":       90.0,
    },
    {
        "chave":     "d_lim",
        "categoria": "CONTROLADOR",
        "descricao": "Distância acima da qual a velocidade linear satura em V_max",
        "unidade":   "m",
        "default":   0.5,
        "tipo":      "float",
        "min":       0.05,
        "max":       5.0,
    },
    # ── REDE ───────────────────────────────────────────────────────
    {
        "chave":     "ip_robo",
        "categoria": "REDE",
        "descricao": "IP do ESP32 do robô (placeholder = modo simulado)",
        "unidade":   "",
        "default":   "IP_DO_ROBO",
        "tipo":      "str",
        "min":       None,
        "max":       None,
    },
    {
        "chave":     "porta_udp",
        "categoria": "REDE",
        "descricao": "Porta UDP do ESP32 para receber comandos motores",
        "unidade":   "",
        "default":   5005,
        "tipo":      "int",
        "min":       1024,
        "max":       65535,
    },
]


# ─────────────────────────────────────────────
#  PERSISTÊNCIA
# ─────────────────────────────────────────────
def _defaults() -> dict[str, Any]:
    return {entry["chave"]: entry["default"] for entry in ESQUEMA}


def carregar() -> dict[str, Any]:
    """
    Carrega o ficheiro parametros.json. Se não existir, cria-o com os
    defaults. Se existir mas tiver chaves em falta (versão antiga, por
    exemplo), preenche as em falta com defaults e regrava.
    Devolve o dicionário completo de parâmetros.
    """
    PASTA_CONFIG.mkdir(parents=True, exist_ok=True)

    if not FICH_PARAMS.exists():
        cfg = _defaults()
        guardar(cfg)
        return cfg

    try:
        with open(FICH_PARAMS, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        # ficheiro corrompido — recria com defaults
        cfg = _defaults()
        guardar(cfg)
        return cfg

    # Preenche chaves em falta com defaults (compatibilidade evolutiva)
    defaults = _defaults()
    alterado = False
    for chave, valor_default in defaults.items():
        if chave not in cfg:
            cfg[chave] = valor_default
            alterado = True
    if alterado:
        guardar(cfg)

    return cfg


def guardar(cfg: dict[str, Any]):
    """Guarda o dicionário de parâmetros no JSON."""
    PASTA_CONFIG.mkdir(parents=True, exist_ok=True)
    with open(FICH_PARAMS, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


# ─────────────────────────────────────────────
#  VALIDAÇÃO
# ─────────────────────────────────────────────
def validar_valor(esquema_entry: dict, valor_str: str) -> tuple[bool, Any, str]:
    """
    Valida e converte um valor introduzido pelo utilizador.
    Devolve (ok, valor_convertido, erro_msg).
    """
    tipo = esquema_entry["tipo"]
    chave = esquema_entry["chave"]

    if tipo == "str":
        # Para strings (ex: ip_robo), aceitamos qualquer coisa não-vazia.
        # A validação fina (formato IP) fica para quem usa.
        valor = valor_str.strip()
        if not valor:
            return False, None, "valor não pode estar vazio"
        return True, valor, ""

    try:
        if tipo == "int":
            valor = int(valor_str)
        else:  # float
            valor = float(valor_str.replace(",", "."))
    except ValueError:
        return False, None, f"valor inválido para tipo {tipo}"

    mn, mx = esquema_entry["min"], esquema_entry["max"]
    if mn is not None and valor < mn:
        return False, None, f"valor abaixo do mínimo ({mn})"
    if mx is not None and valor > mx:
        return False, None, f"valor acima do máximo ({mx})"

    return True, valor, ""


# ─────────────────────────────────────────────
#  FORMATAÇÃO PARA APRESENTAÇÃO
# ─────────────────────────────────────────────
def formatar_valor(esquema_entry: dict, cfg: dict[str, Any]) -> str:
    chave = esquema_entry["chave"]
    valor = cfg.get(chave, esquema_entry["default"])
    unidade = esquema_entry["unidade"]
    if esquema_entry["tipo"] == "float":
        txt = f"{valor:.3f}"
        # remove zeros redundantes mas mantém pelo menos 1 decimal
        if "." in txt:
            txt = txt.rstrip("0").rstrip(".") if txt.rstrip("0").rstrip(".") else txt
            if "." not in txt:
                txt += ".0"
    else:
        txt = str(valor)
    return f"{txt} {unidade}".strip()


def formatar_intervalo(esquema_entry: dict) -> str:
    if esquema_entry["tipo"] == "str":
        return "qualquer texto"
    mn, mx = esquema_entry["min"], esquema_entry["max"]
    if mn is None and mx is None:
        return "—"
    return f"{mn} a {mx}"


if __name__ == "__main__":
    # Demo: carrega e imprime o estado atual dos parâmetros
    cfg = carregar()
    print(f"Parâmetros lidos de {FICH_PARAMS}:")
    print()
    cat_atual = None
    for i, entry in enumerate(ESQUEMA, 1):
        if entry["categoria"] != cat_atual:
            cat_atual = entry["categoria"]
            print(f"\n── {cat_atual} ──")
        print(f"  [{i:2d}] {entry['chave']:25s} = {formatar_valor(entry, cfg):20s}  "
              f"({entry['descricao']})")