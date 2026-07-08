"""
parametros.py — Gestão dos Parâmetros Configuráveis UFSC/FEUP
==============================================================
Os parâmetros do sistema vivem em resultados/configuracao/parametros.json.
Cada parâmetro tem nome interno, descrição em PT, unidade, valor atual,
intervalo válido e tipo (int ou float). Se o ficheiro não existir, é
criado automaticamente com os defaults.

Categorias e respetivos parâmetros (25 no total):

  CAMARA
    perfil_camara             "IPHONE16_1X" | "IPHONE16_05X" | "SAMSUNG_S25_PLUS_1X" | "SAMSUNG_S25_PLUS_UW" | "XIAOMI_MI_9T_1X" | "EXTERNO"
    altura_camara_m           altura da câmara ao solo (m)
    altura_bola_m             raio da bola de ténis = 0.0325 m (para correção de paralaxe)
    altura_aruco_m            altura dos marcadores ArUco ao solo (m) (topo do robô)

    ── apenas ativos quando perfil_camara = "EXTERNO" ──
    ext_fx                    distância focal horizontal (px)
    ext_fy                    distância focal vertical (px)
    ext_cx                    ponto principal X (px)
    ext_cy                    ponto principal Y (px)
    ext_k1 … ext_k3           coeficientes de distorção radial
    ext_p1 … ext_p2           coeficientes de distorção tangencial
    ext_res_w / ext_res_h     resolução de referência para os parâmetros (px)

  MODO
    modo_operacao             "FAIXAS" ou "GLOBAL"
    k_min_global              mínimo de bolas totais para disparar modo GLOBAL
    n_obs_estavel_global      mínimo de obs. consecutivas para bola contar (global)

  FAIXAS
    largura_robo_cm           largura usada para espaçar as faixas horizontais
    raio_dedup_cm             raio de deduplicação espacial em cm

  DISPARO
    threshold_pct             percentagem mínima para disparar (5–100)
    k_min                     mínimo absoluto de bolas
    timeout_varrimento_s      segundos para libertar faixa por segurança

  TRAJETÓRIA
    tolerancia_distancia_cm   distância para considerar "chegou" (cm)
    tolerancia_angulo_graus   tolerância de alinhamento (graus)

  CONTROLADOR
    v_max / omega_max / k_ang / thr_ang_grosso_graus / d_lim

  REDE
    ip_robo / porta_udp

Uso típico:

    from parametros import carregar, obter_intrinsics
    cfg = carregar()
    K, D = obter_intrinsics(cfg)
    altura_cam = cfg["altura_camara_m"]
    altura_bola = cfg["altura_bola_m"]
"""

import json
from pathlib import Path
from typing import Any
import numpy as np


BASE_PATH = Path(__file__).resolve().parents[1]
PASTA_CONFIG   = BASE_PATH / "resultados" / "configuracao"
FICH_PARAMS    = PASTA_CONFIG / "parametros.json"


# ─────────────────────────────────────────────
#  PARÂMETROS INTRÍNSECOS POR PERFIL
# ─────────────────────────────────────────────
# iPhone 16 — lente principal 1× (f ≈ 26 mm eq., sensor 4032×3024)
_IPHONE16_1X_K = [
    [5823.0,    0.0, 2016.0],
    [   0.0, 5823.0, 1512.0],
    [   0.0,    0.0,    1.0],
]
_IPHONE16_1X_D = [0.122, -0.246, 0.0001, -0.0002, 0.176]
_IPHONE16_1X_RES = [4032, 3024]

# iPhone 16 — lente ultra-grande angular 0.5× (f ≈ 13 mm eq., sensor 4032×3024)
# Distorção radial muito mais pronunciada na UGA.
_IPHONE16_05X_K = [
    [2912.0,    0.0, 2016.0],
    [   0.0, 2912.0, 1512.0],
    [   0.0,    0.0,    1.0],
]
_IPHONE16_05X_D = [0.281, -0.612, 0.0003, -0.0005, 0.438]
_IPHONE16_05X_RES = [4032, 3024]

PERFIS_INTRINSICOS = {
    "IPHONE16_1X":  (_IPHONE16_1X_K,  _IPHONE16_1X_D,  _IPHONE16_1X_RES),
    "IPHONE16_05X": (_IPHONE16_05X_K, _IPHONE16_05X_D, _IPHONE16_05X_RES),
}

# Samsung Galaxy S25 Plus — perfis aproximados.
# A câmara principal é 50 MP, 24 mm eq.; a ultrawide é 12 MP, 13 mm eq.
# Estes valores são bons para selecionar rapidamente o telemóvel no sistema,
# mas a opção "CALIBRADO" continua a ser preferível para máxima precisão.
_SAMSUNG_S25_PLUS_1X_K = [
    [5375.0,    0.0, 2040.0],
    [   0.0, 5375.0, 1530.0],
    [   0.0,    0.0,    1.0],
]
_SAMSUNG_S25_PLUS_1X_D = [0.10, -0.22, 0.0001, -0.0002, 0.15]
_SAMSUNG_S25_PLUS_1X_RES = [4080, 3060]

_SAMSUNG_S25_PLUS_UW_K = [
    [2910.0,    0.0, 2000.0],
    [   0.0, 2910.0, 1500.0],
    [   0.0,    0.0,    1.0],
]
_SAMSUNG_S25_PLUS_UW_D = [0.28, -0.62, 0.0003, -0.0005, 0.44]
_SAMSUNG_S25_PLUS_UW_RES = [4000, 3000]

PERFIS_INTRINSICOS.update({
    "SAMSUNG_S25_PLUS_1X": (
        _SAMSUNG_S25_PLUS_1X_K,
        _SAMSUNG_S25_PLUS_1X_D,
        _SAMSUNG_S25_PLUS_1X_RES,
    ),
    "SAMSUNG_S25_PLUS_UW": (
        _SAMSUNG_S25_PLUS_UW_K,
        _SAMSUNG_S25_PLUS_UW_D,
        _SAMSUNG_S25_PLUS_UW_RES,
    ),
})

# Xiaomi Mi 9T — câmara principal 1× de 48 MP (resolução máxima 8000×6000).
# Perfil aproximado para seleção imediata; para medições rigorosas deve ser
# substituído por uma calibração intrínseca real através do perfil CALIBRADO.
_XIAOMI_MI_9T_1X_K = [
    [5950.0,    0.0, 4000.0],
    [   0.0, 5950.0, 3000.0],
    [   0.0,    0.0,    1.0],
]
_XIAOMI_MI_9T_1X_D = [0.10, -0.20, 0.0001, -0.0002, 0.14]
_XIAOMI_MI_9T_1X_RES = [8000, 6000]

PERFIS_INTRINSICOS["XIAOMI_MI_9T_1X"] = (
    _XIAOMI_MI_9T_1X_K,
    _XIAOMI_MI_9T_1X_D,
    _XIAOMI_MI_9T_1X_RES,
)


# ─────────────────────────────────────────────
#  ESQUEMA DOS PARÂMETROS
# ─────────────────────────────────────────────
ESQUEMA: list[dict] = [

    # ── CÂMARA ──────────────────────────────────────────────────────
    {
        "chave":     "modo_localizacao_robo",
        "categoria": "CAMARA",
        "descricao": "Método de localização do robô: marcadores ArUco ou círculos vermelho/azul",
        "unidade":   "",
        "default":   "ARUCO",
        "tipo":      "str",
        "min":       None,
        "max":       None,
        "opcoes":    ["ARUCO", "COR"],
    },
    {
        "chave":     "perfil_camara",
        "categoria": "CAMARA",
        "descricao": "Perfil de câmara: iPhone 16, Samsung S25 Plus, Xiaomi Mi 9T, calibrado, ou câmara externa configurável",
        "unidade":   "",
        "default":   "IPHONE16_1X",
        "tipo":      "str",
        "min":       None,
        "max":       None,
        "opcoes":    [
            "IPHONE16_1X",
            "IPHONE16_05X",
            "SAMSUNG_S25_PLUS_1X",
            "SAMSUNG_S25_PLUS_UW",
            "XIAOMI_MI_9T_1X",
            "CALIBRADO",
            "EXTERNO",
        ],
    },
    {
        "chave":     "altura_camara_m",
        "categoria": "CAMARA",
        "descricao": "Altura da câmara ao solo (necessária para correção de paralaxe)",
        "unidade":   "m",
        "default":   2.5,
        "tipo":      "float",
        "min":       0.5,
        "max":       10.0,
    },
    {
        "chave":     "altura_bola_m",
        "categoria": "CAMARA",
        "descricao": "Raio da bola de ténis — desloca o centro detetado para o solo",
        "unidade":   "m",
        "default":   0.0325,
        "tipo":      "float",
        "min":       0.005,
        "max":       0.20,
    },
    {
        "chave":     "bola_ancora_px",
        "categoria": "CAMARA",
        "descricao": "Ponto da bounding box usado para mapear a bola no chão",
        "unidade":   "",
        "default":   "BOTTOM_CENTER",
        "tipo":      "str",
        "min":       None,
        "max":       None,
        "opcoes":    ["BOTTOM_CENTER", "LOWER_CENTER_80", "CENTER"],
    },
    {
        "chave":     "altura_aruco_m",
        "categoria": "CAMARA",
        "descricao": "Altura dos marcadores ArUco ao solo (topo do robô)",
        "unidade":   "m",
        "default":   0.12,
        "tipo":      "float",
        "min":       0.005,
        "max":       1.0,
    },

    # ── CÂMARA EXTERNA (ativo só quando perfil = "EXTERNO") ─────────
    {
        "chave":     "ext_fx",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Distância focal horizontal fx (px)",
        "unidade":   "px",
        "default":   1000.0,
        "tipo":      "float",
        "min":       100.0,
        "max":       20000.0,
    },
    {
        "chave":     "ext_fy",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Distância focal vertical fy (px)",
        "unidade":   "px",
        "default":   1000.0,
        "tipo":      "float",
        "min":       100.0,
        "max":       20000.0,
    },
    {
        "chave":     "ext_cx",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Ponto principal cx (px) — normalmente largura/2",
        "unidade":   "px",
        "default":   960.0,
        "tipo":      "float",
        "min":       0.0,
        "max":       8000.0,
    },
    {
        "chave":     "ext_cy",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Ponto principal cy (px) — normalmente altura/2",
        "unidade":   "px",
        "default":   540.0,
        "tipo":      "float",
        "min":       0.0,
        "max":       6000.0,
    },
    {
        "chave":     "ext_k1",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Coeficiente de distorção radial k1",
        "unidade":   "",
        "default":   0.0,
        "tipo":      "float",
        "min":       -5.0,
        "max":       5.0,
    },
    {
        "chave":     "ext_k2",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Coeficiente de distorção radial k2",
        "unidade":   "",
        "default":   0.0,
        "tipo":      "float",
        "min":       -5.0,
        "max":       5.0,
    },
    {
        "chave":     "ext_k3",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Coeficiente de distorção radial k3",
        "unidade":   "",
        "default":   0.0,
        "tipo":      "float",
        "min":       -5.0,
        "max":       5.0,
    },
    {
        "chave":     "ext_p1",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Coeficiente de distorção tangencial p1",
        "unidade":   "",
        "default":   0.0,
        "tipo":      "float",
        "min":       -1.0,
        "max":       1.0,
    },
    {
        "chave":     "ext_p2",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Coeficiente de distorção tangencial p2",
        "unidade":   "",
        "default":   0.0,
        "tipo":      "float",
        "min":       -1.0,
        "max":       1.0,
    },
    {
        "chave":     "ext_res_w",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Largura de referência para os intrínsecos (px)",
        "unidade":   "px",
        "default":   1920,
        "tipo":      "int",
        "min":       320,
        "max":       8000,
    },
    {
        "chave":     "ext_res_h",
        "categoria": "CAMARA",
        "descricao": "[EXTERNO] Altura de referência para os intrínsecos (px)",
        "unidade":   "px",
        "default":   1080,
        "tipo":      "int",
        "min":       240,
        "max":       6000,
    },

    # ── MODO DE OPERAÇÃO ────────────────────────────────────────────
    {
        "chave":     "modo_operacao",
        "categoria": "MODO",
        "descricao": "Modo de varrimento: FAIXAS (por zona) ou GLOBAL (trajetória TSP)",
        "unidade":   "",
        "default":   "FAIXAS",
        "tipo":      "str",
        "min":       None,
        "max":       None,
        "opcoes":    ["FAIXAS", "GLOBAL"],
    },
    {
        "chave":     "k_min_global",
        "categoria": "MODO",
        "descricao": "Mínimo de bolas estáveis em todo o campo para disparar o modo GLOBAL",
        "unidade":   "bolas",
        "default":   10,
        "tipo":      "int",
        "min":       1,
        "max":       500,
    },
    {
        "chave":     "n_obs_min_estavel",
        "categoria": "MODO",
        "descricao": "Observações coerentes necessárias para confirmar permanentemente uma bola",
        "unidade":   "frames",
        "default":   3,
        "tipo":      "int",
        "min":       1,
        "max":       30,
    },
    {
        "chave":     "tempo_min_estavel_s",
        "categoria": "MODO",
        "descricao": "Tempo mínimo para confirmar permanentemente uma bola",
        "unidade":   "s",
        "default":   0.7,
        "tipo":      "float",
        "min":       0.1,
        "max":       10.0,
    },
    {
        "chave":     "velocidade_max_bola_parada_cm_s",
        "categoria": "MODO",
        "descricao": "Velocidade máxima para tratar uma bola como parada",
        "unidade":   "cm/s",
        "default":   8.0,
        "tipo":      "float",
        "min":       0.5,
        "max":       100.0,
    },
    {
        "chave":     "raio_confirmar_bola_parada_cm",
        "categoria": "MODO",
        "descricao": "Raio máximo onde a candidata deve permanecer para ser confirmada como parada",
        "unidade":   "cm",
        "default":   3.0,
        "tipo":      "float",
        "min":       0.5,
        "max":       20.0,
    },
    {
        "chave":     "tempo_expirar_bola_s",
        "categoria": "MODO",
        "descricao": "Legado: bolas detetadas ficam persistentes até recolha/reset",
        "unidade":   "s",
        "default":   1.5,
        "tipo":      "float",
        "min":       0.2,
        "max":       10.0,
    },

    # ── FAIXAS ─────────────────────────────────────────────────────
    {
        "chave":     "largura_robo_cm",
        "categoria": "FAIXAS",
        "descricao": "Largura do robô usada como espaçamento entre trajetórias horizontais",
        "unidade":   "cm",
        "default":   40.0,
        "tipo":      "float",
        "min":       5.0,
        "max":       200.0,
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
        "descricao": "Percentagem mínima do total para uma faixa disparar (modo FAIXAS)",
        "unidade":   "%",
        "default":   25.0,
        "tipo":      "float",
        "min":       5.0,
        "max":       100.0,
    },
    {
        "chave":     "k_min",
        "categoria": "DISPARO",
        "descricao": "Número mínimo absoluto de bolas para uma faixa disparar (modo FAIXAS)",
        "unidade":   "bolas",
        "default":   4,
        "tipo":      "int",
        "min":       1,
        "max":       50,
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
        "chave":     "modo_correcao_orientacao_esp32",
        "categoria": "CONTROLADOR",
        "descricao": "Estratégia de correção angular do ESP32: primeira trajetória lenta, sempre lenta ou sempre rápida",
        "unidade":   "",
        "default":   "PRIMEIRA_DEVAGAR",
        "tipo":      "str",
        "min":       None,
        "max":       None,
        "opcoes":    ["PRIMEIRA_DEVAGAR", "SEMPRE_DEVAGAR", "SEMPRE_RAPIDO"],
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
        "chave":     "modo_supervisao_udp",
        "categoria": "REDE",
        "descricao": "Modo de teste UDP: trajetória completa aberta ou supervisão ponto a ponto",
        "unidade":   "",
        "default":   "PONTO_A_PONTO",
        "tipo":      "str",
        "min":       None,
        "max":       None,
        "opcoes":    ["TRAJETORIA_COMPLETA", "PONTO_A_PONTO"],
    },
    {
        "chave":     "porta_udp",
        "categoria": "REDE",
        "descricao": "Porta UDP do ESP32 para receber metas e autorizações do supervisor",
        "unidade":   "",
        "default":   5005,
        "tipo":      "int",
        "min":       1024,
        "max":       65535,
    },
    {
        "chave":     "porta_udp_feedback",
        "categoria": "REDE",
        "descricao": "Porta UDP local onde o servidor recebe eventos do ESP32",
        "unidade":   "",
        "default":   5006,
        "tipo":      "int",
        "min":       1024,
        "max":       65535,
    },
    {
        "chave":     "supervisor_desvio_angulo_graus",
        "categoria": "CONTROLADOR",
        "descricao": "Erro angular persistente que faz o supervisor mandar parar/corrigir",
        "unidade":   "°",
        "default":   25.0,
        "tipo":      "float",
        "min":       1.0,
        "max":       90.0,
    },
    {
        "chave":     "supervisor_desvio_distancia_cm",
        "categoria": "CONTROLADOR",
        "descricao": "Aumento de distância ao alvo que conta como desvio em movimento",
        "unidade":   "cm",
        "default":   25.0,
        "tipo":      "float",
        "min":       1.0,
        "max":       200.0,
    },
    {
        "chave":     "supervisor_reenviar_meta_s",
        "categoria": "CONTROLADOR",
        "descricao": "Intervalo para reenviar orient_goal enquanto aguarda o ESP32",
        "unidade":   "s",
        "default":   0.75,
        "tipo":      "float",
        "min":       0.1,
        "max":       5.0,
    },
    {
        "chave":     "supervisor_timeout_feedback_s",
        "categoria": "CONTROLADOR",
        "descricao": "Tempo sem feedback do ESP32 antes de avisar falha de comunicação",
        "unidade":   "s",
        "default":   6.0,
        "tipo":      "float",
        "min":       1.0,
        "max":       60.0,
    },
    {
        "chave":     "supervisor_assentamento_orientacao_s",
        "categoria": "CONTROLADOR",
        "descricao": "Espera após orientation_done antes de validar com um novo frame ArUco",
        "unidade":   "s",
        "default":   0.45,
        "tipo":      "float",
        "min":       0.05,
        "max":       3.0,
    },
    {
        "chave":     "supervisor_margem_liberacao_angulo_graus",
        "categoria": "CONTROLADOR",
        "descricao": "Margem adicional para não bloquear por pequeno jitter angular",
        "unidade":   "°",
        "default":   3.0,
        "tipo":      "float",
        "min":       0.0,
        "max":       15.0,
    },
    {
        "chave":     "supervisor_max_reenvios_orient_goal",
        "categoria": "CONTROLADOR",
        "descricao": "Número máximo de reenvios de orient_goal antes de bloquear o segmento",
        "unidade":   "",
        "default":   20,
        "tipo":      "int",
        "min":       1,
        "max":       200,
    },
    {
        "chave":     "raio_recolha_bola_cm",
        "categoria": "TRAJETORIA",
        "descricao": "Raio para remover uma bola quando o robô passa perto dela durante a trajetória",
        "unidade":   "cm",
        "default":   18.0,
        "tipo":      "float",
        "min":       2.0,
        "max":       80.0,
    },
    {
        "chave":     "guardar_resultados_disco",
        "categoria": "PERFORMANCE",
        "descricao": "Guardar JSONs de posições em resultados/posicoes (0=desligado, 1=ligado)",
        "unidade":   "",
        "default":   0,
        "tipo":      "int",
        "min":       0,
        "max":       1,
    },
    {
        "chave":     "guardar_imagens_debug",
        "categoria": "PERFORMANCE",
        "descricao": "Guardar frames anotados e top-down em worker assíncrono (0=desligado, 1=ligado)",
        "unidade":   "",
        "default":   1,
        "tipo":      "int",
        "min":       0,
        "max":       1,
    },
    {
        "chave":     "intervalo_guardar_imagens_s",
        "categoria": "PERFORMANCE",
        "descricao": "Intervalo mínimo entre imagens debug guardadas",
        "unidade":   "s",
        "default":   5.0,
        "tipo":      "float",
        "min":       0.1,
        "max":       60.0,
    },
    {
        "chave":     "processamento_largura_px",
        "categoria": "PERFORMANCE",
        "descricao": "Largura enviada ao processamento em produção (0 mantém resolução original)",
        "unidade":   "px",
        "default":   960,
        "tipo":      "int",
        "min":       0,
        "max":       3840,
    },
    {
        "chave":     "aruco_largura_px",
        "categoria": "PERFORMANCE",
        "descricao": "Largura interna usada só para deteção ArUco (0 usa frame recebido)",
        "unidade":   "px",
        "default":   1280,
        "tipo":      "int",
        "min":       0,
        "max":       3840,
    },
    {
        "chave":     "aruco_usar_clahe",
        "categoria": "PERFORMANCE",
        "descricao": "Aplicar CLAHE antes do ArUco (0=mais rápido, 1=mais robusto com má iluminação)",
        "unidade":   "",
        "default":   0,
        "tipo":      "int",
        "min":       0,
        "max":       1,
    },
    {
        "chave":     "aruco_persistencia_s",
        "categoria": "PERFORMANCE",
        "descricao": "Tempo para manter o último ArUco válido quando um frame falha",
        "unidade":   "s",
        "default":   0.35,
        "tipo":      "float",
        "min":       0.0,
        "max":       2.0,
    },
    {
        "chave":     "aruco_suavizacao",
        "categoria": "PERFORMANCE",
        "descricao": "Suavização temporal dos centros ArUco (0=sem memória, 0.8=muito suave)",
        "unidade":   "",
        "default":   0.35,
        "tipo":      "float",
        "min":       0.0,
        "max":       0.95,
    },
]


# ─────────────────────────────────────────────
#  PERSISTÊNCIA
# ─────────────────────────────────────────────
def _defaults() -> dict[str, Any]:
    return {entry["chave"]: entry["default"] for entry in ESQUEMA}


def carregar() -> dict[str, Any]:
    """
    Carrega parametros.json. Cria com defaults se não existir.
    Preenche chaves em falta com defaults (compatibilidade evolutiva).
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
        cfg = _defaults()
        guardar(cfg)
        return cfg

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
    PASTA_CONFIG.mkdir(parents=True, exist_ok=True)
    with open(FICH_PARAMS, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


# ─────────────────────────────────────────────
#  INTRÍNSECOS — API PÚBLICA
# ─────────────────────────────────────────────
def obter_intrinsics(cfg: dict[str, Any]) -> tuple:
    """
    Devolve (K, D, resolucao) para o perfil configurado.

      K          — matriz 3×3 (numpy float64)
      D          — vetor de distorção [k1,k2,p1,p2,k3] (numpy float64)
      resolucao  — [largura, altura] de referência (lista de int)

    Para perfil "EXTERNO" usa os parâmetros ext_* do dicionário.
    Para "IPHONE16_1X" e "IPHONE16_05X" usa os valores embutidos.
    """
    perfil = str(cfg.get("perfil_camara", "IPHONE16_1X")).upper()

    if perfil in PERFIS_INTRINSICOS:
        k_lst, d_lst, res = PERFIS_INTRINSICOS[perfil]
        K = np.array(k_lst, dtype=np.float64)
        D = np.array(d_lst, dtype=np.float64)
        return K, D, res

    if perfil == "CALIBRADO":
        calib_path = BASE_PATH / "resultados" / "calibracao" / "intrinsicos_camera.json"
        if not calib_path.exists():
            raise FileNotFoundError(
                f"Intrínsecos calibrados não encontrados: {calib_path}"
            )
        with open(calib_path, encoding="utf-8") as f:
            calib = json.load(f)
        K = np.array(calib["K"], dtype=np.float64)
        D = np.array(calib["D"], dtype=np.float64)
        return K, D, [int(v) for v in calib["resolucao"]]

    # Perfil EXTERNO
    K = np.array([
        [cfg.get("ext_fx", 1000.0),              0.0, cfg.get("ext_cx", 960.0)],
        [            0.0, cfg.get("ext_fy", 1000.0), cfg.get("ext_cy", 540.0)],
        [            0.0,              0.0,                               1.0],
    ], dtype=np.float64)
    D = np.array([
        cfg.get("ext_k1", 0.0),
        cfg.get("ext_k2", 0.0),
        cfg.get("ext_p1", 0.0),
        cfg.get("ext_p2", 0.0),
        cfg.get("ext_k3", 0.0),
    ], dtype=np.float64)
    res = [int(cfg.get("ext_res_w", 1920)), int(cfg.get("ext_res_h", 1080))]
    return K, D, res


# ─────────────────────────────────────────────
#  VALIDAÇÃO
# ─────────────────────────────────────────────
def validar_valor(esquema_entry: dict, valor_str: str) -> tuple[bool, Any, str]:
    tipo  = esquema_entry["tipo"]

    if tipo == "str":
        valor = valor_str.strip().upper()
        if not valor:
            return False, None, "valor não pode estar vazio"
        opcoes = esquema_entry.get("opcoes")
        if opcoes and valor not in opcoes:
            return False, None, f"opções válidas: {', '.join(opcoes)}"
        return True, valor, ""

    try:
        if tipo == "int":
            valor = int(valor_str)
        else:
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
    chave   = esquema_entry["chave"]
    valor   = cfg.get(chave, esquema_entry["default"])
    unidade = esquema_entry["unidade"]
    if esquema_entry["tipo"] == "float":
        txt = f"{valor:.3f}"
        if "." in txt:
            txt = txt.rstrip("0").rstrip(".") if txt.rstrip("0").rstrip(".") else txt
            if "." not in txt:
                txt += ".0"
    else:
        txt = str(valor)
    return f"{txt} {unidade}".strip()


def formatar_intervalo(esquema_entry: dict) -> str:
    opcoes = esquema_entry.get("opcoes")
    if opcoes:
        return " | ".join(opcoes)
    if esquema_entry["tipo"] == "str":
        return "qualquer texto"
    mn, mx = esquema_entry["min"], esquema_entry["max"]
    if mn is None and mx is None:
        return "—"
    return f"{mn} a {mx}"


if __name__ == "__main__":
    cfg = carregar()
    K, D, res = obter_intrinsics(cfg)
    print(f"Parâmetros lidos de {FICH_PARAMS}:")
    print(f"\nPerfil câmara : {cfg['perfil_camara']}  (resolução de ref. {res[0]}×{res[1]}px)")
    print(f"K (intrínseca):\n{K}")
    print(f"D (distorção) : {D.tolist()}")
    print()
    cat_atual = None
    for i, entry in enumerate(ESQUEMA, 1):
        if entry["categoria"] != cat_atual:
            cat_atual = entry["categoria"]
            print(f"\n── {cat_atual} ──")
        print(f"  [{i:2d}] {entry['chave']:28s} = {formatar_valor(entry, cfg):22s}  "
              f"({entry['descricao']})")




