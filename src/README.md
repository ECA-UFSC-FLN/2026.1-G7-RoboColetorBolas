# Sistema de Recolha de Bolas UFSC/FEUP

Sistema de visao e controlo para detetar bolas de tenis numa quadra, estimar a posicao do robo por marcadores ArUco, gerar trajetorias e enviar comandos ao robo.

## Arranque

Para instalar o projeto num computador novo, segue [INSTALACAO.md](INSTALACAO.md).

Usa os atalhos da raiz:

- `ASTART.bat` arranca o sistema normal.
- `ASTART_DEBUG.bat` arranca com consola de debug.

O ponto de entrada real e `_APP/master_control.py`.

## Pastas

Pastas de codigo:

- `_APP/` orquestra o pipeline e menus principais.
- `_VISION/` captura, detecao YOLO/ArUco e retificacao para metros.
- `_PLANNING/` acumula bolas, decide disparos e gera trajetorias.
- `_CONTROL/` calcula comandos e envia UDP ao robo.
- `_CONFIG/` gere parametros configuraveis em `resultados/configuracao/parametros.json`.
- `_COMMON/` utilitarios partilhados, principalmente logging.
- `_CALIBRATION/` calibracao intrinseca opcional da camara.
- `_TOOLS/` ferramentas auxiliares, como consola de debug.
- `_TRAINING/` treino YOLO.

Pastas de dados e artefactos:

- `DataSet/` dataset YOLO.
- `runs/` saidas do treino YOLO.
- `models/` pesos YOLO necessários para produção e treino.
- `assets/` marcadores ArUco e outros recursos estaticos.
- `resultados/` calibracoes, configuracoes, trajetorias e dumps de debug.
- `_backup_codigo_*` backup automatico feito antes da reorganizacao.

## Fluxo

1. `_VISION/camera_stream.py` captura frames.
2. `_VISION/vision_processor.py` deteta bolas por YOLO e robo por ArUco.
3. `_VISION/court_rectifier.py` converte pixeis para metros com a homografia.
4. `_PLANNING/court_graph_processor.py` acumula bolas e gera trajetorias.
5. `_CONTROL/robot_controller.py` envia comandos ao robo.

Durante a execucao de uma trajetoria, o YOLO e pausado e fica apenas a detecao ArUco.

## Parametros Importantes

Todos vivem em `resultados/configuracao/parametros.json`.

- `tempo_min_estavel_s`: tempo minimo observado antes de uma bola contar como parada.
- `velocidade_max_bola_parada_cm_s`: velocidade maxima para tratar a bola como parada.
- `tempo_expirar_bola_s`: tempo sem observacao antes de remover uma bola temporaria.
- `processamento_largura_px`: largura do frame enviado para processamento. `960` e rapido; `0` usa resolucao original.
- `aruco_largura_px`: largura interna para ArUco. `640` e rapido; aumenta se os marcadores forem pequenos.
- `aruco_usar_clahe`: `0` e mais rapido; `1` ajuda quando a iluminacao e dificil.
- `guardar_resultados_disco`: `0` evita escrita no disco no ciclo principal.
- `guardar_imagens_debug`: grava imagens apenas quando `guardar_resultados_disco=1`.
- `intervalo_guardar_imagens_s`: intervalo minimo entre imagens debug guardadas.

## Calibracao

A calibracao da homografia e obrigatoria. A calibracao intrinseca da camara e opcional.

Para melhorar a precisao da homografia sem calibracao intrinseca:

- usar 6 ou mais pontos espalhados pela quadra;
- evitar pontos quase colineares ou muito concentrados;
- introduzir coordenadas reais em metros com a mesma origem/eixos que queres usar no robo;
- recalibrar sempre que a camara, zoom, resolucao ou posicao fisica mudar.

O sistema guarda uma homografia direta imagem corrigida -> metros, reduzindo erros de escala e offset no caminho de producao.
