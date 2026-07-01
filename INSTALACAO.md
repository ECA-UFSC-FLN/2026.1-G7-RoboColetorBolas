# Instalação num computador novo

Este guia assume Windows 10/11 de 64 bits e Python 3.12.

## 1. Instalar programas

Instala:

1. Git para Windows.
2. Python 3.12 de 64 bits, ativando a opção para adicionar Python ao PATH.
3. Camo Studio, caso a câmara continue a ser o iPhone por USB.
4. Arduino IDE 2.x, caso seja necessário programar o ESP32.

## 2. Clonar o projeto

Abre PowerShell ou Terminal e executa:

```powershell
git clone https://github.com/ECA-UFSC-FLN/2026.1-G7-RoboColetorBolas.git
cd 2026.1-G7-RoboColetorBolas
```

## 3. Criar o ambiente Python

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação, não é necessário alterar políticas: usa
diretamente `.\.venv\Scripts\python.exe` nos comandos seguintes.

## 4. Confirmar instalação e modelos

```powershell
.\.venv\Scripts\python.exe -c "import cv2, numpy, matplotlib, shapely, ultralytics, torch; print('Dependencias OK'); print('CUDA:', torch.cuda.is_available())"
Get-ChildItem .\models\*.pt
```

Devem existir:

- `models/balls_best.pt`: modelo treinado usado para detetar bolas;
- `models/yolov8s.pt`: modelo base do script de treino;
- `models/yolo26n.pt`: modelo base alternativo guardado com o projeto.

O sistema escolhe automaticamente CUDA quando disponível; caso contrário usa CPU.

## 5. Preparar a câmara

1. Liga o iPhone por USB.
2. Abre Camo no iPhone e Camo Studio no computador.
3. Confirma que aparece uma imagem estável no Camo Studio.
4. Mantém iguais zoom, lente, orientação e resolução usados na calibração.

Se a câmara ou a sua posição física mudar, recalibra a homografia pelo menu do sistema.
A homografia incluída no repositório só é válida para a montagem em que foi criada.

## 6. Configurar rede e localização

Revê `resultados/configuracao/parametros.json`:

- `ip_robo`: IP atual do ESP32;
- `porta_udp`: normalmente `5005`;
- `porta_udp_feedback`: normalmente `5006`;
- `modo_localizacao_robo`: `COR` ou `ARUCO`;
- `modo_operacao`: normalmente `GLOBAL`;
- `modo_supervisao_udp`: normalmente `PONTO_A_PONTO`.

No modo `COR`, a marca azul fica à frente e a vermelha atrás.

## 7. Programar o ESP32, se necessário

Abre `_ROBOT/robo_coletor_esp32/robo_coletor_esp32.ino` no Arduino IDE.

Instala pelo gestor de bibliotecas:

- Adafruit MPU6050;
- Adafruit Unified Sensor;
- ArduinoJson.

Instala também o suporte da placa ESP32, confirma SSID/password e seleciona a
placa/porta corretas antes de carregar o firmware.

## 8. Arrancar

Faz duplo clique em `ASTART.bat` ou executa:

```powershell
.\ASTART.bat
```

Para abrir a consola de debug:

```powershell
.\ASTART_DEBUG.bat
```

No menu principal:

1. Revê os parâmetros.
2. Recalibra a homografia se a montagem da câmara não for exatamente a mesma.
3. Inicia produção.

## 9. Diagnóstico rápido

- `Modelo não encontrado`: confirma `models/balls_best.pt`.
- `Nenhuma câmera encontrada`: abre Camo Studio antes do sistema.
- `ESP32 sem feedback`: confirma Wi-Fi, IP e portas 5005/5006.
- `CUDA` indisponível: é normal; o sistema usa CPU, mas a inferência será mais lenta.
- ArUco indisponível em OpenCV: confirma a instalação com
  `python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"` e
  reinstala `opencv-python` a partir de `requirements.txt` se o resultado for `False`.
