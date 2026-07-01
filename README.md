# Coletor de Bolas UFSC/FEUP

Sistema de visão, planeamento e controlo para detetar bolas de ténis, estimar
a pose do robô, gerar uma trajetória e supervisionar a execução pelo ESP32.

## Instalação

Segue o tutorial completo em [INSTALACAO.md](INSTALACAO.md).

Resumo para Windows:

```powershell
git clone https://github.com/ECA-UFSC-FLN/2026.1-G7-RoboColetorBolas.git
cd 2026.1-G7-RoboColetorBolas\src
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ..\requirements.txt
.\ASTART.bat
```

## Estrutura

- `src/`: aplicação Python, firmware ESP32, configuração e modelos YOLO.
- `hardware/`: documentação de hardware.
- `docs/`: documentação e entregas do projeto.

O modelo usado em produção está em `src/models/balls_best.pt`.
