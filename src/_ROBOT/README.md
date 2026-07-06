# ESP32 - Robo coletor

Sketch Arduino:

- `robo_coletor_esp32/robo_coletor_esp32.ino`

## Configuracao rapida

No sketch, ajustar:

- `WIFI_SSID`
- `WIFI_PASS`
- `USE_UDP`
- `VISION_TURN_SIGN`

Para teste integrado com o servidor, manter:

```cpp
#define USE_UDP 1
```

O ESP32 escuta comandos em `UDP_CMD_PORT` (`5005` por default) e envia feedback para `UDP_FB_PORT` (`5006` por default).

O IP do servidor e atualizado automaticamente a partir do primeiro pacote UDP recebido. Assim, normalmente so e necessario configurar no servidor o IP do ESP32.

## Protocolo usado

Servidor -> ESP32:

- `orient_goal`
- `orientation_correction`
- `move_permission`
- `stop`
- `stop_correct`
- `arrived_ok`
- `arrived_bad`

ESP32 -> servidor:

- `orientation_done`
- `arrived`
- `stopped`

O controlo de motores, encoders e IMU fica no ESP32. O servidor apenas envia metas/permissoes e valida pela visao.

## Modos de teste do supervisor

Em `resultados/configuracao/parametros.json`:

```json
"modo_supervisao_udp": "TRAJETORIA_COMPLETA"
```

Neste modo o servidor envia um unico pacote `trajectory_full` com a pose
inicial e todos os waypoints. O ESP32 executa localmente cada segmento,
sem aguardar validacao do servidor entre pontos, e no fim envia
`trajectory_done`.

Para voltar ao protocolo supervisionado:

```json
"modo_supervisao_udp": "PONTO_A_PONTO"
```

Neste modo a orientação é incremental:

1. O servidor envia o erro angular medido pela visão.
2. O ESP32 roda no máximo `MAX_VISION_TURN_STEP_DEG` (10 graus).
3. O ESP32 para e envia `orientation_done`.
4. O servidor espera o assentamento configurado e um novo frame ArUco.
   Se estiver alinhado, envia `move_permission`;
   caso contrário, envia outro pequeno passo.
5. Depois do movimento, o ESP32 envia `arrived` e permanece parado.
6. O servidor responde `arrived_ok` ou volta a orientar/corrigir o mesmo alvo.

Pacotes UDP com `seq` antigo são ignorados pelo ESP32 para evitar executar
correções acumuladas enquanto os motores estavam ocupados.

Perto da tolerância, o ESP32 não tenta zerar todo o erro: corrige apenas até
metade da margem angular. Se o erro já estiver dentro da tolerância, não dá
nenhum toque adicional e apenas pede nova validação.

Os passos angulares de 10 graus são usados apenas na primeira trajetória
após o ESP32 arrancar. Quando o servidor envia uma nova `trajectory_id`, o
ESP32 desativa `firstTrajectoryCalibration` e passa a usar a correção angular
normal, mantendo a validação ponto a ponto pelo servidor.

O parâmetro `modo_correcao_orientacao_esp32` permite escolher a estratégia:

- `PRIMEIRA_DEVAGAR`: passos de até 10 graus apenas na primeira trajetória;
- `SEMPRE_DEVAGAR`: passos de até 10 graus em todas as trajetórias;
- `SEMPRE_RAPIDO`: correção angular completa em todas as trajetórias.

PWM temporariamente reforçado para vencer atrito:

- mínimo esquerdo: 105
- mínimo direito: 125
- mínimo geral/rotação: 140
- cruzeiro: 175

## Se roda mas nao anda

Se o Serial Monitor mostrar muitos `orient_goal` / `orientation_correction` e nunca aparecer `move_permission`, a comunicacao esta OK, mas a visao ainda nao validou a orientacao.

O sketch imprime:

- `vision_heading`
- `desired`
- `error`
- `local_target`

Se o robo estiver claramente a rodar para o lado errado, trocar:

```cpp
const int VISION_TURN_SIGN = 1;
```

para:

```cpp
const int VISION_TURN_SIGN = -1;
```
