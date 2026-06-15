#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ============================================================
// AJUSTES RAPIDOS
// ============================================================

// 0 = teste local sem servidor; 1 = recebe metas por UDP.
#define USE_UDP 0

// Teste local.
// false = anda uma distancia reta; true = segue a lista TEST_POINTS_CM.
const bool TEST_USE_POINTS = true;

// Distancia reta, usada quando TEST_USE_POINTS = false.
const float TEST_DIST_CM = 150.0;

// Pontos em cm, usados quando TEST_USE_POINTS = true.
// O primeiro ponto e a pose inicial assumida do robo.
struct PointCm {
  float x;
  float y;
};

PointCm TEST_POINTS_CM[] = {
  {0.0, 0.0},
  {30.0, 0.0},
  {30.0, 30.0},
  {60.0, 60.0},
  {120.0, 150.0},
  {30.0, 0.0},
  {0.0, 0.0},
};

const int TEST_POINT_COUNT = sizeof(TEST_POINTS_CM) / sizeof(TEST_POINTS_CM[0]);
const bool REPEAT_TEST = false;

// Calibracao principal da distancia.
// Depois de um teste: ticksPerCm = media_de_ticks / distancia_real_cm.
float ticksPerCm = 12.0;

// Correcao fina de distancia.
// Se pediu 150 cm e andou 160 cm: DIST_GAIN ~= 150.0 / 160.0 = 0.9375.
// Se a curta distancia ficar pequena demais, reduz STOP_LEAD_CM antes de mexer aqui.
const float DIST_GAIN = 0.94;
const float STOP_LEAD_CM = 0.7;

// Motores. Se um lado andar ao contrario, troca o respetivo sinal para -1.
const int MOTOR_L_SIGN = 1;
const int MOTOR_R_SIGN = 1;

// PWM. PWM_CRUISE baixo costuma dar menos derrapagem e menos sobrecurso.

const int PWM_MIN_L = 83; // Coloque o valor achado
const int PWM_MIN_R = 107; // Coloque o valor achado
const float FATOR_DIREITO = 1.0321; // Coloque o fator achado
const int PWM_MIN = 120;
const int PWM_CRUISE = 160;
const int PWM_MAX = 255;
const float SLOWDOWN_CM = 18.0;

// PID para virar no sitio.
const float TURN_TOL_DEG = 2.0;
const float TURN_KP = 3.2;
const float TURN_KI = 0.0;
const float TURN_KD = 0.10;

// PID para andar reto.
// TICKS corrige diferenca entre rodas; HEADING corrige desvio angular pela IMU.
const float TICKS_KP = 0.9;
const float TICKS_KI = 0.0;
const float TICKS_KD = 0.02;
const float HEADING_KP = 2.6;
const float HEADING_KI = 0.0;
const float HEADING_KD = 0.08;

// IMU.
const float GYRO_DEADBAND_DPS = 0.6;
const int IMU_CAL_SAMPLES = 500;

// Logs.
const uint32_t LOG_EVERY_MS = 150;

// ============================================================
// PINOS
// ============================================================

const int SDA_PIN = 40;
const int SCL_PIN = 39;

const int MOT_L_A = 11;
const int MOT_L_B = 10;
const int MOT_R_A = 12;
const int MOT_R_B = 13;

const int ENC_L_A = 14;
const int ENC_R_A = 7;

// ============================================================
// UDP
// ============================================================

#if USE_UDP
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "NOME_DA_REDE";
const char* WIFI_PASS = "PASSWORD_DA_REDE";

const uint16_t UDP_CMD_PORT = 5005;  // servidor -> ESP32
const uint16_t UDP_FB_PORT = 5006;   // ESP32 -> servidor

IPAddress serverIp(192, 168, 1, 100);
WiFiUDP udp;
String segmentId;
#endif

// ============================================================
// ESTADO
// ============================================================

Adafruit_MPU6050 mpu;

volatile long ticksL = 0;
volatile long ticksR = 0;

float headingDeg = 0.0;
float gyroBiasZ = 0.0;
float accBiasX = 0.0;
float accBiasY = 0.0;
float accBiasZ = 0.0;
float accX = 0.0;
float accY = 0.0;
float accZ = 0.0;
uint32_t lastImuMs = 0;

struct PID {
  float kp, ki, kd;
  float i = 0.0;
  float prev = 0.0;

  void set(float p, float in, float d) {
    kp = p;
    ki = in;
    kd = d;
  }

  void reset() {
    i = 0.0;
    prev = 0.0;
  }

  float step(float err, float dt, float limit) {
    if (dt <= 0.0) return 0.0;
    i += err * dt;
    float d = (err - prev) / dt;
    prev = err;
    return constrain(kp * err + ki * i + kd * d, -limit, limit);
  }
};

PID pidTurn;
PID pidTicks;
PID pidHeading;

// ============================================================
// ENCODERS
// ============================================================

void IRAM_ATTR isrLeft() { ticksL++; }
void IRAM_ATTR isrRight() { ticksR++; }

long readTicksL() {
  noInterrupts();
  long v = ticksL;
  interrupts();
  return v;
}

long readTicksR() {
  noInterrupts();
  long v = ticksR;
  interrupts();
  return v;
}

void resetTicks() {
  noInterrupts();
  ticksL = 0;
  ticksR = 0;
  interrupts();
}

// ============================================================
// MOTORES
// ============================================================

void motorRaw(int a, int b, int pwm) {
  pwm = constrain(pwm, -255, 255);

  if (pwm > 0) {
    analogWrite(a, pwm);
    analogWrite(b, 0);
  } else if (pwm < 0) {
    analogWrite(a, 0);
    analogWrite(b, -pwm);
  } else {
    analogWrite(a, 0);
    analogWrite(b, 0);
  }
}

void setMotors(int left, int right) {
  motorRaw(MOT_L_B, MOT_L_A, left * MOTOR_L_SIGN);
  motorRaw(MOT_R_B, MOT_R_A, right * MOTOR_R_SIGN);
}

void stopMotors() {
  setMotors(0, 0);
}

void brakeShort() {
  digitalWrite(MOT_L_A, LOW);
  digitalWrite(MOT_L_B, LOW);
  digitalWrite(MOT_R_A, LOW);
  digitalWrite(MOT_R_B, LOW);
  delay(120);
}

// ============================================================
// IMU
// ============================================================

float wrapDeg(float a) {
  while (a > 180.0) a -= 360.0;
  while (a < -180.0) a += 360.0;
  return a;
}

void calibrateImu() {
  sensors_event_t a, g, t;
  float gz = 0.0;
  float ax = 0.0;
  float ay = 0.0;
  float az = 0.0;

  for (int k = 0; k < IMU_CAL_SAMPLES; k++) {
    mpu.getEvent(&a, &g, &t);
    gz += g.gyro.z;
    ax += a.acceleration.x;
    ay += a.acceleration.y;
    az += a.acceleration.z;
    delay(3);
  }

  gyroBiasZ = gz / IMU_CAL_SAMPLES;
  accBiasX = ax / IMU_CAL_SAMPLES;
  accBiasY = ay / IMU_CAL_SAMPLES;
  accBiasZ = (az / IMU_CAL_SAMPLES) - 9.80665;
}

void readImu() {
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);

  uint32_t now = millis();
  float dt = (now - lastImuMs) / 1000.0;
  lastImuMs = now;

  float zDps = (g.gyro.z - gyroBiasZ) * 180.0 / PI;
  if (abs(zDps) > GYRO_DEADBAND_DPS) {
    headingDeg = wrapDeg(headingDeg + zDps * dt);
  }

  accX = a.acceleration.x - accBiasX;
  accY = a.acceleration.y - accBiasY;
  accZ = a.acceleration.z - accBiasZ;
}

// ============================================================
// MOVIMENTO
// ============================================================

int rampBase(float remainingCm) {
  if (remainingCm >= SLOWDOWN_CM) return PWM_CRUISE;
  int x = constrain((int)(remainingCm * 10.0), 0, (int)(SLOWDOWN_CM * 10.0));
  return map(x, 0, (int)(SLOWDOWN_CM * 10.0), PWM_MIN, PWM_CRUISE);
}

float correctedDistanceCm(float cm) {
  return max(0.0f, cm * DIST_GAIN - STOP_LEAD_CM);
}

void turnTo(float targetDeg) {
  pidTurn.reset();
  readImu();

  uint32_t last = millis();
  while (true) {
    readImu();
    float err = wrapDeg(targetDeg - headingDeg);
    if (abs(err) <= TURN_TOL_DEG) break;

    uint32_t now = millis();
    float dt = (now - last) / 1000.0;
    last = now;

    int pwm = (int)abs(pidTurn.step(err, dt, PWM_MAX));
    pwm = constrain(pwm, PWM_MIN, PWM_MAX);

    if (err > 0) setMotors(-pwm, pwm);
    else setMotors(pwm, -pwm);

    delay(8);
  }

  stopMotors();
  brakeShort();
}

void driveStraight(float cm) {
  float controlCm = correctedDistanceCm(cm);
  long targetTicks = lround(controlCm * ticksPerCm);

  resetTicks();
  pidTicks.reset();
  pidHeading.reset();
  readImu();

  float headingRef = headingDeg;
  uint32_t last = millis();
  uint32_t lastLog = 0;

  Serial.println();
  Serial.println("drive_start");
  Serial.print("cmd_cm=");
  Serial.println(cm, 2);
  Serial.print("control_cm=");
  Serial.println(controlCm, 2);
  Serial.print("target_ticks=");
  Serial.println(targetTicks);
  Serial.print("heading_ref=");
  Serial.println(headingRef, 2);

  while (true) {
    readImu();

    long l = readTicksL();
    long r = readTicksR();
    long avg = (l + r) / 2;
    long rem = targetTicks - avg;
    if (rem <= 0) break;

    uint32_t now = millis();
    float dt = (now - last) / 1000.0;
    last = now;

    float remCm = rem / ticksPerCm;
    int base = rampBase(remCm);

    float errTicks = (float)(l - r);
    float errHead = wrapDeg(headingRef - headingDeg);

    float corrTicks = pidTicks.step(errTicks, dt, 45.0);
    float corrHead = pidHeading.step(errHead, dt, 35.0);
    int corr = (int)(corrTicks - corrHead);

    // Aplica o mínimo independente e multiplica o direito pelo fator de correção
    int pwmL = constrain(base - corr, PWM_MIN_L, PWM_MAX);
    int pwmR = constrain((base + corr) * FATOR_DIREITO, PWM_MIN_R, PWM_MAX);
    setMotors(pwmL, pwmR);

    if (now - lastLog >= LOG_EVERY_MS) {
      lastLog = now;
      Serial.print("l=");
      Serial.print(l);
      Serial.print(" r=");
      Serial.print(r);
      Serial.print(" h=");
      Serial.print(headingDeg, 1);
      Serial.print(" eh=");
      Serial.print(errHead, 1);
      Serial.print(" ax=");
      Serial.print(accX, 2);
      Serial.print(" ay=");
      Serial.print(accY, 2);
      Serial.print(" pwm=");
      Serial.print(pwmL);
      Serial.print(",");
      Serial.println(pwmR);
    }

    delay(5);
  }

  stopMotors();
  brakeShort();
  delay(350);

  long lf = readTicksL();
  long rf = readTicksR();
  float meanTicks = (lf + rf) / 2.0;

  Serial.println("drive_end");
  Serial.print("ticks_l=");
  Serial.print(lf);
  Serial.print(" ticks_r=");
  Serial.print(rf);
  Serial.print(" mean=");
  Serial.println(meanTicks, 1);
  Serial.println("ticksPerCm = mean / distancia_real_cm");
}

void goToPoint(float x0, float y0, float x1, float y1) {
  float dx = x1 - x0;
  float dy = y1 - y0;
  float targetHeading = atan2(dy, dx) * 180.0 / PI;
  float dist = sqrt(dx * dx + dy * dy);

  turnTo(targetHeading);
  driveStraight(dist);
}

void runPointTest() {
  if (TEST_POINT_COUNT < 2) {
    Serial.println("point_test_skip");
    return;
  }

  Serial.println();
  Serial.println("point_test_start");
  Serial.print("points=");
  Serial.println(TEST_POINT_COUNT);

  for (int i = 0; i < TEST_POINT_COUNT - 1; i++) {
    PointCm a = TEST_POINTS_CM[i];
    PointCm b = TEST_POINTS_CM[i + 1];

    Serial.println();
    Serial.print("segment=");
    Serial.print(i + 1);
    Serial.print("/");
    Serial.println(TEST_POINT_COUNT - 1);
    Serial.print("from=");
    Serial.print(a.x, 1);
    Serial.print(",");
    Serial.print(a.y, 1);
    Serial.print(" to=");
    Serial.print(b.x, 1);
    Serial.print(",");
    Serial.println(b.y, 1);

    goToPoint(a.x, a.y, b.x, b.y);
    delay(500);
  }

  Serial.println("point_test_end");
}

// ============================================================
// UDP
// ============================================================

#if USE_UDP
void udpBegin() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }

  udp.begin(UDP_CMD_PORT);
  Serial.println();
  Serial.print("wifi_ip=");
  Serial.println(WiFi.localIP());
}

void udpEvent(const char* ev) {
  StaticJsonDocument<192> doc;
  doc["event"] = ev;
  doc["segment_id"] = segmentId;
  doc["heading_deg"] = headingDeg;
  doc["ticks_l"] = readTicksL();
  doc["ticks_r"] = readTicksR();

  char buf[192];
  size_t n = serializeJson(doc, buf);
  udp.beginPacket(serverIp, UDP_FB_PORT);
  udp.write((const uint8_t*)buf, n);
  udp.endPacket();
}

void udpPoll() {
  if (udp.parsePacket() <= 0) return;

  StaticJsonDocument<768> doc;
  if (deserializeJson(doc, udp)) return;

  const char* type = doc["type"] | "";
  segmentId = (const char*)(doc["segment_id"] | "");

  if (!strcmp(type, "orient_goal") || !strcmp(type, "orientation_correction")) {
    turnTo(doc["desired_heading_deg"] | headingDeg);
    udpEvent("orientation_done");
  } else if (!strcmp(type, "move_permission")) {
    driveStraight((doc["distance_m"] | 0.0) * 100.0);
    udpEvent("arrived");
  } else if (!strcmp(type, "stop") || !strcmp(type, "stop_correct")) {
    stopMotors();
    udpEvent("stopped");
  }
}
#endif

// ============================================================
// SETUP / LOOP
// ============================================================

void setup() {
  Serial.begin(115200);

  pinMode(MOT_L_A, OUTPUT);
  pinMode(MOT_L_B, OUTPUT);
  pinMode(MOT_R_A, OUTPUT);
  pinMode(MOT_R_B, OUTPUT);
  stopMotors();

  pinMode(ENC_L_A, INPUT_PULLUP);
  pinMode(ENC_R_A, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrLeft, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrRight, RISING);

  Wire.begin(SDA_PIN, SCL_PIN);
  if (!mpu.begin()) {
    Serial.println("mpu_fail");
    while (true) delay(100);
  }

  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  pidTurn.set(TURN_KP, TURN_KI, TURN_KD);
  pidTicks.set(TICKS_KP, TICKS_KI, TICKS_KD);
  pidHeading.set(HEADING_KP, HEADING_KI, HEADING_KD);

  calibrateImu();
  lastImuMs = millis();

#if USE_UDP
  udpBegin();
#endif

  Serial.println("ready");
}

void loop() {
#if USE_UDP
  udpPoll();
  delay(5);
#else
  if (TEST_USE_POINTS) runPointTest();
  else driveStraight(TEST_DIST_CM);

  if (!REPEAT_TEST) while (true) delay(1000);
  delay(2000);
#endif
}