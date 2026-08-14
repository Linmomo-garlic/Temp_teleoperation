/*
 * ADKeyboard V2: S->GPIO0, VCC->3.3, GND->G
 * 五键共用一根模拟线。黄键 S1 ≈2530–2640mV；绿键更高、仍低于空闲。
 * 白键贴近满量程，ADC 顶满认不出。
 * 按下沿发一行：S1=黄键锁主臂；S2=绿键从臂夹爪。
 */

const int PIN_S = 0;
const int IDLE_MV = 2880;
const int S1_MAX_MV = 2660;   // 黄键
const int S2_MAX_MV = 2800;   // 绿键
const int DEBOUNCE_MS = 30;
const int AVG_N = 4;

bool down = false;
bool sent = false;
unsigned long edgeMs = 0;
int pendingKey = 0;

int readMv() {
  long sum = 0;
  for (int i = 0; i < AVG_N; i++) {
    sum += analogReadMilliVolts(PIN_S);
    delay(1);
  }
  return (int)(sum / AVG_N);
}

int classifyKey(int mv) {
  if (mv >= IDLE_MV) return 0;
  if (mv < S1_MAX_MV) return 1;   // 黄 S1
  if (mv < S2_MAX_MV) return 2;   // 绿 S2
  return 0;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  analogSetPinAttenuation(PIN_S, ADC_11db);
  pinMode(PIN_S, INPUT);
  delay(400);
  Serial.println("BTN_READY S1=YELLOW S2=GREEN");
}

void loop() {
  int mv = readMv();
  unsigned long now = millis();
  int key = classifyKey(mv);

  if (!down && key != 0) {
    down = true;
    sent = false;
    pendingKey = key;
    edgeMs = now;
    Serial.print("PRESS MV=");
    Serial.print(mv);
    Serial.print(" KEY=S");
    Serial.println(key);
  }
  if (down && !sent && (now - edgeMs) >= DEBOUNCE_MS) {
    int k = classifyKey(mv);
    if (k == 0) k = pendingKey;
    if (k == 1) Serial.println("S1");
    else if (k == 2) Serial.println("S2");
    sent = true;
  }
  if (down && mv > IDLE_MV) {
    down = false;
    pendingKey = 0;
    Serial.println("RELEASE");
  }
}
