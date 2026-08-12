/*
 * PMM_Firmware.ino  v2.2
 * WiFi + HTTPS version — sends raw samples to Railway backend.
 *
 * Data flow:
 *   ESP32 --HTTPS POST--> Railway /pumps/{id}/raw --> computes FFT --> Frontend polls
 *
 * Binary payload format (5120 bytes):
 *   [acc_x int16 x 512][acc_y int16 x 512][acc_z int16 x 512][mic uint16 x 1024]
 *
 * Hardware (PMM_V1):
 *   MIC  SPU0410  -> GPIO5
 *   SDA  BMA400   -> GPIO8
 *   SCL  BMA400   -> GPIO9
 */

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

// -- WiFi credentials ----------------------------------------------------------
const char* WIFI_SSID = "HUJI-guest";
const char* WIFI_PASS = "";

// -- Backend URL (Railway HTTPS) -----------------------------------------------
const char* BACKEND_URL = "https://web-production-c30b0.up.railway.app";
// const char* BACKEND_URL = "http://10.0.0.3:9000/pumps/pump_01/raw";
const char* PUMP_ID     = "pump_01";

// -- Pin map -------------------------------------------------------------------
#define MIC_PIN     5
#define SDA_PIN     8
#define SCL_PIN     9

// -- ACC config ----------------------------------------------------------------
#define ACC_RATE    800
#define ACC_N       512

// -- MIC config ----------------------------------------------------------------
#define MIC_RATE    8000
#define MIC_N       1024

// -- Binary payload size -------------------------------------------------------
#define PAYLOAD_SIZE (ACC_N * 3 * sizeof(int16_t) + MIC_N * sizeof(uint16_t))

// -- BMA400 registers ----------------------------------------------------------
#define BMA400_ADDR         0x14
#define BMA400_CHIP_ID_REG  0x00
#define BMA400_ACC_DATA_REG 0x04
#define BMA400_ACC_CONFIG0  0x19
#define BMA400_ACC_CONFIG1  0x1A

// -- Buffers -------------------------------------------------------------------
int16_t  axBuf[ACC_N];
int16_t  ayBuf[ACC_N];
int16_t  azBuf[ACC_N];
uint16_t micBuf[MIC_N];
uint8_t payload[PAYLOAD_SIZE];

// -- BMA400 --------------------------------------------------------------------
void bma400Write(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(BMA400_ADDR);
  Wire.write(reg); Wire.write(val);
  Wire.endTransmission();
}

bool bma400Init() {
  Wire.beginTransmission(BMA400_ADDR);
  Wire.write(BMA400_CHIP_ID_REG);
  Wire.endTransmission(false);
  Wire.requestFrom(BMA400_ADDR, 1);
  if (!Wire.available()) return false;
  if (Wire.read() != 0x90) return false;

  bma400Write(BMA400_ACC_CONFIG0, 0x02);
  bma400Write(BMA400_ACC_CONFIG1, 0x4B);

  delay(2);
  return true;
}

void bma400Read(int16_t &ax, int16_t &ay, int16_t &az) {
  Wire.beginTransmission(BMA400_ADDR);
  Wire.write(BMA400_ACC_DATA_REG);
  Wire.endTransmission(false);
  Wire.requestFrom(BMA400_ADDR, 6);
  if (Wire.available() < 6) { ax = ay = az = 0; return; }
  uint8_t b[6];
  for (int i = 0; i < 6; i++) b[i] = Wire.read();

  ax = (int16_t)((b[1] << 4) | (b[0] >> 4));
  ay = (int16_t)((b[3] << 4) | (b[2] >> 4));
  az = (int16_t)((b[5] << 4) | (b[4] >> 4));

  if (ax & 0x800) ax |= 0xF000;
  if (ay & 0x800) ay |= 0xF000;
  if (az & 0x800) az |= 0xF000;
}

// -- WiFi connect --------------------------------------------------------------
void wifiConnect() {
  Serial.printf("# Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500);
    Serial.print(".");
    tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n# WiFi OK  IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n# WiFi FAIL  check credentials");
  }
}

// -- Send raw data to backend (HTTPS) ------------------------------------------
bool sendToBackend() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("# WiFi disconnected  reconnecting");
    wifiConnect();
    if (WiFi.status() != WL_CONNECTED) return false;
  }


  size_t offset = 0;

  memcpy(payload + offset, axBuf, ACC_N * sizeof(int16_t));
  offset += ACC_N * sizeof(int16_t);

  memcpy(payload + offset, ayBuf, ACC_N * sizeof(int16_t));
  offset += ACC_N * sizeof(int16_t);

  memcpy(payload + offset, azBuf, ACC_N * sizeof(int16_t));
  offset += ACC_N * sizeof(int16_t);

  memcpy(payload + offset, micBuf, MIC_N * sizeof(uint16_t));

  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  String url = String(BACKEND_URL) + "/pumps/" + PUMP_ID + "/raw";
  http.begin(client, url);
  http.addHeader("Content-Type", "application/octet-stream");
  http.setTimeout(10000);

  int code = http.POST(payload, PAYLOAD_SIZE);
  http.end();

  if (code == 200) {
    return true;
  } else {
    Serial.printf("# POST failed: %d\n", code);
    return false;
  }
}

// -- Setup ---------------------------------------------------------------------
void setup() {
  Serial.begin(115200);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  for (int i = 0; i < 64; i++) { analogRead(MIC_PIN); delayMicroseconds(5); }

  bool ok = bma400Init();
  Serial.println(ok ? "# BMA400 OK" : "# BMA400 FAIL  check wiring");
  Serial.printf("# ACC: N=%d Fs=%d Hz | MIC: N=%d Fs=%d Hz\n",
                ACC_N, ACC_RATE, MIC_N, MIC_RATE);
  Serial.printf("# Payload size: %d bytes\n", PAYLOAD_SIZE);

  wifiConnect();
  Serial.println("# READY");
}

// -- Main loop -----------------------------------------------------------------
void loop() {

  // -- 1) Capture ACC  512 samples @ 800 Hz -----------------------------------
  {
    const uint32_t interval_us = 1000000UL / ACC_RATE;
    uint32_t t0 = micros();
    for (int i = 0; i < ACC_N; i++) {
      bma400Read(axBuf[i], ayBuf[i], azBuf[i]);
      while ((micros() - t0) < (uint32_t)(i + 1) * interval_us);
    }
  }

  // -- 2) Capture MIC  1024 samples @ 8000 Hz ---------------------------------
  {
    const uint32_t interval_us = 1000000UL / MIC_RATE;
    uint32_t t0 = micros();
    for (int i = 0; i < MIC_N; i++) {
      micBuf[i] = (uint16_t)analogRead(MIC_PIN);
      while ((micros() - t0) < (uint32_t)(i + 1) * interval_us);
    }
  }

  // -- 3) Send raw data to backend via HTTPS ----------------------------------
  bool ok = sendToBackend();
  Serial.printf("# Frame sent: %s\n", ok ? "OK" : "FAIL");
}
