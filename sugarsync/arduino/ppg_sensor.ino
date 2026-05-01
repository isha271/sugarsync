/*
 * SugarSync — NIR-PPG Signal Acquisition
 * Arduino Uno (ATmega328P)
 *
 * Hardware:
 *   - NIR-PPG clip sensor (660nm + 940nm LEDs, photodiode)
 *   - LM358 dual op-amp (analog front end)
 *   - 16x2 LCD (I2C, address 0x27)
 *   - Piezoelectric buzzer (pin 8, optional)
 *
 * Output:
 *   - Serial @ 115200 baud: "TIMESTAMP,NIR_ADC,RED_ADC\n"
 *   - LCD: real-time HR display
 */

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ── Pin Assignments ────────────────────────────────────────────────────────
#define NIR_PIN     A0    // NIR (940nm) photodiode output
#define RED_PIN     A1    // Red (660nm) photodiode output
#define BUZZER_PIN  8     // Optional piezo feedback

// ── LCD ───────────────────────────────────────────────────────────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ── Sampling ──────────────────────────────────────────────────────────────
const int    SAMPLE_RATE_HZ  = 100;   // 100 Hz → 10ms per sample
const int    SAMPLE_INTERVAL = 1000 / SAMPLE_RATE_HZ;  // ms
unsigned long lastSampleTime = 0;

// ── Peak Detection (real-time HR) ─────────────────────────────────────────
const int    WINDOW_SIZE     = 100;   // 1s rolling window
int          nirBuffer[100];
int          bufIdx          = 0;
unsigned long lastPeakTime   = 0;
float        heartRate       = 0.0;
int          peakThreshold   = 512;   // adaptive

// ── Session Counter ───────────────────────────────────────────────────────
unsigned long sampleCount    = 0;

// ──────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial);

  // LCD init
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("SugarSync v1.0");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  delay(1500);
  lcd.clear();

  // Buzzer startup chirp
  pinMode(BUZZER_PIN, OUTPUT);
  tone(BUZZER_PIN, 880, 100);
  delay(150);
  tone(BUZZER_PIN, 1100, 100);
  delay(150);
  noTone(BUZZER_PIN);

  // Print CSV header
  Serial.println("timestamp_ms,nir_adc,red_adc");

  lcd.setCursor(0, 0);
  lcd.print("HR: -- bpm");
  lcd.setCursor(0, 1);
  lcd.print("Samples: 0");
}

// ──────────────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = now;

    // Read ADC
    int nirVal = analogRead(NIR_PIN);
    int redVal = analogRead(RED_PIN);

    // Store in ring buffer
    nirBuffer[bufIdx] = nirVal;
    bufIdx = (bufIdx + 1) % WINDOW_SIZE;

    // Emit CSV line
    Serial.print(now);
    Serial.print(',');
    Serial.print(nirVal);
    Serial.print(',');
    Serial.println(redVal);

    sampleCount++;

    // ── Adaptive peak detection for HR ──────────────────────────────────
    // Update threshold: rolling mean + 10% offset
    long sum = 0;
    for (int i = 0; i < WINDOW_SIZE; i++) sum += nirBuffer[i];
    int rollingMean = sum / WINDOW_SIZE;
    peakThreshold   = rollingMean + (int)(rollingMean * 0.08);

    // Simple peak trigger: signal crosses threshold upward
    static bool inPeak = false;
    if (!inPeak && nirVal > peakThreshold) {
      inPeak = true;
      unsigned long dt = now - lastPeakTime;
      if (dt > 300 && dt < 1500) {          // 40–200 bpm validity
        heartRate     = 60000.0 / (float)dt;
        // Buzz on beat (optional, quiet)
        // tone(BUZZER_PIN, 440, 20);
      }
      lastPeakTime = now;
    } else if (inPeak && nirVal < rollingMean) {
      inPeak = false;
    }

    // ── LCD Update (every 50 samples = 0.5s) ─────────────────────────────
    if (sampleCount % 50 == 0) {
      lcd.setCursor(0, 0);
      lcd.print("HR: ");
      if (heartRate > 0) {
        lcd.print((int)heartRate);
        lcd.print(" bpm   ");
      } else {
        lcd.print("-- bpm  ");
      }

      lcd.setCursor(0, 1);
      lcd.print("n:");
      lcd.print(sampleCount);
      lcd.print("        ");
    }
  }
}
