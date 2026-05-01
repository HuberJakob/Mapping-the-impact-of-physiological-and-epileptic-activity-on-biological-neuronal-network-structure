//[ChatGPT (OpenAI) and Claude (Antrophic) were used as
//assistance for code generation, which was reviewed and verified by the Creator]

// include libraries
#include <Arduino.h>
#include <Wire.h>
#include "RTClib.h"
#include <SPI.h>
#include "SD.h"
#include "digitalWriteFast.h"
// initialize time module
RTC_DS3231 rtc;

//set stimulation pins
const int stimPin = 9; // stimulation output pin
const int channelSwitchPin = 7; // channel switch output pin
const int chipSelectPin = 10; // SD card chip select pin
//set Input pin to check if channel 1 is selected
const int channel1ActivePin = 5;
const int debugLED = 3;
// set stimulation variables
const int maxChannels = 16; // maximum number of channels 8 or 16 - check channel switch on Bluebox
int channel = 1; // currently selected channel
int nextChannel = 1; // next channel to stimulate from protocol
unsigned long stimDuration = 1000; // duration of stimulation in µs
const int ttlPulseDurationMs = 20; // duration of TTL pulse to change channel
bool stimulatedFlag = true;

// declaration of file management variables
File protocolFile;
File logFile;
String line;
String protocolFileName = "protocol.csv";
String logFileName = "log.csv";
// declare variables to read the protocol 
String fields[10];
String channelStr;
String timeStr;
String repetitionsStr;
String intervalStr;
int repetitions;
float intervalSeconds;
int year;
int month;
int day;
int hour;
int minute;
int second;
// find next possible stimulation
DateTime preStimTime(2025, 10, 1, 23, 59, 59) ; // pseudo time

// === Channel stimulation config (from SD-card protocol) ===
struct ChannelStimConfig {
  int channel;
  String mode;  // "burst" or "single"
  float p1; // pulse- or burst-count
  float p2; // pulses per burst
  float p3; // interstimulus frequency
  float p4; // interburst interval
};
ChannelStimConfig channelConfigs[maxChannels];
int configCount = 0;
bool sdAvailable = false;


//------------------------------------------------------------
// emergency blink debug LED
void error_blink() {
  Serial.println("Entered Error-mode");
  while (1) {
    // Blink in SOS pattern
    for (int i = 0; i < 3; i++) {
      digitalWrite(debugLED, HIGH);
      delay(200);
      digitalWrite(debugLED, LOW);
      delay(200);
    }
    delay(400);
    for (int i = 0; i < 3; i++) {
      digitalWrite(debugLED, HIGH);
      delay(600);
      digitalWrite(debugLED, LOW);
      delay(600);
    }
    for (int i = 0; i < 3; i++) {
      digitalWrite(debugLED, HIGH);
      delay(200);
      digitalWrite(debugLED, LOW);
      delay(200);
    }

    Serial.println("SOS blink cycle complete");
    delay(1000); // pause before repeating
  }
}

//------------------------------------------------------------
// Align Bluebox relay board stimulation channel with variable channel 0
void channel1_align() {
  const int maxAttempts = 3; // in case channel 0 was not found on the first cycle due to setup issues
  int channelsSwitched = 0;
  //iterate through the channel unitl chanel 1 is aligned
  while (digitalRead(channel1ActivePin) == HIGH && channelsSwitched < maxAttempts*maxChannels) { // INPUT_PULLUP pin == LOW if LED of channel 0 is powered -> Channel 0 found
    digitalWrite(channelSwitchPin, HIGH);
    delay(ttlPulseDurationMs);
    digitalWrite(channelSwitchPin, LOW);
    delay(ttlPulseDurationMs);
    channelsSwitched++;
  }
  if(digitalRead(channel1ActivePin) == HIGH){
    Serial.print("Channel 1 alignment was NOT successful");
    error_blink();
    while(1);
  }
  channel = 1;
}


void splitString(const String &line, char delimiter, String parts[], int maxParts) {
  // split a string by ';' to read the parameter of the protocol.
  int start = 0;
  int partIndex = 0;
  for (unsigned int i = 0; i < line.length() && partIndex < maxParts; i++) { // iterate through every letter of the line
    if (line[i] == delimiter) {                                              // split at ";"
      parts[partIndex++] = line.substring(start, i); 
      start = i + 1;
    }
  }
  if (partIndex < maxParts) parts[partIndex++] = line.substring(start);
  for (; partIndex < maxParts; partIndex++) parts[partIndex] = ""; // fill empty parts with nothing to avoid error
}

void loadChannelStimConfig() {
  // open the protocol from the SD card
  File f = SD.open("protocol.csv", FILE_READ);
  if (!f) {
    Serial.println("ERROR: protocol.csv not found!");
    error_blink();
    return;
  }

  String header = f.readStringUntil('\n');  // skip header row
  configCount = 0;
  // read the parameter from the protocol
  while (f.available() && configCount < maxChannels) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;

    String parts[6];
    splitString(line, ',', parts, 6); 

    channelConfigs[configCount].channel = parts[0].toInt();
    channelConfigs[configCount].mode    = parts[1];
    channelConfigs[configCount].p1      = parts[2].toFloat(); 
    channelConfigs[configCount].p2      = parts[3].toFloat();
    channelConfigs[configCount].p3      = parts[4].toFloat();
    channelConfigs[configCount].p4      = parts[5].toFloat();

    configCount++;
  }
  f.close();

  // Sort by ascending channel number
  for (int i = 0; i < configCount - 1; i++) {
    for (int j = i + 1; j < configCount; j++) {
      if (channelConfigs[j].channel < channelConfigs[i].channel) {
        ChannelStimConfig temp = channelConfigs[i];
        channelConfigs[i] = channelConfigs[j];
        channelConfigs[j] = temp;
      }
    }
  }

  Serial.println("Loaded channel stimulation config:");
  for (int i = 0; i < configCount; i++) {
    Serial.print("CH ");
    Serial.print(channelConfigs[i].channel);
    Serial.print("  MODE=");
    Serial.print(channelConfigs[i].mode);
    Serial.print("  PARAMS=");
    Serial.print(channelConfigs[i].p1); Serial.print(",");
    Serial.print(channelConfigs[i].p2); Serial.print(",");
    Serial.print(channelConfigs[i].p3); Serial.print(",");
    Serial.println(channelConfigs[i].p4);
  }
}

inline void checkSDCard() {
  // check if SDCard is inserted and accessable
    if (sdAvailable) {
        // test a very tiny operation
        File f = SD.open("test.tmp", FILE_WRITE);
        if (!f) {
            Serial.println("SD REMOVED");
            sdAvailable = false;
            return;
        }
        f.close();
        SD.remove("test.tmp");
        return;
    }

    // If unavailable, attempt init (safe because this is outside stimulation timing)
    if (SD.begin(chipSelectPin)) {
        sdAvailable = true;
        Serial.println("SD INSERTED");
    }
}


// ------------------------------------------------------------
// Setup
// ------------------------------------------------------------
void setup() {
  // start serial for debugging
  Serial.begin(9600);
  unsigned long start = millis();
  while (!Serial && millis() - start < 3000); // try to start serial for max 3 seconds

  pinMode(debugLED, OUTPUT); // builtin LED for screenless debugging if initialisation went wrong

  // SD card + protocol file Initialization
  if (!SD.begin(chipSelectPin)) {
    Serial.println("Card failed, or not present");
    error_blink();
    while (1);
  }
  Serial.println("Card initialized.");

  // Initialize log file
  logFile = SD.open(logFileName, FILE_WRITE);
  if (!logFile) {
    Serial.println("Failed to open log file");
    error_blink();
    while (1);
  }

  // Initialize time module
  Wire.begin();
  if (!rtc.begin()) {
    Serial.println("Couldn't find RTC");
    error_blink();
    while (1);
  }
  // rtc lost Power if coin battery empty
  if (rtc.lostPower()) {
    Serial.println("RTC lost power!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    logFile.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RTC lost power - schedule incorrect!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    logFile.println("Check coin battery");
    logFile.flush();
  }
  // log restart
  startTime = rtc.now();
  logFile.print("Code Reset at:");
  logFile.println(rtc.now().timestamp());
  logFile.flush();

  // Initialize Pins
  pinMode(channel1ActivePin, INPUT_PULLUP);
  pinMode(stimPin, OUTPUT);
  digitalWrite(stimPin, LOW);
  pinMode(channelSwitchPin, OUTPUT);
  digitalWrite(channelSwitchPin, LOW);
  digitalWrite(debugLED, LOW);

  // Align to channel 1
  channel1_align();
  Serial.println("Aligned to channel 1.");
  loadChannelStimConfig();

}

// ------------------------------------------------------------
// Channel change function
void changeChannel(int newChannel) {
  if (newChannel < 1 || newChannel > maxChannels) return; // sanity check

  int steps = (newChannel - channel + maxChannels) % maxChannels;  // how many steps to move
  Serial.print("Changing from channel ");
  Serial.print(channel);
  Serial.print(" to ");
  Serial.print(newChannel);
  Serial.print(" (");
  Serial.print(steps);
  Serial.println(" steps)");
  // iterate through the channel until selected channel was reached
  for (int i = 0; i < steps; i++) {
    // Pulse the relay to move one step
    digitalWrite(channelSwitchPin, HIGH);
    delay(ttlPulseDurationMs);
    digitalWrite(channelSwitchPin, LOW);
    delay(ttlPulseDurationMs);

    // Update the channel count (wrap around)
    channel = (channel % maxChannels) + 1;

    Serial.print("Stepped to channel ");
    Serial.println(channel);

    // Repeated check if channels are aligned with the stimulation distribution box
    if (channel == 1) {
      delay(5); // short settle delay before reading
      if (digitalRead(channel1ActivePin) == HIGH) { // HIGH = not active (INPUT_PULLUP)
        Serial.println("⚠️ Channel mismatch detected while passing channel 1!");
        error_blink(); // stops execution
      } else {
        Serial.println("✅ Verified channel 1 active.");
      }
    }
  }

  Serial.print("✅ Final channel: ");
  Serial.println(channel);
}

// ------------------------------------------------------------
// Stimulation
// 
void stimulateChannel(int channelToStim, int repetitions, float pulse_frequency_hz ) {
  unsigned long pulseWidthUs = stimDuration; // stimDuration is µs
  float intervalS = 1.0f / pulse_frequency_hz;
  unsigned long intervalUs = (unsigned long) (intervalS * 1e6f);  
  // Log the stimulation train onset
  if(sdAvailable){
    logFile.print("Stimulating channel ");
    logFile.print(channelToStim);
    logFile.print(" at ");
    logFile.println(rtc.now().timestamp());
    logFile.flush();
    Serial.print("Stimulated at:");
    Serial.println(rtc.now().timestamp());
  }
  // Perform pulses using microsecond-precise delays
  for (int i = 0; i < repetitions; ++i) {
    digitalWriteFast(stimPin, HIGH);
    delayMicroseconds(0.95*pulseWidthUs); //delayMicroseconds showed 5% offset on oscilloscope -> adjusted to 95% to actually get 1ms pulsewidth
    digitalWriteFast(stimPin, LOW);
    delayMicroseconds(0.95*(intervalUs-pulseWidthUs));
  }
}
void stimulateBurstTrain(int ch, int bursts_per_train, int pulses_per_burst, float pulse_frequency_hz, float inter_burst_interval_s) {

  unsigned long pulseWidthUs = stimDuration; //us
  float pulseIntervalS = 1.0f / pulse_frequency_hz;   // seconds between pulse onsets
  unsigned long pulseIntervalUs = (unsigned long) (pulseIntervalS * 1e6f);
  long interBurstMs = (long) (inter_burst_interval_s * 1000.0f);
  unsigned long interBurstUsRem = (unsigned long) ((inter_burst_interval_s * 1e6f) - (pulseIntervalUs*(pulses_per_burst)));

  // Log
  if (sdAvailable){
    logFile.print("Burst-train stimulation channel ");
    logFile.print(ch);
    logFile.print(" at ");
    logFile.println(rtc.now().timestamp());
    logFile.flush();
  }

  for (int b = 0; b < bursts_per_train; ++b) {
    // single burst: pulses at pulse_frequency_hz
    for (int p = 0; p < pulses_per_burst; ++p) {
      digitalWriteFast(stimPin, HIGH);
      delayMicroseconds(0.95*pulseWidthUs); //delayMicroseconds showed 5% offset on oscilloscope -> adjusted to 95% to actually get 1ms pulsewidth
      digitalWriteFast(stimPin, LOW);
      delayMicroseconds(0.95*(pulseIntervalUs-pulseWidthUs));
    }
    delayMicroseconds(0.95*interBurstUsRem); 

  }

  // finalize
  stimulatedFlag = true;
}
// ------------------------------------------------------------
// Main loop
// ------------------------------------------------------------
void loop() {
  checkSDCard();
  startBlock = millis();  // inter-train interval start
  Serial.print("Start Block: ");
  Serial.println(startBlock);

  for (int i = 0; i < configCount; i++) {
    int ch = channelConfigs[i].channel;

    // change to the correct channel
    changeChannel(ch);

    if (channelConfigs[i].mode == "burst") {
          stimulateBurstTrain(
              ch,
              (int)channelConfigs[i].p1,  // bursts_per_train
              (int)channelConfigs[i].p2,  // pulses_per_burst
              channelConfigs[i].p3,       // pulse freq
              channelConfigs[i].p4        // interburst interval
          );

    }
    else if (channelConfigs[i].mode == "single") {

          stimulateChannel(
            ch,
            (int)channelConfigs[i].p1,  // repetitions
            channelConfigs[i].p2        // frequency
          );
    }
  }
  burstTrainCycleCounter++;
  // Return to channel 1 after cycle
  changeChannel(1);
 
  endBlock = millis();
  unsigned long blockDuration = endBlock - startBlock; // Time for all trains for all channels
  const unsigned long cyclePeriodMs = 48000UL; // change to 28800000UL for iTBS and 48000UL for cTBS

  if (blockDuration < cyclePeriodMs) {
    delay(cyclePeriodMs - blockDuration); // wait until next cycle
  } else {
    Serial.print("Warning: block overran by ");
    Serial.println(blockDuration - cyclePeriodMs);
  }
}

