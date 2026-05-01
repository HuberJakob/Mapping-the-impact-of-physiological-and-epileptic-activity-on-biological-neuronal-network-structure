//Input-output calibration 
// Set stimulation Pin
int TTL1 = 2; 

void setup() {
pinMode(TTL1, OUTPUT);
}
// Set parameter
int baseline_duration = 10000;
int stim_pulses = 1200;
int stim_frequency = 40;
int equilibration_duration = 10000;

void loop() {
  //initiate stimulation Pin
  digitalWrite(TTL1,LOW);
  
  delay(baseline_duration); // record baseline
  // run stimulation
  for (int i=0; i <= stim_pulses; i++){ 
    digitalWrite(TTL1, HIGH);
    delay(1); // 1 ms pulse duration
    digitalWrite(TTL1, LOW);
    delay((1000/stim_frequency)-1); 
  }
  digitalWrite(TTL1,LOW);

  // record post-stimulus baseline
  delay(equilibration_duration);

// only run the code once
  while(1){};
}

