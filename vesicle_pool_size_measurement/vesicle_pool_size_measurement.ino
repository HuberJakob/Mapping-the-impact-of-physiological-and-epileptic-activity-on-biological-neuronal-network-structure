// This stimulation protocol was provided by Dahlmanns and Dahlmanns [1] and adjusted for the vesicle pool size measurement
//[1] M. Dahlmanns and J. K. Dahlmanns, “Synaptic vesicle pool monitoring
//    with synapto-phluorin,” in Synaptic Vesicles: Methods and Protocols, J.
//    Dahlmanns and M. Dahlmanns, Eds. New York, NY: Springer US, 2022,
//    pp. 181–192. doi: 10.1007/978-1-0716-1916-2_14.


//This section defines the pins on the Arduino. We specify the intigers (int) TTL1 through TTL4.
//If you expanding this to require 6 channels, uncomment the last two lines here and add additional code below where needed.
int TTL1 = 2; // electrical stimulation
int TTL2 = 4; // perfusion
// int TTL3 = 7; // trigger input
//int TTL4 = 8;
//int TTL5 = 12;
//int TTL6 = 13; //Though, not recommend, as there are slight differences in this pin. If you must use pin 13 as a digital input, set its pinMode() to INPUT and use an external pull down resistor.

//This tells the Arduino to treat the pins as output. In this example, all channels are output. For an example with input (closed loop), see the Arduino-Simple_Closed_Loop.ino file section.
void setup() {
pinMode(TTL1, OUTPUT);
//pinMode(TTL2, OUTPUT);
//pinMode(TTL3, INPUT); // third connection could be used as synchronizing trigger input
}

// Define the length of time (in ms) for each condition. And define all variables.
int baseline_duration = 15000;
int RRP_pulses = 40;
int RRP_frequency = 20;
int equilibration_duration = 30000;
int RecP_pulses = 1200;  
int RecP_frequency = 40; 
int perfusion_duration = 20000;

//This is the code that the Arduino loops through. It will start on condition 1 and then move through the rest of the conditions and then auto restart back at condition 1.
//It writes each pin as High/ON or Low/Off for each condition, and the waits for the duration of condition (set above). 
void loop() {
    //Condition 1 - Baseline
  digitalWrite(TTL1,LOW);
  //digitalWrite(TTL2,LOW);
  delay(baseline_duration);
  
    //Condition 2 - RRP stimulation
  for (int i=0; i <= RRP_pulses; i++){ // repeat the following RRP_pulses times
    digitalWrite(TTL1, HIGH);
    delay(1); // 1 ms pulse duration
    digitalWrite(TTL1, LOW);
    delay((1000/RRP_frequency)-1); // pause between pulses according to RRP_frequency
  }

    //Condition 3 - Equilibration
  digitalWrite(TTL1,LOW);
  //digitalWrite(TTL2,LOW);
  delay(equilibration_duration);

    //Condition 4 - RecP stimulation
  for (int i=0; i <= RecP_pulses; i++){ // repeat the following RecP_pulses times
    digitalWrite(TTL1, HIGH);
    delay(1); // 1 ms pulse duration
    digitalWrite(TTL1, LOW);
    delay((1000/RecP_frequency)-1); // pause between pulses according to RecP_frequency
  }

    //Condition 5 - Equilibration
  digitalWrite(TTL1,LOW);
  //digitalWrite(TTL2,LOW);
  delay(equilibration_duration);  

  // Condition 6 - Ammonium cholride perfusion
  digitalWrite(TTL1, LOW);
  //digitalWrite(TTL2, HIGH);
  delay(perfusion_duration);


  while(1){};
}

