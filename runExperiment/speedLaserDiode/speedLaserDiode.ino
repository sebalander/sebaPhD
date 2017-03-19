// http://www.himix.lt/arduino/arduino-and-optical-sensor-tcrt1000/
//Watch video here: https://www.youtube.com/watch?v=EfjO4c1DNDU

int sensor = 0; // voltage measurement 
//int ledPin = 13;
int ledStatus = 0; // 0 if laser blocked; 1 if laser reaches led
// double f; // frecuencia

unsigned long mp; // time of previous measurement
unsigned long m; // time of this measurement
unsigned long tau1; // uncertainty of previous flank
unsigned long tau2; // uncertainty of this flank
unsigned long Tp; // time of previous flank
unsigned long I; // interval between flanks
float r1 = 2*3.14159*0.29*1000000*3.6; // radio menor en metros
float r2 = 2*3.14159*0.31*1000000*3.6; // radio mayo en metros
float v1, v2;

void setup() {
  Serial.begin(115200);
  //pinMode(ledPin,OUTPUT);
}

void loop() {
  sensor = analogRead(0);
  m = micros();
  
  // Serial.println(sensor);
  
  if(sensor > 250) {
    if(ledStatus==0) { // FLANCO POSITIVO
      //digitalWrite(ledPin, HIGH);
      ledStatus = 1;
      I = m - Tp; // tiempo desde el flanco anterior (no defino T2=time2 porque es al pepe)
      tau2 = m - mp; // error de esta medicion
      //Serial.print(tau1);
      //Serial.print(" ");
      //Serial.print(tau2);
      //Serial.print(" ");
      //Serial.println(I);
      v1 = r1/(I+tau1); // velocidad menor
      v2 = r2/(I-tau2); // velocidad meyor
      
      Serial.print(v1,4);
      Serial.print("\t");
      Serial.println(v2,4);
      
      Tp = m; // guardo para proximo flanco positivo
      tau1 = tau2;
      }
      // if ledStatus==1 do nothing, laser already reaches led
    }
  else {
    if(ledStatus==1) { // FLANCO NEGATIVO
      //digitalWrite(ledPin, LOW);
      ledStatus=0;
      }
    // if ledStatus==0 do nothing, laser already blocked
    }
  mp=m;
}
