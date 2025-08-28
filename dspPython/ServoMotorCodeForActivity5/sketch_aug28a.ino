#include <ESP32Servo.h>

Servo myServo;
int servoPin = 18;     // choose a safe GPIO pin for ESP32
int servoAngle = 90;   // start at center

void setup() {
  Serial.begin(9600);
  myServo.attach(servoPin);
  myServo.write(servoAngle);
}

void loop() {
  if (Serial.available() > 0) {
    int angle = Serial.parseInt();
    if (angle >= 0 && angle <= 180) {
      servoAngle = angle;
      myServo.write(servoAngle);
    }
  }
}
