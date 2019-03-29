#include "string.h"

void setup() 
{
  Serial.begin(9600);
  pinMode(13, HIGH);
}

int counter = 0;

void loop() 
{
  Serial.println(++counter);
  delay(1000);
}

void serialEvent()
{
  delay(10);
  String input = "";
  while(Serial.available())
  {
    char inChar = (char)Serial.read();
    if(inChar != '\n')
    {
      input.concat(inChar);
      //Serial.println(input);
    }
  }
  //Serial.print(input);
  String onString = "on";
  String offString = "off";
  if(input.compareTo(onString) == 0)
  {
    //Serial.println("Turning LED On");
    digitalWrite(13, HIGH);
  }
  
  if(input.compareTo(offString) == 0)
  {
    //Serial.println("Turning LED Off");
    digitalWrite(13, LOW);
  }
}
