void setup() 
{
  Serial.begin(115200);
  pinMode(13, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:

}

void serialEvent()
{
  delay(10);
  String inputString = "";
  while(Serial.available())
  {
    char character = (char)Serial.read();
    if(character != '\n')
    {
      inputString += character;
    }
  }
  Serial.println("I received: "  + inputString);
  if(inputString == "open")
  {
    digitalWrite(13, HIGH);
  }
  else if(inputString == "close")
  {
    digitalWrite(13, LOW);
  }
}
