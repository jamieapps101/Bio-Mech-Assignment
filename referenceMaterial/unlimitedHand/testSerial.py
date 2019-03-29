#!/usr/bin/python3
import time
import serial

ser = serial.Serial(port='/dev/ttyUSB0',baudrate=115200)  # open serial port
print(ser.name)         # check which port was really used
while(1 == 1):
    output = ser.read_until()
    print(output.decode("utf-8"))

ser.close()             # close port
