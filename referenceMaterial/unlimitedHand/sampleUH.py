#!/usr/bin/python3
import time
import serial
import csv
from io import StringIO
import re

name = input("Specify Serial Port")
if name == "" or name == "\n":
    name = '/dev/ttyUSB0'

ser = serial.Serial(port=name,baudrate=115200)  # open serial port
print(ser.name)         # check which port was really used
fileName = input("specify filename (*.csv)")
if fileName == "" or fileName == "\n":
    fileName = 'output'
fileName += ".csv"

with open(fileName, mode = 'w') as outputFile:
    writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    try:
        while(1 == 1):
            output = ser.read_until()
            outputString = output.decode("utf-8")
            #print(outputString)
            data = re.findall("[0-9]+",outputString)
            print("Extracted:")
            print(data)
            writer.writerow(data)


    except KeyboardInterrupt:
        pass

ser.close()             # close port
