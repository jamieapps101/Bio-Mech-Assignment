#!/usr/bin/python

from myoConnectFunctions import *
import sys
import csv

m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)

fileName = raw_input("specify filename (*.csv)")
if fileName == "" or fileName == "\n":
    fileName = 'output'
fileName += ".csv"

def proc_emg(emg, moving, times=[]):
    with open(fileName, mode = 'a') as outputFile:
        writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        try:
            print(emg)
            writer.writerow(emg)
        except KeyboardInterrupt:
            pass

m.add_emg_handler(proc_emg)
m.connect()

try:
    while True:
        m.run(1)
except KeyboardInterrupt:
    pass
finally:
    m.disconnect()
    print()
