#!/usr/bin/python
# to run this, run:
# sudo leapd
# sudo -H LeapControlPanel
#
#

# https://www.blog.pythonlibrary.org/2016/07/28/python-201-a-tutorial-on-threads/

import threading
import sys, time, thread
sys.path.insert(0, "../lib")
import Leap
import math
import numpy as np
from leapInterfaces import SampleListener
import Queue
from myoConnectFunctions import *
import csv

print("Beginning")

enableCollection = False
leapMotionDetection = False

def emgCallback(emg, moving, times=[]):
    with open(fileName, mode = 'a') as outputFile:
        writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        try:
            print(emg)
            writer.writerow(emg)
        except KeyboardInterrupt:
            pass

fileName = raw_input("specify filename (*.csv)")
if fileName == "dummy":
    print("running dummy session, no csv output")
else:
    if fileName == "" or fileName == "\n":
        fileName = 'output'
    fileName += ".csv"

print("Please put on myo band, and place hand above leapmotion sensor")
print("if leap motion cannot properly track hand, recording will be ")
print("paused and automatically resumed on recognition. Press enter")
print("when ready to connect")
rubbish = raw_input()
m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
m.add_emg_handler(emgCallback)
print("Connecting to Myo Band")
# Create a leap listener and controller
lock = threading.Lock()
q = Queue.LifoQueue(maxsize=1)
listener = SampleListener(lock,q)
controller = Leap.Controller()
controller.add_listener(listener)

# Keep this process running until Enter is pressed
print("Press Enter to quit...")
try:
    internalAngles = None
    while True:
        print("Hi from main thread!")
        try:
            internalAngles = q.get(block=False)
            print(internalAngles)
        except Queue.Empty:
            print("Nothing to see here....")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("time to go to sleep!")
finally:
    # Remove the sample listener when done
    controller.remove_listener(listener)
