#!/usr/bin/python
# to run this, run:
# sudo leapd
# LeapControlPanel
### try LeapControlPanel --showsettings if not working
#

# https://www.blog.pythonlibrary.org/2016/07/28/python-201-a-tutorial-on-threads/

# myo connect came from: http://www.fernandocosentino.net/pyoconnect/

import threading
import sys, time, thread
import math
import numpy as np
from leapInterfaces2 import SampleListener
import Queue
from myoConnectFunctions import *
import csv
sys.path.insert(0, "../lib")
import Leap

print("Beginning")

enableCollection = False
leapMotionDetection = False
q = Queue.LifoQueue(maxsize=1)

fileName = raw_input("specify filename (*.csv)")
if fileName == "dummy":
    print("running dummy session, no csv output")
else:
    if fileName == "" or fileName == "\n":
        fileName = 'output'

fileName += ".csv"
fileName = "trainingData/" + fileName


writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
dataLabels = []
for a in range(8):
    dataLabels.append("emg"+str(a))

finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
#bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
bone_names = ['Proximal', 'Intermediate', 'Distal'] # 3 angles are calculated, which are considered to be at the bases of each bone
for a in finger_names:
    for b in bone_names:
        dataLabels.append(a+"_"+b)



writer.writerow(dataToWrite)


def proc_emg(emg, moving, times=[]):
    if True:
        try:
            leapData = None
            gotData = False
            try:
                ######### Put a time out on everything!!!!
                leapData = q.get_nowait() # is this the last data, or the data put into the buffer initially, not read then everythning else was lost
                gotData = True
            except Queue.Empty:
                print("No hand recognised, data collection paused")
                pass

            if(gotData == True):
                rows,cols = leapData.shape
                leapDataList = []
                for row in range(rows):
                    leapDataList += list(leapData[row,:])
                emgDataList = list(emg)
                dataToWrite = emgDataList+leapDataList
                print("Sampling....")
                #print("Data:{}".format(dataToWrite))
                #self.dataCount += 1
                #if (self.dataCount%100 == 0):
                    #print("{} Samples".format(self.dataCount))
                with open(fileName, mode = 'a') as outputFile:
                    writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(dataToWrite)
            #else:
                #print("gotData = false")
        except KeyboardInterrupt:
                exit()

print("Please put on myo band, and place hand above leapmotion sensor")
print("if leap motion cannot properly track hand, recording will be ")
print("paused and automatically resumed on recognition. Press enter")
print("when ready to connect")
rubbish = raw_input()

# Myo setup
m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
m.add_emg_handler(proc_emg)
print("Connecting to Myo Band")
m.connect()
print("Myo Connected")


# leap setup
lock = threading.Lock()
listener = SampleListener(lock,q)
controller = Leap.Controller()
controller.add_listener(listener)

# Keep this process running until Enter is pressed
print("Press Enter to quit...")
try:
    internalAngles = None
    while True:
        #start = time.time()
        m.run(1)
        #end = time.time()
        #print("F {}".format(1/(end-start)))
        #print("Hi from main thread!")
        #try:
    #        internalAngles = q.get(block=False)
    #        print(internalAngles)
    #    except Queue.Empty:
#            print("Nothing to see here....")
#        time.sleep(0.5)
except KeyboardInterrupt:
    print("time to go to sleep!")
finally:
    # Remove the sample listener when done
    controller.remove_listener(listener)
