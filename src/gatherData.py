#!/usr/bin/python
# to run this, run:
# sudo leapd
# LeapControlPanel
### try LeapControlPanel --showsettings if not working
#

# https://www.blog.pythonlibrary.org/2016/07/28/python-201-a-tutorial-on-threads/

# myo connect came from: http://www.fernandocosentino.net/pyoconnect/

# ave sample freq: 52.0070258816
# std dev: 0.00443972074748

# plan of action
# independently sample myo and leapmotion
# produce script to go though slower sampled sensor, combine with data from high sampled sensor using linear interpolation
# pass through fft with window of 0.2-3.0 seconds
# output new data
# train neural net

import pandas as pd
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
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sampleTimesOriginal = []
# samples = 0
#
# print("Beginning")
#
# enableCollection = True
# leapMotionDetection = False
# q = Queue.LifoQueue(maxsize=1)
#
# myoFileName = "trainingData/myoOutput.csv"
# leapFilename = "trainingData/leapOutput.csv"
# with open(myoFileName, "w+") as outputFile:
#     writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     dataLabels = []
#     for a in range(8):
#         dataLabels.append("emg"+str(a))
#     writer.writerow(dataLabels)




# averageList = []
sampleCounter = 0
dataStore = []

def proc_emg(emg, moving, times=[]):
    global dataStore
    global sampleCounter
    #print("myo running")
    sampleCounter+= 1
    dataStore.append(list(emg))
#
#
# print("Please put on myo band, and place hand above leapmotion sensor")
# print("if leap motion cannot properly track hand, recording will be ")
# print("paused and automatically resumed on recognition. Press enter")
# print("when ready to connect")
# rubbish = raw_input()

# Myo setup
m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
m.add_emg_handler(proc_emg)
print("Connecting to Myo Band")
m.connect()
print("Myo Connected")
#
#
# # leap setup
# lock = threading.Lock()
# listener = SampleListener(lock,q)
# controller = Leap.Controller()
# controller.add_listener(listener)

# Keep this process running until Enter is pressed
print("Press Enter to quit...")
init = time.time()
try:
    while True:
        m.run(1)
        if len(dataStore) >= 100:
            temp = dataStore
            # with open(myoFileName, mode = 'a') as outputFile:
            #     writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            #     for data in temp:
            #         writer.writerow(data)

except KeyboardInterrupt:
    print("\ntime to go to sleep!")
    stop = time.time()
    duration = stop - init
    print("{}".format(sampleCounter) + " myo samples in {}ms".format(int(duration*1000)))
finally:
    pass
    #controller.remove_listener(listener)


# now to process the data
def aveFreq(data):
    myoTimes = data[:,0]
    myoTimes = np.diff(myoTimes)
    myoTimes = [time for time in myoTimes if time < (myoTimes.mean()+2*myoTimes.std()) and time > (myoTimes.mean()-2*myoTimes.std())] # remove top and bottom 2.2% of values
    myoAveSampleTimes = np.array(myoTimes).mean()
    myoAveFreq = 1/(myoAveSampleTimes * 1000)
    return myoAveFreq

myoData = pd.read_csv("trainingData/myoOutput.csv")
myoFs = aveFreq(myoData)
print("Myo average sampling freq = {}Hz".format(myoFs))
leapData = pd.read_csv("trainingData/leapOutput.csv")
leapFs = aveFreq(leapData)
print("Myo average sampling freq = {}Hz".format(leapFs))
#
# combinedData = []
#
# def interpolateData(preserveTime, superSample):
#     combinedData = []
#     leapData = preserveTime
#     myoData = superSample
#     for leapRow in leapData:
#         leapRowTime = leapRow[0]
#         myoRowCount,myoColCount = myoData.shape
#         for rowIndex in (range(myoRowCount)-1):
#             myoRowTime = myoData[rowIndex,0]
#             nextMyoRowTime = myoData[rowIndex+1,0]
#             if (myoRowTime <= leapRowTime) and (nextMyoRowTime >= leapRowTime): # ie we've found the samples either side of the sample in the leap data
#                 myoRowData = myoData[rowIndex,1:]
#                 nextMyoRowData = myoData[rowIndex+1,1:]
#                 myoGradient = (np.array(nextMyoRowData) - np.array(myoRowData))/(nextMyoRowTime-myoRowTime)
#                 interpolatedMyoData = (myoGradient * (leapRowTime-myoRowTime)) + np.array(myoRowData)
#                 combinedData.append(list(leapRow) + list(interpolatedMyoData))
#             if myoRowTime >= leapRowTime and nextMyoRowTime >= leapRowTime: # somehow we've gone past it
#                 break
#
#
# if myoFs > leapFs:
#     print("leap sampled slower, super sampling myo")
#     combinedData = interpolateData(leapData, myoData)
# else:
#     print("myo sampled slower, super sampling leap")
#     combinedData = interpolateData(myoData, leapData)
#
# dataToWrite = np.array(combinedData[0])
#
# for row in combinedData[1:]:
#     dataToWrite = np.vstack((dataToWrite,row))
#
# np.savetxt("combinedData.csv", dataToWrite, delimiter=",")
