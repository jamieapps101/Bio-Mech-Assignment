#!/usr/bin/python

# ave sample freq: 58.7015798967
# std dev: 0.00800620253063

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

sampleTimesOriginal = []

def proc_emg(emg, moving, times=[]):
    global sampleTimesOriginal
    sampleTimesOriginal.append(time.time())
    #print("sample")



# Myo setup
m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
m.add_emg_handler(proc_emg)
print("Connecting to Myo Band")
m.connect()
print("Myo Connected")
time.sleep(1)
print("running")
runMode = True
init = time.time()
try:
    while runMode:
        m.run()
        if len(sampleTimesOriginal) > 300:
            runMode = False
        else:
            print(300 - len(sampleTimesOriginal))
            #time.sleep(0.1)
except KeyboardInterrupt:
    print("\ntime to go to sleep!")
    stop = time.time()
    duration = stop - init
    print("{}".format(sampleCounter) + " myo samples in {}ms".format(int(duration*1000)))
finally:
    pass

sampleTimes = sampleTimesOriginal
#print(sampleTimes)
#print(len(sampleTimes))
baseTime = sampleTimes[0]
#print(baseTime)
sampleTimes = np.array(sampleTimes) - baseTime
samplePeriods = np.diff(sampleTimes)
sampleFreqs = 1/np.array(samplePeriods).mean()
aveSamplePeriod = np.array(samplePeriods).mean()
aveSampleFreq = sampleFreqs.mean()
print("ave sample freq: {}".format(sampleFreqs))
exit()
