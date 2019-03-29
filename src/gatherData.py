#!/usr/bin/python

import threading

print "Press Escape to Quit"

# Global variable
data = None

class getLeapMotionData(threading.Thread): #I don't understand this or the next line
    def run(self):
        self.setup()

    def setup(self):
        global data
        # read leap motion data
        # convert to a list of angles for each finger
        with lock:
            print "Thread one has lock"
            # assign values to shared variable
            data = "Some value"


class getMyoBandData(threading.Thread):
    def run(self):
        global data
        # read myoband data, wait for leap motion data
        with lock:
            print "Thread two has lock"
            # get leap motion data into non-shared variable
            print data
        #output to CSV

lock = threading.Lock()

getLeapMotionData().start()
getMyoBandData().start()
