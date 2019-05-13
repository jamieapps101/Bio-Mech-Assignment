#!/usr/bin/python
import sys, thread, time

import threading

import sys
sys.path.insert(0, "../lib")
import Leap
import math
import numpy as np
import Queue
import warnings
import os
import csv


class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def __init__(self, lockVariable, q):
        super(SampleListener,self).__init__()
        self.internalLockVariable = lockVariable
        self.internalQueue = q
        self.filename = "trainingData/leapOutput.csv"
        with open(self.filename, "w+") as outputFile:
            writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            dataLabels = []
            #for a in range(8):
            dataLabels.append("time")

            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            #bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
            bone_names = ['Proximal', 'Intermediate', 'Distal'] # 3 angles are calculated, which are considered to be at the bases of each bone
            for a in finger_names:
                for b in bone_names:
                    dataLabels.append(a+"_"+b)
            writer.writerow(dataLabels)

    def on_init(self, controller):
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE);
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP);
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP);
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE);

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")


    def getAngle(self,a,b):
        #print("a: {}".format(a))
        #print("b: {}".format(b))
        aDotb = 0
        magProd = 0
        g = 0
        angle = 0
        try:
            aDotb = np.dot(a,b)
            #print("aDotb {}".format(aDotb))
            magProd = (np.linalg.norm(a)*np.linalg.norm(b))
            #print("magProd {}".format(magProd))
            g = aDotb/magProd # G
            angle = np.arccos(g)*(180.0/3.14) # getAngle
            angle = angle.reshape((1,1))
        except RuntimeWarning:
            print("aDotb {}".format(aDotb))
            print("magProd {}".format(magProd))
            print("g {}".format(g))
            print("angle {}".format(angle))
        return angle


    def on_frame(self, controller):
                # Get the most recent frame and report some basic information
                print("leap running")
                frame = controller.frame()
                if (len(frame.hands) == 0):
                    pass
                elif (len(frame.hands) > 1):
                    print("There are too many hands in the frame")
                else:
                    for hand in frame.hands: # for each hand, should only have one entry by this point
                        handAngles = None
                        for finger in hand.fingers: # for each finger
                            boneDirections = np.array([0])
                            for boneIndex in range(0,4): # for each bone in each finger
                                bone = finger.bone(boneIndex)
                                start = np.array(bone.prev_joint.to_float_array()).reshape((1,3))
                                end = np.array(bone.next_joint.to_float_array()).reshape((1,3))
                                boneVector = end-start
                                try:
                                    boneDirections = np.concatenate((boneDirections,boneVector),axis=0)
                                except ValueError:
                                    boneDirections = boneVector
                            fingerJointAngles = np.array([])
                            for joint in range(len(boneDirections)-1): # for each joint in each finger
                                try:
                                    angle = self.getAngle(boneDirections[joint,:],boneDirections[joint+1,:]) # get a joint angle
                                except ValueError:
                                    print("Value Error boneDirections: {}".format(boneDirections))
                                except IndexError:
                                    print("Index Error boneDirections: {}".format(boneDirections))

                                try: # append it to joint angle record
                                    fingerJointAngles = np.concatenate((fingerJointAngles,angle),axis=1)
                                except ValueError:
                                    fingerJointAngles = angle
                            try: # for each finger, append joint data to hand angles variable
                                handAngles = np.concatenate((handAngles,fingerJointAngles),axis=0)
                            except ValueError:
                                handAngles = fingerJointAngles

                        outputData = [int(time.time())*1000]
                        outputData = outputData + list(handAngles.flatten())
                        with open(self.filename, "a") as outputFile:
                            writer = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(outputData)

                        # write to file!

def main():
    # Create a sample listener and controller
    print("PID: {}".format(os.getpid()))
    lock = threading.Lock()
    q = Queue.LifoQueue(maxsize=1)
    listener = SampleListener(lock,q)
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        while True:
            time.sleep(1)
            try:
                data = q.get(block=True)
                print("Hand data:")
                print(data)
            except QueueEmpty:
                pass

    except KeyboardInterrupt:
        print("time to go to sleep!")
    finally:
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
