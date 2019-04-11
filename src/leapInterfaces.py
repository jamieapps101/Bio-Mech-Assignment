#!/usr/bin/python
import sys, thread, time

import threading

import sys
sys.path.insert(0, "../lib")
import Leap
import math
import numpy as np
import Queue


printState = True
def myPrint( string):
    if printState==True:
        print(string)

class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def __init__(self, lockVariable, q):
        super(SampleListener,self).__init__()
        self.internalLockVariable = lockVariable
        self.internalQueue = q

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

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        time = frame.timestamp
        if (len(frame.hands) == 0):
            #print("please place hand in frame")
            pass
        elif (len(frame.hands) > 1):
            print("There are too many hands in the frame")
        else:
            for hand in frame.hands: # should only have one entry by this point
                angles = None
                f = 0
                for finger in hand.fingers: # should be 5
                    # Get bones
                    directions = 5
                    r = range(1,4) # Metacarpals are hard
                    #r.reverse()
                    myPrint("Finger {}  /////////////////////////////////".format(f))
                    f = f +1
                    for b in r:
                        myPrint("Bone {} //////////////////////".format(b))
                        bone = finger.bone(b)
                        if type(directions) != type(np.array([4,5])):
                            #boneDirectionList.append()
                            #directions = np.array(bone.direction.to_float_array()).reshape((1,3))
                            start = np.array(bone.prev_joint.to_float_array()).reshape((1,3))
                            print("start: {}".format(start))
                            end = np.array(bone.next_joint.to_float_array()).reshape((1,3))
                            print("end: {}".format(end))
                            directions = end-start
                            myPrint("directions {}".format(directions))

                        else:
                            #temp = np.array(bone.direction.to_float_array()).reshape((1,3))
                            start = np.array(bone.prev_joint.to_float_array()).reshape((1,3))
                            print("start: {}".format(start))
                            end = np.array(bone.next_joint.to_float_array()).reshape((1,3))
                            temp = end-start
                            print("end: {}".format(end))
                            directions = np.concatenate([directions,temp],axis=0)
                            myPrint("directions {}".format(directions))
                    myPrint("Finished bones //////////////////////".format(b))
                    angleList = []
                    for d in range(len(directions)-1):
                        a = directions[d,:]
                        myPrint("a at index {}:{}".format(d,a))
                        b = directions[d+1,:]
                        myPrint("b at index {}:{}".format(d,b))
                        aDotb = np.dot(a,b)
                        myPrint("a*b :{}".format(aDotb))
                        magProd = (np.linalg.norm(a)*np.linalg.norm(b))
                        myPrint("magProd :{}".format(magProd))
                        g = aDotb/magProd # G
                        myPrint("g at index {}:{}".format(d,g))
                        angle = np.arccos(g)*(180.0/3.14) # getAngle

                        myPrint("angle at index {}:{}".format(d,angle))
                        angleList.append(angle)
                    if type(angles) != type(np.array([4,5])):
                        myPrint("angleList {}".format(angleList))
                        angles = np.array(angleList).reshape(1,2)
                    else:
                        myPrint("angleList {}".format(angleList))
                        temp = np.array(angleList).reshape(1,2)
                        angles = np.concatenate([angles,temp],axis=0)

            # we now have a list of angles for each joint in the hand
            myPrint("shape of final matrix: {}+++++++++++++++++++++++++".format(angles.shape))
            #print("all angles:")
            #print(angles)
            # global globalAngles
            # self.internalLockVariable.acquire()
            # try:
            #     print("leap got a lock!")
            #     globalAngles = angles
            # except:
            #     print("leap no lock")
            # finally:
            #     self.internalLockVariable.release()
            self.internalQueue.put([time,angles])


        #if not (frame.hands.is_empty and frame.gestures().is_empty):
            #print ("")

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"



def main():
    # Create a sample listener and controller
    lock = threading.Lock()
    q = Queue.LifoQueue(maxsize=1)
    listener = SampleListener(lock,q)
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        #sys.stdin.readline()
        global globalAngles
        internalAngles = None
        while True:
            #print("Hi from main thread!")
            try:
                data = q.get(block=False)
                timeStamp = data[0]
                print("timeStamp {}".format(timeStamp))
                internalAngles = data[1]
                fingers = ['thumb', 'fore','middle','ring','little']
                for f in range(len(fingers)):
                    print("{}:{}".format(fingers[f],internalAngles[f,:]))
                #print(internalAngles)
                print("")
                print("")
                time.sleep(5)
            except Queue.Empty:
                #print("Nothing to see here....")
                pass

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("time to go to sleep!")
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
