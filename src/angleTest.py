#!/usr/bin/python
import numpy as np


a = np.array([0,1,1])
b = np.array([0,1,0])
aDotb = np.dot(-a,b)
print("a*b :{}".format(aDotb))
magProd = (np.linalg.norm(a)*np.linalg.norm(b))
print("magProd :{}".format(magProd))
g = aDotb/magProd # G
print("g:{}".format(g))
angle = np.arccos(g) # getAngles
print("angle(rads):{}".format(angle))
print("angle(degrees):{}".format(angle*(180/3.14)))
