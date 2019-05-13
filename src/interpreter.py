#!/usr/bin/python3

#import threading
import os
import sys, time
import math
import numpy as np
#import Queue
sys.path.insert(0, "/home/jamie/Coding/projects/Bio-Mech Assignment/lib")

from myoConnectFunctions import *
#import lib
import serial
import argparse
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np

port = serial.Serial("/dev/ttyACM1",115200)

inputDataLabels = []
for a in range(8):
    inputDataLabels.append("emg"+str(a))

my_feature_columns=[]
for key in inputDataLabels:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(
feature_columns=my_feature_columns,
hidden_units=[10, 10],
n_classes=2,
model_dir='neuralNetdata'
)


def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)# Convert the inputs to a Dataset.
    assert batch_size is not None, "batch_size must not be None" # Batch the examples
    dataset = dataset.batch(batch_size)
    return dataset


def proc_emg(emg, moving, times=[]):
        try:
            emgDataList = np.array(list(emg))
            #print("emgDataList: {}".format(emgDataList))
            input_x = pd.DataFrame([emgDataList], columns = inputDataLabels)
            #print("input_x: {}".format(input_x))
            model1 = keras.Sequential([
                #keras.layers.Flatten(input_shape=(1, 8)),
                keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.002), activation=tf.nn.relu,input_shape=(8,)),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.002),activation=tf.nn.relu),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.002),activation=tf.nn.relu),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.002),activation=tf.nn.relu),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(2, activation=tf.nn.softmax)
            ])
            model.load_weights(checkpoint_path)
            pre
            for a in output:
                classID  = a['class_ids'][0]
                print(classID)



        except KeyboardInterrupt:
                exit()


m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
m.add_emg_handler(proc_emg)
print("Connecting to Myo Band")
m.connect()
print("Myo Connected")

pause = input("press enter when ready to begin")

print("Press Enter to quit...")
try:
    internalAngles = None
    while True:
        m.run(1)
except KeyboardInterrupt:
    print("time to go to sleep!")
