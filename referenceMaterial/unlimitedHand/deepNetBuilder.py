#!/usr/bin/python3

#useful: https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt

#openHandDataFile = input("Enter name of open hand data file")
#if openHandDataFile == "" or openHandDataFile == "\n":
openHandDataFile = 'open.csv'

openHandDataInput = np.genfromtxt(openHandDataFile, delimiter=',')
shape  = openHandDataInput.shape
openHandData = np.zeros((shape[0],shape[1]+1))
openHandData[:,:-1] = openHandDataInput # concat-ed with ones to indicate classified closed dataset

#closedHandDataFile = input("Enter name of closed hand data file")
#if closedHandDataFile == "" or closedHandDataFile == "\n":
closedHandDataFile = 'closed.csv'

closedHandDataInput = np.genfromtxt(closedHandDataFile, delimiter=',')
shape  = closedHandDataInput.shape
closedHandData = np.ones((shape[0],shape[1]+1))
closedHandData[:,:-1] = closedHandDataInput # concat-ed with ones to indicate classified closed dataset

totalData = np.concatenate((openHandData,closedHandData),axis=0) # stack datasets vertically
np.random.shuffle(totalData) # randomly shuffle data avoid biasing initial or final epochs
#print(totalData)
trainX = totalData[:,:-1]
trainY = totalData[:,-1]
print(trainX.shape)
print(trainY.shape)

#graph inputs/outpu
#X = tf.placeholder("int", shape=(1,8))
#Y = tf.placeholder("float")

#graph design
model = keras.Sequential([
    keras.layers.Dense(8, input_dim=8),
    keras.layers.Dense(8, activation='sigmoid'),
    #keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
loss='mean_squared_error',
metrics=['accuracy'])

model.summary()
#exit()
#exit()


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


#with tf.Graph().as_default():
#    with tf.Session() as sess:
        #sess.run(tf.global_variables_intializer())
#        K.set_session(sess)
        #model = load_model(model_path)
        #preds = model.predict(in_data)
history = model.fit(x=trainX, y=trainY, epochs=5)#, batch_size=1)
#plot_history(history)
