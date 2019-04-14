#!/usr/bin/python3


import argparse
import tensorflow as tf
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def amalgamateData(files=["jamieLeft1.csv", "jamieLeft2.csv", "livLeft1.csv", "livLeft2.csv"]):
    totalData = np.array([8])
    for file in files:
        if files.index(file) == 0:
            path = "trainingData/" + file
            fileData = pd.read_csv(path)
            totalData = fileData.to_numpy()
        else:
            path = "trainingData/" + file
            fileData = pd.read_csv(path)
            tempData = fileData.to_numpy()
            totalData = np.concatenate((totalData,tempData),axis=0)# ie going down

    np.random.shuffle(totalData)
    x = totalData[:,:8]
    y = totalData[:,8:]
    #y = totalData[:,9:] # bc lower joint of thumb (ie first entry) has value nan
    return x,y

def processData(x,y):
    y = y[:,1:] # to eliminate nan
    y = y.mean(axis=1)
    length = y.size
    y = y.reshape(length,1)
    return x,y

def train_input_fn(features, labels, batch_size): #"""An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset


def main(argv):

    args = parser.parse_args(argv[1:])
    #https://www.tensorflow.org/guide/feature_columns
    my_feature_columns = []
    inputDataLabels = []
    for a in range(8):
        inputDataLabels.append("emg"+str(a))

    #finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    #bone_names = ['Proximal', 'Intermediate', 'Distal']
    #outputDataLabels = []
    #for a in finger_names:
    #    for b in bone_names:
    #        outputDataLabels.append(a+"_"+b)
    outputDataLabels = ['aveAngle']


    for key in inputDataLabels:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    for key in outputDataLabels:
        #my_feature_columns.append(tf.feature_column.bucketized_column(key=key))
        numeric_feature_column = tf.feature_column.numeric_column(key)
        bucketized_feature_column = tf.feature_column.bucketized_column( source_column = numeric_feature_column, boundaries = [20])


    classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=2,
    model_dir='neuralNetdata')

    dataX,dataY = amalgamateData() # maybe search through .csv files in trainingData folder
    dataX,dataY = processData(dataX,dataY)

    proportionTrain = 5/6

    length,width = dataX.shape
    trainIndex = int(length*proportionTrain)
    train_x = dataX[:trainIndex,:]
    train_y = dataY[:trainIndex,:]
    test_x = dataX[trainIndex:,:]
    test_y = dataY[trainIndex:,:]


    classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),steps=args.train_steps)




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
