#!/usr/bin/python3

#
#
# [[2], [3], [4], [5], [6], [7], [8], [2, 2], [4, 4], [6, 6], [8, 8], [8, 2, 2], [8, 4, 4], [8, 6, 6]]
#  [0.5530086, 0.63419294, 0.64565426, 0.64374405, 0.62750715, 0.63514805, 0.63037246, 0.5530086, 0.5530086, 0.6150907, 0.62750715, 0.5530086, 0.61604583, 0.63801336]
#
# [[4, 4, 4, 4], [8, 8, 4, 4], [13, 4], [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13]]
# [0.64374405, 0.65234, 0.63228273, 0.6618911]


import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=18000, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def amalgamateData(files=["outputLong01.csv"]):
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
# emg0,emg1,emg2,emg3,emg4,emg5,emg6,emg7,
# Thumb_Proximal,Thumb_Intermediate,Thumb_Distal,
# Index_Proximal,Index_Intermediate,Index_Distal,
# Middle_Proximal,Middle_Intermediate,Middle_Distal,
# Ring_Proximal,Ring_Intermediate,Ring_Distal,
# Pinky_Proximal,Pinky_Intermediate,Pinky_Distal
def processData(x,y):
    y = y[:,1:] # to eliminate nan
    y = y[:,3:6] # look only at Index_Proximal Index_Intermediate,Index_Distal (end not inclusive)
    y = y.mean(axis=1)
    length = y.size
    print("Using {} samples".format(length))
    A = y.reshape(length,1)
    B = []
    meanData = y.mean()
    print("mean angle: {}".format(y.mean()))
    print("median angle: {}".format(np.median(y)))
    print("mode angle: {}".format(stats.mode(y)))
    sns.distplot(y)

    for a in A:
        if a > meanData:
            B.append(1)
        else:
            B.append(0)

    print("mean class: {}".format(np.array(B).mean()))
    return x,B

def train_input_fn(features, labels, batch_size): #"""An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(20000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

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

    #for key in outputDataLabels:
        #numeric_feature_column = tf.feature_column.numeric_column(key)
        #bucketized_feature_column = tf.feature_column.bucketized_column( source_column = numeric_feature_column, boundaries = [20])
        #my_feature_columns.append(bucketized_feature_column)
    livSet = []
    for i in range(13):
        livSet.append(13)

    #structures = [[4,4,4,4],[8,8,4,4], [13,4],livSet]
    accuracies = []
    dataX,dataY = amalgamateData() # maybe search through .csv files in trainingData folder
    dataX,dataY = processData(dataX,dataY)

    proportionTrain = 17/18
    print("proportionTrain type: {}".format(type(proportionTrain)))
    length,width = dataX.shape
    trainIndex = int(length*proportionTrain)
    print("trainIndex type: {}".format(type(trainIndex)))
    train_x = dataX[:trainIndex,:]
    train_y = dataY[:trainIndex]
    test_x = dataX[trainIndex:,:]
    test_y = dataY[trainIndex:]
    train_x = pd.DataFrame(train_x, columns = inputDataLabels)
    train_y = pd.DataFrame(train_y, columns = outputDataLabels)
    test_x = pd.DataFrame(test_x, columns = inputDataLabels)
    test_y = pd.DataFrame(test_y, columns = outputDataLabels)
    print("shape: {}".format(train_x.shape))
    print("shape: {}".format(train_y.shape))
    print("shape: {}".format(test_x.shape))
    print("shape: {}".format(test_y.shape))

    #for structure in structures:
    structure = [8,8,8,8]
    classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=structure,
    n_classes=2,
    #model_dir='neuralNetdata'
    )

    a=classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),steps=args.train_steps)

    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y,
                                                args.batch_size))
    val = eval_result["accuracy"]
    print('\nTest set accuracy: {}\n'.format(val))

    eval_result2 = classifier.evaluate(
        input_fn=lambda:eval_input_fn(train_x, train_y,
                                                args.batch_size))
    val = eval_result2["accuracy"]
    print('\ntrain set accuracy: {}\n'.format(val))




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
