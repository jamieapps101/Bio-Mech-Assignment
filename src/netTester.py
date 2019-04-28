#!/usr/bin/python3


import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
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

def processData(x,y):
    y = y[:,1:] # to eliminate nan
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
    return x,B

def train_input_fn(features, labels, batch_size): #"""An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset


def main(argv):

    args = parser.parse_args(argv[1:])
    #https://www.tensorflow.org/guide/feature_columns
    my_feature_columns = []

    inputDataLabels = []
    for a in range(8):
        inputDataLabels.append("data"+str(a))

    for key in inputDataLabels:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    outputDataLabels = ["output"]


    classifier = tf.estimator.LinearClassifier(
    feature_columns=my_feature_columns,
    #hidden_units=[8,4],
    n_classes=2,
    #model_dir='neuralNetdata'
    )

    trainDataSize = 18000
    testDataSize = 1000
    secondDimension = 8
    print("lets make some data!")
    start = time.time()*1000
    numberOfWeights = 8
    maxWeight = 8
    #weights = np.random.uniform(size=(numberOfWeights,1))*maxWeight
    weights = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
    x =     np.random.uniform(size=(secondDimension,trainDataSize))
    xTest = np.random.uniform(size=(secondDimension,testDataSize))
    y =     np.round(np.divide(np.sum(np.multiply(x,weights),axis=0),np.sum(weights)))
    yTest = np.round(np.divide(np.sum(np.multiply(xTest,weights),axis=0),np.sum(weights)))
    end = time.time()*1000
    print("Done, that took {}ms".format(end-start))
    print()
    x = pd.DataFrame(x.transpose(), columns = inputDataLabels)
    y = pd.DataFrame(y.transpose(), columns = outputDataLabels)
    xTest = pd.DataFrame(xTest.transpose(), columns = inputDataLabels)
    yTest = pd.DataFrame(yTest.transpose(), columns = outputDataLabels)
    print("mean: {}".format(np.mean(y)))
    start = time.time()*1000
    a=classifier.train(input_fn=lambda:train_input_fn(x, y, 18000),steps=10000)
    print("Done, that took {}ms".format(end-start))

    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(xTest, yTest,10))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    #plt.show()



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
