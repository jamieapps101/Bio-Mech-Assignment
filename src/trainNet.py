#!/usr/bin/python3


import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=18000, type=int, help='batch size')
parser.add_argument('--train_steps', default=5000, type=int,
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
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
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

    structures = [[2],[3],[4],[5],[6],[7],[8],[2,2],[4,4],[6,6],[8,8],[8,2,2],[8,4,4],[8,6,6]]
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

    for structure in structures:
        classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=structure,
        n_classes=2,
        #model_dir='neuralNetdata'
        )

        a=classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),steps=args.train_steps)
        print("----------------------------------------------------------")
        print(dir(a))
        print("----------------------------------------------------------")
        eval_result = classifier.evaluate(
            input_fn=lambda:eval_input_fn(test_x, test_y,
                                                    args.batch_size))
        print(type(eval_result))
        print(eval_result)
        val = eval_result["accuracy"]
        print('\nTest set accuracy: {}\n'.format(val))
        accuracies.append(val)
    #plt.show()
    print(structures)
    print(accuracies)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
