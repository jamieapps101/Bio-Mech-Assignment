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
from tensorflow import keras
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10000, type=int, help='batch size')
parser.add_argument('--train_steps', default=100, type=int,
                    help='number of training steps')

cp_callback = tf.keras.callbacks.ModelCheckpoint("kerasModels/cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(1)

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['acc'],
                      '--', label=name.title()+' Val')

    plt.xlabel('Epochs')
    # plt.ylabel(key.replace('_',' ').title())
    plt.ylabel("acc")
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.grid(which="both")

    plt.figure(2)
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                     '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
               label=name.title()+' Train')
        # plt.plot(history.epoch, , color='g',
        #     label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.grid(which="both")



def amalgamateData():
    path = "/home/jamie/storageDrive/Drive/Uni/Third Year/ACS340 - Biomechatronics/Bio-Mech Assignment Windows/src/trainingData/combinedData.csv"
    fileData = pd.read_csv(path)
    totalData = fileData.to_numpy()
    np.random.shuffle(totalData)
    x = totalData[:,1:((5*8)+1)] #ignore time col, get 8*5 emg col
    print("x shape: {}".format(x.shape))
    y = totalData[:,-1] # get last col
    print("y shape: {}".format(y.shape))
    return x,y

def processData(x,y):
    #x = np.divide(x,1024)
    #y = y[:,1:] # to eliminate nan
    #y = y[:,3:6] # look only at Index_Proximal Index_Intermediate,Index_Distal (end not inclusive)
    #y = y.mean(axis=1)
    #length = y.size
    #print("Using {} samples".format(length))
    #A = y.reshape(length,1)
    #B = []
    #meanData = y.mean()
    #print("mean angle: {}".format(y.mean()))
    #print("median angle: {}".format(np.median(y)))
    #print("mode angle: {}".format(stats.mode(y)))
    #plt.figure(1)
    #sns.distplot(y) # for plotting initial data distribution
    #return x,y
    #yMax = np.max(y)
    #y = np.divide(y,yMax) # normalise
    bins = 10
    #y = np.multiply(y,bins-1)
    #y = np.round(y)
    #print("mean angle translated: {}".format(y.mean()))
    #dataBins = [[],[],[],[],[],[],[],[],[],[]]
    #tempBin = []
    #sizes = []
    #y = y.reshape(len(y),1)
    #data = np.append(x,y,axis=1)
    #print("mean angle translated: {}".format(data[:,8].mean()))
    #for a in range(bins):
        #print("a = {}".format(a))
#    for sample in data:
    #   dataBins[int(sample[8])].append(sample)
    #sum = 0
    #for bin in dataBins:
        #print(len(bin))
        #sum += len(bin)

    # print("Total: {}".format(sum))
    # newSamples = 20000
    # newData = [] # np.array([]).reshape(0,9)
    # #print("newData shape:{}".format(newData.shape))
    # sampleIndexes = np.random.randint(bins,size=newSamples)
    # print("mean index = {}".format(np.mean(sampleIndexes)))
    # for sampleIndex in sampleIndexes:
    #     sampleBin = dataBins[sampleIndex]
    #     if len(sampleBin) != 0:
    #         sampleNumber = np.random.randint(len(sampleBin))
    #         sample = sampleBin[sampleNumber]
    #         newData.append(sample)
    # newData = np.array(newData).astype(np.int64)
    # print("newData shape; {}".format(newData.shape))
    # print("newData distplot shape; {}".format(newData[:,8].shape))
    # print("newData example: \n{}".format(newData[:4,:]))
    # print("classification mean: {}".format(np.mean(newData[:,8])))
    # #plt.figure(2)
     #sns.distplot(newData[:,8]) # for fitting corrected distribution
    #plt.show()
    # x = newData[:,:8]
    # y = newData[:,8:]

    newY = []
    for sample in y:
        # if sample > (bins-1)/2:
        #     newY.append(1)
        # else:
        #     newY.append(0)
        newY.append(int(sample/3))
    y = np.array(newY)
    sns.distplot(y) # for fitting corrected distribution
    return x,y
    # exit()

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
        for b in range(5):
            inputDataLabels.append("emg"+str(a)+"fft"+str(b))

    outputDataLabels = ['aveAngle']
    for key in inputDataLabels:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    accuracies = []
    #path = "/home/jamie/storageDrive/Drive/Uni/Third Year/ACS340 - Biomechatronics/Bio-Mech Assignment Windows/src/trainingData/combinedData.csv"
    dataX,dataY = amalgamateData() # maybe search through .csv files in trainingData folder
    dataX,dataY = processData(dataX,dataY)
    print("dataY mean:{}".format(dataY.mean()))

    #plt.show()
    #exit()
    proportionTrain = 16/20
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
    plt.show()
    time.sleep(5)
#    structures = [[50],[100],[150],[200],[250],[300],[350],[400],[450],[500],
#                [50,50],[100,100],[200,200],[250,250],[300,300],[400,400]]
#                [0.685063, 0.68686265, 0.68326336, 0.71025795, 0.69526094, 0.7006599, 0.69286144, 0.7024595, 0.7132574, 0.7132574, 0.71385723, 0.72645473, 0.7306539, 0.7396521, 0.7324535, 0.73425317]


            #structure = [300]

        # classifier = tf.estimator.DNNClassifier(
        # feature_columns=my_feature_columns,
        # hidden_units=structure,
        # n_classes=2,
        # #model_dir='neuralNetdata'
        # )
        #
        # a=classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),steps=epoch)
        #
        # eval_result = classifier.evaluate(
        #     input_fn=lambda:eval_input_fn(test_x, test_y,
        #                                             args.batch_size))
        # val = eval_result["accuracy"]
        # accuracies.append(val)
        # print('\nTest set accuracy: {}\n'.format(val))
    model1 = keras.Sequential([
        #keras.layers.Flatten(input_shape=(1, 8)),
        keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.002), activation=tf.nn.relu,input_shape=(40,)),
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

    model2 = keras.Sequential([
        keras.layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.002), activation=tf.nn.relu,input_shape=(8,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.002),activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.002),activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.002),activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model3 = keras.Sequential([
        keras.layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,input_shape=(8,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.softmax),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model1.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy','binary_crossentropy'])

    model2.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy','binary_crossentropy'])

    model3.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy','binary_crossentropy'])
    tf.logging.set_verbosity(tf.logging.ERROR)
    #model1.summary()
    history1 = model1.fit(train_x, train_y, epochs=500,batch_size=20000,validation_data=(test_x, test_y),verbose=2, callbacks=[cp_callback])
    #history2 = model2.fit(train_x, train_y, epochs=600,batch_size=20000,validation_data=(test_x, test_y),verbose=2)
    #history3 = model3.fit(train_x, train_y, epochs=700,batch_size=20000,validation_data=(test_x, test_y),verbose=2)

    plot_history([('model1', history1)])#,('model2', history2),('model3', history3)])
    #test_loss, test_acc, bin_cross = model1.evaluate(test_x, test_y)
    #print("model1: {} ".format(test_acc))
    #test_loss, test_acc, bin_cross = model2.evaluate(test_x, test_y)
    #print("model2: {} ".format(test_acc))
    plt.show()






if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
