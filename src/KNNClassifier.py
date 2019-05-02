#!/usr/bin/python3

import argparse
#import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
#import seaborn as sns
import matplotlib.pyplot as plt



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
    y = y[:,3:6] # look only at Index_Proximal Index_Intermediate,Index_Distal (end not inclusive)
    y = y.mean(axis=1)
    length = y.size
    print("Using {} samples".format(length))
    A = y.reshape(length,1)
    B = []
    meanData = y.mean()
    #print("mean angle: {}".format(y.mean()))
    #print("median angle: {}".format(np.median(y)))
    #print("mode angle: {}".format(stats.mode(y)))
    #sns.distplot(y)

    for a in A:
        if a > meanData:
            B.append(1)
        else:
            B.append(0)

    #print("mean class: {}".format(np.array(B).mean()))
    #print("B: \n{}".format(np.array(B)))
    return x,np.array(B)

class KNNClassifier:
    def __init__(self):
        self.refDataX = None
        self.refDataY = None

    def setReferenceData(self,inputx,inputy):
        self.refDataX = inputx
        #print("self.refDataX: \n{}".format(self.refDataX))
        self.refDataY = inputy
        #print("self.refDataY: \n{}".format(self.refDataY))

    def classify(self,inputx):
        if(type(self.refDataX) == type(None) or type(self.refDataY) == type(None)):
            print("KNN Classify called with no reference data input")
            return None
        else:
            #print("inputx: \n{}".format(inputx))
            #print("self.refDataX: \n{}".format(self.refDataX))
            differences = np.subtract(self.refDataX,inputx)
            #print("differences: \n{}".format(differences))
            squared_differences = np.power(differences,2)
            #print("squared_differences: \n{}".format(squared_differences))
            summed_squared_differences = np.sum(squared_differences,axis=1).reshape(len(squared_differences),1)
            #print("summed_squared_differences: \n{}".format(summed_squared_differences))
            totalDataSet=np.append(summed_squared_differences,self.refDataY.reshape(len(self.refDataY),1),axis=1)
            #print("totalDataSet: \n{}".format(totalDataSet))
            totalDataSet = pd.DataFrame(data=totalDataSet,columns=["diff","class"])
            #exit()
            sortedData = totalDataSet.sort_values(by="diff",axis=0)
            sortedData = sortedData.values
            topSortedResult=sortedData[:1,:]# take top 100 closest points
            #print("topSortedResult: \n{}".format(topSortedResult))
            res = np.histogram(topSortedResult[:,1],bins=(0,0.5,1),density=True)
            res = res[0]
            #print("Res: \n{}".format(res))
            dist = res/np.sum(res)
            probClassA,probClassB = dist[0],dist[1]
            if probClassA > probClassB:
                return 0
            elif probClassA < probClassB:
                return 1

def main():
        dataX,dataY = amalgamateData() # maybe search through .csv files in trainingData folder
        dataX,dataY = processData(dataX,dataY)
        proportionTrain = 4/5
        #print("proportionTrain type: {}".format(type(proportionTrain)))
        length,width = dataX.shape
        trainIndex = int(length*proportionTrain)
        #print("trainIndex: \n{}".format(trainIndex))
        #print("trainIndex type: {}".format(type(trainIndex)))
        train_x = dataX[:trainIndex,:]
        #print("train_x: \n{}".format(train_x))
        train_y = dataY[:trainIndex]
        test_x = dataX[trainIndex:,:]
        test_y = dataY[trainIndex:]
        classifier = KNNClassifier()
        classifier.setReferenceData(train_x,train_y)
        # results
        AC0PC0 = 0
        AC0PC1 = 0
        AC1PC0 = 0
        AC1PC1 = 0
        for sample in zip(test_x,test_y):
            result = classifier.classify(sample[0])
            if sample[1] == 1:
                if result == 1:
                    AC1PC1 += 1
                else:
                    AC1PC0 += 1
            else:
                if result == 1:
                    AC0PC1 += 1
                else:
                    AC0PC0 += 1
        print("good:")
        print(AC0PC0+AC1PC1)
        print((AC0PC0+AC1PC1)/((AC0PC0+AC1PC1)+(AC0PC1+AC1PC0)))
        print("bad")
        print(AC0PC1+AC1PC0)
        print((AC0PC1+AC1PC0)/((AC0PC0+AC1PC1)+(AC0PC1+AC1PC0)))

if __name__ == '__main__':
    main()
