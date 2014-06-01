import os
import numpy as np
from glob import glob
from pandas import DataFrame
from platform import system
from sklearn import svm
from sys import exit

if system()=='Darwin':
    dataDir = '/Users/aden/Dropbox/School/STA_298/data/'
else:
    dataDir = '/home/cbaden/Github/Data/'

trainDir = dataDir+'results/trainLayers/'
outDir = dataDir+'results/'
trainingLayers = glob(trainDir+'layer_*10.csv')


def make_layer_matrix(layers, idxList):
    idx = [item for sublist in idxList for item in sublist]
    layers = [layers[i] for i in idx]

    mat = np.zeros((2*len(layers), 4096))  # Big collection of pixels
    trueVec = len(layers)*['Cat', 'Dog']  # List of true labels
    idVec = [2*[os.path.basename(l)[0: -4].split('_')[1]] for l in layers]
    idVec = [x for sl in idVec for x in sl]

    i = 0
    for layer in layers:
        l = np.genfromtxt(layer, delimiter=',', dtype=None, names=None, skip_header=True)
        mat[i, ] = l[:, 0]  # Cat first
        mat[i+1, ] = l[:, 1]  # Dog next
        i += 2
    return mat, trueVec, idVec

trainIdx = [range(len(trainingLayers))]
trainMat, trainTrue, trainID = make_layer_matrix(trainingLayers, trainIdx)

# Getting a true matrix.
testingLayers = glob(trainDir+'/test/'+'layer_*.csv')
testingLayers.sort()
testMat = np.zeros((len(testingLayers), 4096))  # Big collection of pixels
idVec = [int(os.path.basename(l).split('.')[0].split('_')[1]) for l in testingLayers]

i = 0
for layer in testingLayers:
    testMat[i, ] = np.genfromtxt(layer, delimiter=',', dtype=None, names=None, skip_header=False)
    i += 1

svmFitRBF = svm.SVC(kernel='rbf', cache_size=4096).fit(trainMat, trainTrue)
svmFit3 = svm.SVC(kernel='poly', degree=3, cache_size=4096).fit(trainMat, trainTrue)
predictionsRBF = svmFitRBF.predict(testMat)
predictions3 = svmFit3.predict(testMat)

predictDF = DataFrame({"RadialBasisFunction_Predictions": predictionsRBF,
                       "Cubic_Predictions": predictions3,
                       "ID": idVec})
predictDF.to_csv(outDir+"SVM_predictions.csv")
exit()
