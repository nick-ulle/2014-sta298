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
    dataDir = '/home/caden11/Github/Data/'

trainDir = dataDir+'results/layers/'
outDir = dataDir+'results/'
layers = glob(trainDir+'layer_1*.csv')


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

holdout = int(len(layers)*.05)
trainIdx = [range(holdout, len(layers)-holdout)]
trainMat, trainTrue, trainID = make_layer_matrix(layers, trainIdx)

testIdx = [range(0, holdout), range(len(layers)-holdout, len(layers))]
testMat, testTrue, testID = make_layer_matrix(layers, testIdx)

svmFitLinear = svm.SVC(kernel='linear', cache_size=4096).fit(trainMat, trainTrue)
svmFitRBF = svm.SVC(kernel='rbf', cache_size=4096).fit(trainMat, trainTrue)
svmFit3 = svm.SVC(kernel='poly', degree=3, cache_size=4096).fit(trainMat, trainTrue)
svmFitSigmoid = svm.SVC(kernel='sigmoid', cache_size=4096).fit(trainMat, trainTrue)
predictionsLinear = svmFitLinear.predict(testMat)
predictionsRBF = svmFitRBF.predict(testMat)
predictions3 = svmFit3.predict(testMat)
predictionsSigmoid = svmFitSigmoid.predict(testMat)

predictDF = DataFrame({"Linear_Predictions": predictionsLinear,
                       "RadialBasisFunction_Predictions": predictionsRBF,
                       "Cubic_Predictions": predictions3,
                       "Sigmoid_Predictions": predictionsSigmoid,
                       "Truth": testTrue,
                       "ID": testID})
predictDF.to_csv(outDir+"SVM_predictions.csv")
exit()
