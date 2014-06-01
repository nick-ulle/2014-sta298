import os
import numpy as np
from glob import glob
from pandas import DataFrame
from platform import system
from sklearn import svm

if system() == 'Darwin':
    dataDir = '/Users/aden/Dropbox/School/STA_298/data/'
else:
    dataDir = '/home/cbaden/Github/Data/'

trainDir = dataDir + 'results/train_layers/'
testDir = dataDir + 'results/test_layers/'
outDir = dataDir+'results/'
trainLayers = glob(trainDir+'layer_100*.csv')
testLayers = glob(testDir+'layer_100*.csv')

def make_training_matrix(layers):
    mat = np.zeros((2*len(layers), 4096))  # Big collection of features
    trueVec = len(layers)*['cat', 'dog']  # List of true labels
    idVec = [2*[os.path.basename(l)[0: -4].split('_')[1]] for l in layers]
    idVec = [x for sl in idVec for x in sl]

    i = 0
    for layer in layers:
        l = np.genfromtxt(layer, delimiter=',', dtype=None, names=None, skip_header=True)
        mat[i, ] = l[:, 0]  # Cat first
        mat[i+1, ] = l[:, 1]  # Dog next
        i += 2
    return mat, trueVec, idVec


def make_testing_matrix(layers):
    mat = np.zeros((len(layers), 4096))  # Big collection of features
    idVec = [os.path.basename(l)[0: -4].split('_')[1] for l in layers]

    i = 0
    for layer in layers:
        l = np.genfromtxt(layer, delimiter=',', dtype=None, names=None, skip_header=False)
        mat[i, ] = l
        i += 1
    return mat, idVec

trainMat, trainTrue, trainID = make_training_matrix(trainLayers)
testMat, testID = make_testing_matrix(testLayers)

svmFitLinear = svm.SVC(kernel='linear', cache_size=4096, probability=True).fit(trainMat, trainTrue)
svmFitRBF = svm.SVC(kernel='rbf', cache_size=4096, probability=True).fit(trainMat, trainTrue)
svmFitCubic = svm.SVC(kernel='poly', degree=3, cache_size=4096, probability=True).fit(trainMat, trainTrue)

predictionsLinear = svmFitLinear.predict(testMat)
predictionsRBF = svmFitRBF.predict(testMat)
predictionsCubic = svmFitCubic.predict(testMat)

def calcProbs(preds, probs):
    singleProb = np.zeros(len(probs))
    i = 0
    for i in xrange(len(singleProb)):
        if preds[i] == 'cat':
            singleProb[i] = probs[i, 0]
        else:
            singleProb[i] = probs[i, 1]
    return singleProb

probsLinearBoth = svmFitLinear.predict_proba(testMat)
probsRBFBoth = svmFitRBF.predict_proba(testMat)
probsCubicBoth = svmFitCubic.predict_proba(testMat)
probsLinear = calcProbs(predictionsLinear, probsLinearBoth)
probsRBF = calcProbs(predictionsRBF, probsRBFBoth)
probsCubic = calcProbs(predictionsCubic, probsCubicBoth)

predictLinear = DataFrame({"imageID": testID, "animal": predictionsLinear, "probability": probsLinear})
predictRBF = DataFrame({"imageID": testID, "animal": predictionsRBF, "probability": probsRBF})
predictCubic = DataFrame({"imageID": testID, "animal": predictionsCubic, "probability": probsCubic})
predictLinear.to_csv(outDir+"Linear.csv", cols=['imageID', 'animal', 'probability'])
predictRBF.to_csv(outDir+"RBF.csv", cols=['imageID', 'animal', 'probability'])
predictCubic.to_csv(outDir+"Cubic.csv", cols=['imageID', 'animal', 'probability'])