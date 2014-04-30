import os
from glob import glob
import PIL
import numpy as np
from sklearn import svm
from pandas import DataFrame

dataDir = '/Users/aden/Dropbox/School/STA_298/Data/'
inDir = dataDir+'train_med/'  # train_med: 1000 cats, 1000 dogs
# outDir = dataDir+'results/'
# outputFile = dataDir+'predictions.csv'
files = glob(inDir + '*.jpg')
newRes = (500, 350)  # Super arbitrary
holdout = int(len(files)*.05)

def importImage(f, newRes):
    """
    Takes in a file name pointing to a raw image, rescales it to a global standard,
    then converts the pixels into a numpy RGB array of dimension (height, width, 3).
    Returns a flattened row vector of dimension height*width*3 x 1.
    """
    raw_img = PIL.Image.open(f).convert(mode='RGB')
    rescaled = raw_img.resize(newRes)
    imgMatrix = np.asarray(rescaled)
    imgVec = imgMatrix.flatten('F')  # Turn pixel matrix into column-major vector.
    return imgVec

def make_pixel_matrix(images, newRes, idxList):
    idx = [item for sublist in idxList for item in sublist]
    images = [images[i] for i in idx]
    mat = np.zeros((len(images), 3*newRes[0]*newRes[1]))  # Big collection of pixels
    trueVec = []  # List of true labels
    idVec = []
    i = 0
    for f in images:
        true, id = os.path.basename(f)[0: -4].split('_')  # Remove ".jpg" from the end, get true label and ID
        true = true.capitalize()
        vec = importImage(f, newRes)
        mat[i, :] = vec
        trueVec.append(true)
        idVec.append(id)
        i += 1
    return mat, trueVec, idVec

trainIdx = [range(holdout, len(files)-holdout)]
trainMat, trainTrue, idVec = make_pixel_matrix(files, newRes, trainIdx)

testIdx = [range(0, holdout), range(len(files)-holdout, len(files))]
testMat, testTrue, idVec = make_pixel_matrix(files, newRes, testIdx)

svmFitLinear = svm.SVC(kernel='linear').fit(trainMat, trainTrue)
svmFitRBF = svm.SVC(kernel='rbf').fit(trainMat, trainTrue)
svmFit3 = svm.SVC(kernel='poly', degree=3).fit(trainMat, trainTrue)

predictionsLinear = svmFitLinear.predict(testMat)
predictionsRBF = svmFitRBF.predict(testMat)
predictions3 = svmFit3.predict(testMat)

predictDF = DataFrame({"Linear_Predictions": predictionsLinear, \
                       "RadialBasisFunction_Predictions": predictionsRBF, \
                       "Cubic_Predictions": predictions3, \
                       "Truth": testTrue, \
                       "ID": idVec})
predictDF.to_csv("SVM_predictions.csv")