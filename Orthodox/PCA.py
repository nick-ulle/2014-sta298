import os
from glob import glob
import PIL
import numpy as np
from sklearn.decomposition import RandomizedPCA
from pandas import DataFrame
from ggplot import *

dataDir = '/Users/aden/Dropbox/School/STA_298/Data/'
inDir = dataDir+'train_med/'  # train_med: 1000 cats, 1000 dogs
# outDir = dataDir+'results/'
# outputFile = dataDir+'predictions.csv'
files = glob(inDir + '*.jpg')

newRes = (500, 350)  # Super arbitrary

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

bigMatrix = np.zeros((len(files), 3*newRes[0]*newRes[1]))  # Big collection of pixels
trueVec = []  # List of true labels
idVec = []
i = 0

for f in files:
    true, id = os.path.basename(f)[0: -4].split('_')  # Remove ".jpg" from the end, get true label and ID
    true = true.capitalize()
    vec = importImage(f, newRes)
    bigMatrix[i, :] = vec
    trueVec.append(true)
    idVec.append(id)
    i += 1

PCA = RandomizedPCA(n_components=2).fit_transform(bigMatrix)
pcaData = DataFrame({"PCA1": PCA[:, 0], "PCA2": PCA[:, 1], "label":trueVec, "ID": id})

PCAplot = ggplot(aes(x='PCA1', y='PCA2', colour='label'), data=pcaData) + geom_point() \
    + xlab('Principal Axis 1') + ylab('Principal Axis 2') + ggtitle('PCA stands for Pretty Crummy Algorithm, n=700')
ggsave('PCA_plot.pdf', plot = PCAplot, width = 10, height = 8)