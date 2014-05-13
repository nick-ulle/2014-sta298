#from os.path import basename
import numpy as np
from platform import system
from glob import glob
from sys import exit
from pandas import DataFrame
from pickle import dump, load

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

if system()=='Darwin':  # On my mac
    dataDir = '/Users/aden/Dropbox/School/STA_298/data/'
else:  # On any other machine (Stats servers, cluster comp...)
    dataDir = '/home/cbaden/Github/Data/'

trainDir = dataDir+'results/layers/'
outDir = dataDir+'results/'
layers = glob(trainDir+'layer_*.csv')  # Get all layer filenames in layer dir.

alldata = ClassificationDataSet(4096, nb_classes=2, class_labels=['Cat', 'Dog'])

for layer in layers:  # Load layer vectors into PyBrain CDS format.
    cat, dog = np.split(np.genfromtxt(layer, delimiter=',', dtype=None, names=None, skip_header=True), 2, axis=1)
    cat = [float(x) for sublist in cat.tolist() for x in sublist]
    dog = [float(x) for sublist in dog.tolist() for x in sublist]
    alldata.addSample(cat, [0])
    alldata.addSample(dog, [1])

test, train = alldata.splitWithProportion(0.20)  # Split into train and test set.
test._convertToOneOfMany()
train._convertToOneOfMany()

nn = buildNetwork(train.indim, 5, train.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(nn, dataset=train, momentum=0.1, verbose=False, weightdecay=0.01)
trainer.trainUntilConvergence(maxEpochs=300, continueEpochs=10)
trainResult = percentError(trainer.testOnClassData(), train['class'])
testResult = percentError(trainer.testOnClassData(dataset=test), test['class'])
print "Epoch " + str(trainer.totalepochs)
print "Training Error: %2.2f%%" % trainResult, "Testing Error: %2.2f%%" % testResult

guess = trainer.testOnClassData(dataset=test)
truth = [int(x) for x in test['class']]

predictDF = DataFrame({"Guess": guess, "Truth": truth})
predictDF.to_csv(outDir+"NN_predictions.csv")

f = open('NeuralNet', 'w')
dump(nn, f)
f.close()

#f = open('NeuralNet','r')
#nn = load(f)
# idVec = [2*[os.path.basename(l)[0: -4].split('_')[1]] for l in layers]
# idVec = [x for sl in idVec for x in sl]
exit()