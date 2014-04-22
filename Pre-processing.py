import os
from glob import glob

dataDir = 'data/'
inDir = dataDir+'train_small/'
outDir = dataDir+'results/'
overfeat = '/Users/aden/Dropbox/School/STA_298/overfeat/src/overfeat'
overfeatData = '/Users/aden/Dropbox/School/STA_298/overfeat/data/default/'
outputFile = dataDir+'predictions.csv'

cats = dataDir + "cats.txt"
dogs = dataDir + "dogs.txt"
files = glob(inDir + '*.jpg')

cf = open(dataDir+"cats.txt")
df = open(dataDir+"dogs.txt")

cats = cf.read()
cats = cats.split("\n")
cats = {x: x for x in cats}

dogs = df.read()
dogs = dogs.split("\n")
dogs = {x: x for x in dogs}