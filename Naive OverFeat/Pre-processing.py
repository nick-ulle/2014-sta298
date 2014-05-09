import os
from glob import glob

dataDir = '/Users/aden/Dropbox/School/STA_298/data/'
inDir = dataDir+'train/'
outDir = dataDir+'results/'
overfeat = '/Users/aden/overfeat/src/overfeat'
overfeatData = '/Users/aden/overfeat/data/default/'
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