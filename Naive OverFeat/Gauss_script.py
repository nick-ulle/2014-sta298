import os, sys
from random import choice
import csv

import platform
if platform.system()=='Darwin':
    dataDir = '/Users/aden/Dropbox/School/STA_298/Github/Data/'
    overfeat = '/Users/aden/Dropbox/School/STA_298/OverFeat/src/overfeat'
    overfeatData = '/Users/aden/Dropbox/School/STA_298/OverFeat/data/default/'
else:
    dataDir = '/home/caden11/Github/Data/'
    overfeat = '/home/caden11/OverFeat/src/overfeat'
    overfeatData = '/home/caden11/OverFeat/data/default/'

inDir = dataDir+'train_small/'
outDir = dataDir+'results/'

basenum = sys.argv[1]
catImage = inDir+"cat_"+str(basenum)+".jpg"
dogImage = inDir+"dog_"+str(basenum)+".jpg"

cf = open(dataDir+"cats.txt")
cats = cf.read()
cats = cats.split("\n")
cats = {x: x for x in cats}
cf.close()

df = open(dataDir+"dogs.txt")
dogs = df.read()
dogs = dogs.split("\n")
dogs = {x: x for x in dogs}
df.close()

preds = open(outDir+'predictions_'+str(basenum)+'.csv', 'wb')
writer = csv.writer(preds)
# writer.writerow(['Base Number', 'Guess', 'True', 'IsRandomGuess'])

catBase = os.path.basename(catImage)
catCmd = '{} -d {} -l {} > {}{}.txt'.format(overfeat, overfeatData, catImage, outDir, catBase)
os.system(catCmd)

dogBase = os.path.basename(dogImage)
dogCmd = '{} -d {} -l {} > {}{}.txt'.format(overfeat, overfeatData, dogImage, outDir, dogBase)
os.system(dogCmd)


#Guessing for the cat image.
feats = open(outDir+catBase+".txt").readlines()
features = [x.split(' 0.')[0] for x in feats]  # Get just the features--not weights.

guess = False
randomGuess = 0

for feature in features:
    if feature in cats:
        guess = 'cat'
        break
    elif feature in dogs:
        guess = 'dog'
        break
if not guess:  # Features were not in the dictionary of (dog|cat)-related words
    guess = choice(['cat', 'dog'])  # Make a random guess
    randomGuess = 1
writer.writerow([basenum, guess, 'cat', randomGuess])

# Again, but on the dogs this time.
feats = open(outDir+dogBase+".txt").readlines()
features = [x.split(' 0.')[0] for x in feats]  # Get just the features--not weights.

guess = False
randomGuess = 0
for feature in features:
    if feature in cats:
        guess = 'cat'
        break
    elif feature in dogs:
        guess = 'dog'
        break
if not guess:  # Features were not in the dictionary of (dog|cat)-related words
    guess = choice(['cat', 'dog'])  # Make a random guess
    randomGuess = 1
writer.writerow([basenum, guess, 'dog', randomGuess])

#Finish writing cat and dog predictions and truth. Close file, end array script.
preds.close()
sys.exit()
