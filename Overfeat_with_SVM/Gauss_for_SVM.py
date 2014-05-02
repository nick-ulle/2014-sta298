import os, sys, platform, csv
import numpy as np

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


def extract_features(image):
    basename = os.path.basename(image)
    cmd = '{} -d {} -l {} -f > {}{}.txt'.format(overfeat, overfeatData, image, outDir, basename)
    os.system(cmd)
    layer = np.loadtxt(outDir+basename+'.txt', skiprows=1)

    with open(outDir+basename+'.txt', 'r') as layerFile:
      dimList = layerFile.readline()[0:-1]  # Get dim info from first line, kill new-line char.
    dims = [int(x) for x in dimList.split()]
    layer = layer.reshape(dims)
    return np.amax(np.amax(layer, 1), 1)  # Get max in columns, then max in rows, return 4096-vector

catLayer = extract_features(catImage)
dogLayer = extract_features(dogImage)

layer = open(outDir+'layer_'+str(basenum)+'.csv', 'wb')
writer = csv.writer(layer)
writer.writerow(['Cat', 'Dog'])

for idx in xrange(4096):
    writer.writerow([catLayer[idx], dogLayer[idx]])
layer.close()

deleteCat = 'rm -f {}'.format(outDir+os.path.basename(catImage)+'.txt')
deleteDog = 'rm -f {}'.format(outDir+os.path.basename(dogImage)+'.txt')
os.system(deleteCat)
os.system(deleteDog)
sys.exit()
