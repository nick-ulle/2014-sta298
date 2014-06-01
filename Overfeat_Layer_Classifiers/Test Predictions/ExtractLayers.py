import os, sys, platform, Image
import numpy as np

if platform.system() == 'Darwin':
    dataDir = '/Users/aden/Dropbox/School/STA_298/data/'
    overfeat = '/Users/aden/overfeat/src/overfeat'
    overfeatData = '/Users/aden/overfeat/data/default/'
    basenum = 4000
else:
    dataDir = '/home/caden11/Github/Data/'
    overfeat = '/home/caden11/OverFeat/src/overfeat'
    overfeatData = '/home/caden11/OverFeat/data/default/'
    basenum = sys.argv[1]

inDir = dataDir+'test/'
outDir = dataDir+'results/'
image = inDir+str(basenum)+".jpg"

def extract_features(image):
    basename = os.path.basename(image)
    cmd = '{} -d {} -l {} -f > {}{}.txt'.format(overfeat, overfeatData, image, outDir, basename)
    os.system(cmd)

    with open(outDir+basename+'.txt', 'r') as layerFile:
      firstline = layerFile.readline()
    if firstline == '$ Invalid argument 2: conv2Dmv : Input image is smaller than kernel\n':  #Make image bigger
        img = Image.open(image)
        img_resized = img.resize((2*img.size[0], 2*img.size[1]))
        img_resized.save(image)
        vec = extract_features(image)
        return vec

    dimList = firstline[0:-1]  # Get dim info from first line, kill new-line char.
    dims = [int(x) for x in dimList.split()]

    layer = np.loadtxt(outDir+basename+'.txt', skiprows=1)
    layer = layer.reshape(dims)
    return np.amax(np.amax(layer, 1), 1)  # Get max in columns, then max in rows, return 4096-vector

finalLayer = extract_features(image)
np.savetxt(outDir+'layer_'+str(basenum)+'.csv', finalLayer, delimiter=',')
deleteCmd = 'rm -f {}'.format(outDir+os.path.basename(image)+'.txt')
os.system(deleteCmd)
sys.exit()
