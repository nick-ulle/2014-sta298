import os, platform, Image, sys
import numpy as np
from glob import glob
# from shutil import copy

if platform.system() == 'Darwin':
    dataDir = '/Users/aden/Dropbox/School/STA_298/data/'
    overfeat = '/Users/aden/overfeat/src/overfeat'
    overfeatData = '/Users/aden/overfeat/data/default/'
    idx = 0
else:
    dataDir = '/home/caden11/Github/Data/'
    overfeat = '/home/caden11/OverFeat/src/overfeat'
    overfeatData = '/home/caden11/OverFeat/data/default/'
    idx = sys.argv[1]

errorDir = dataDir+'results/Problems/'
outDir = dataDir+'results/'

missingFiles = glob(dataDir+'results/Problems/*.jpg')

# missingFileNum = np.loadtxt(dataDir+'results/missing.txt', dtype=int).tolist()
# missingFiles = [dataDir+'results/Problems/'+str(x)+'.jpg' for x in missingFileNum]
image = missingFiles[idx]
# for f in missingFiles:
#     copy(dataDir + 'test/' + os.path.basename(f), errorDir)


def extract_features(image):
    basename = os.path.basename(image)
    tries = 0
    print "Starting Image " + str(os.path.basename(image))
    cmd = '{} -d {} -l {} -f > {}{}.txt'.format(overfeat, overfeatData, image, outDir, basename)
    os.system("pkill -f overfeat")  # Kill old overfeat instances.
    os.system(cmd)
    with open(outDir+basename+'.txt', 'r') as layerFile:
            firstline = layerFile.readline()
    while firstline == '$ Invalid argument 2: conv2Dmv : Input image is smaller than kernel\n':
            if tries >= 5:
                print "Image " + str(os.path.basename(image)) + " resize failed 5 times. Exiting."
                break
            tries += 1
            print "Image " + str(os.path.basename(image)) + " too small. Resize attempt #" + str(tries)
            img = Image.open(image)
            img_resized = img.resize((2*img.size[0], 2*img.size[1]))
            img_resized.save(image)
            os.system(cmd)
            with open(outDir+basename+'.txt', 'r') as layerFile:
                firstline = layerFile.readline()

    dimList = firstline[0:-1]  # Get dim info from first line, kill new-line char.
    dims = [int(x) for x in dimList.split()]

    layer = np.loadtxt(outDir+basename+'.txt', skiprows=1)
    layer = layer.reshape(dims)
    return np.amax(np.amax(layer, 1), 1)  # Get max in columns, then max in rows, return 4096-vector

if platform.system() == 'Darwin':
    for image in missingFiles:
        finalLayer = extract_features(image)
        basenum = os.path.basename(image).split('.')[0]
        np.savetxt(outDir+'layer_'+str(basenum)+'.csv', finalLayer, delimiter=',')
        deleteTxt = 'rm -f {}'.format(outDir+os.path.basename(image)+'.txt')
        deleteImg = 'rm -f {}'.format(image)
        os.system(deleteTxt)
        os.system(deleteImg)
    sys.exit()
else:
    finalLayer = extract_features(image)
    basenum = os.path.basename(image).split('.')[0]
    np.savetxt(outDir+'layer_'+str(basenum)+'.csv', finalLayer, delimiter=',')
    deleteTxt = 'rm -f {}'.format(outDir+os.path.basename(image)+'.txt')
    deleteImg = 'rm -f {}'.format(image)
    os.system(deleteTxt)
    os.system(deleteImg)
    sys.exit()
