import random
import csv

preds = open(outputFile, 'wb')
writer = csv.writer(preds)
writer.writerow(['ID', 'Guess', 'True'])
i = 0

for f in files:
    #print f
    basename = os.path.basename(f)
    cmd = '{} -d {} -l {} > {}{}.txt'.format(overfeat, overfeatData, f, outDir, basename)
    os.system(cmd)
    i += 1

    truth = os.path.basename(f)[0:3]  # Will print dog if basename is dog_xxx, cat otherwise.

    feats = open(outDir+basename+".txt").readlines()
    features = [x.split(' 0.')[0] for x in feats]  # Get just the features--not weights.

    guess = False

    for feature in features:
        if feature in cats:
            guess = 'cat'
            break
        elif feature in dogs:
            guess = 'dog'
            break
    if not guess:  # Features were not in the dictionary of (dog|cat)-related words
        guess = random.choice(['cat', 'dog'])  # Make a random guess
    writer.writerow([i, guess, truth])
preds.close()