import numpy as np

predictions = np.genfromtxt(outputFile, delimiter=',', dtype=str)
n = len(predictions)-1
successes = 0

for row in predictions:
    if row[1] == row[2]:
        successes += 1

accuracy = float(successes) / float(n)
print("The prediction accuracy is: " + str(accuracy))
