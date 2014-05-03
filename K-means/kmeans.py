#!/usr/bin/env python2

import os

import cv2
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

SEED = 508
N_CHUNKS = 10**5

def main():
    pass

def load_chunks(chunk_size = 16):
    # Load the chunks from the chunks directory.
    file_list = os.listdir('chunks/')
    n = len(file_list)

    chunks = np.empty((n, 3 * chunk_size**2))
    for i, file in enumerate(file_list):
        image = cv2.imread('chunks/' + file)
        chunks[i, ] = np.ravel(image)

    return chunks

def make_chunks(n, chunk_size = 16):
    ''' Generate n chunks of the specified size from training data.

    Args:
        n           number of chunks
        chunk_size  size of each chunk on one side
    '''
    file_list = os.listdir('train/')

    print('Generating {} chunks of size {}...'.format(n, chunk_size))
    if not os.path.isdir('chunks'):
        os.mkdir('chunks')

    indices = np.random.randint(0, len(file_list), n)

    for i, id in enumerate(indices):
        image = cv2.imread('train/' + file_list[id])
        if type(image) == type(None):
            print(i)
            print(file_list[id])
        width, height, _ = image.shape

        x = np.random.randint(0, width - chunk_size)
        y = np.random.randint(0, height - chunk_size)

        chunk = image[x:(x + chunk_size), y:(y + chunk_size), ]
        cv2.imwrite('chunks/{:0>6}.png'.format(i), chunk)

if __name__ == '__main__':
    main()
