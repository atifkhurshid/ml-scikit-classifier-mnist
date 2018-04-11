from __future__ import print_function

import numpy as np
from struct import unpack

class Data():

    def __init__(self):
        return

    def load_train_data(self):
        train_img = 'train-images.idx3-ubyte'
        train_lbl = 'train-labels.idx1-ubyte'
        return self.__get_labeled_data(train_img, train_lbl)

    def load_test_data(self):
        test_img = 't10k-images.idx3-ubyte'
        test_lbl = 't10k-labels.idx1-ubyte'
        return self.__get_labeled_data(test_img, test_lbl)

    def __get_labeled_data(self, imagefile, labelfile):
        # Source: Assignment 2 @ http://www.cse.cuhk.edu.hk/~cslui/csci3320.html
        """
        Read input-vector (image) and target class (label, 0-9) and return it as list of tuples.
        Adapted from: https://martin-thoma.com/classify-mnist-with-pybrain/
        """
        # Open the images with gzip in read binary mode
        images = open(imagefile, 'rb')
        labels = open(labelfile, 'rb')

        # Read the binary data
        # We have to get big endian unsigned int. So we need '>I'

        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]

        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = labels.read(4)
        N = unpack('>I', N)[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')

        # Get the data
        X = np.zeros((N, rows * cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            for id in range(rows * cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                X[i][id] = tmp_pixel
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]
        return (X, y)
