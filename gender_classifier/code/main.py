import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import classifier as cf
import facedetection as f

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector

"""
given a jpg image, vectorize the grayscale pixels to 
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""
def vectorize(filename):
    size = WIDTH, HEIGHT # (width, height)
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array


if __name__ == '__main__':
    """
    load training data
    """
    # create a list for filenames of smiles pictures
    malefiles = []
    with open('male.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            malefiles += rec

    # create a list for filenames of neutral pictures
    femalefiles = []
    with open('female.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            femalefiles += rec

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = np.zeros((len(malefiles) + len(femalefiles), dim))
    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load smile data
    PATH = "../data/proj/new_boys/"
    for idx, filename in enumerate(malefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(1)

    # load neutral data    
    PATH = "../data/proj/new_girls/"
    offset = idx + 1
    for idx, filename in enumerate(femalefiles):
        phi[idx + offset] = vectorize(PATH + filename)
        labels.append(0)

    """
    training the data with logistic regression
    """
    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    