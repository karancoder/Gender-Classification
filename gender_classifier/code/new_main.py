import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import classifier
import facedetection as f
import mouthdetection as m

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector

"""
given an area to be cropped, crop() returns a cropped image
"""
def crop(area): 
    crop = img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] #img[y: y + h, x: x + w]
    return crop

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
    gender_weights = 'weights.npy'
    smile_weights = 'weights_smile.npy'
    cf = classifier.Classifier(dim, gender_weights)
    cs = classifier.Classifier(dim, smile_weights)
    """
    open webcam and capture images
    """
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    print "press space to take picture; press ESC to exit"

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(40)
        key %= 256
        if key == 27: # exit on ESC
            break
        if key == 32: # press space to save images
            cv.SaveImage("webcam.jpg", cv.fromarray(frame))
            img = cv.LoadImage("webcam.jpg") # input image
            # gender
            face = f.findface(img)
            if face != 2: # did not return error
                faceimg = crop(face)
                cv.SaveImage("webcam-f.jpg", faceimg)
                # predict the captured emotion
                result = cf.predict(vectorize('webcam-f.jpg'))
                if result == 1:
                    print "you are a boy! ",
                else:
                    print "you are a girl! ",
            else:
                print "No Face Detected!"
            # smile
            mouth = m.findmouth(img)
            if mouth != 2: # did not return error
                mouthimg = crop(mouth)
                cv.SaveImage("webcam-m.jpg", mouthimg)
                # predict the captured emotion
                result = cs.predict(vectorize('webcam-m.jpg'))
                if result == 1:
                    print "and you are smiling! :-) "
                else:
                    print "and you are not smiling :-| "
            else:
                print "Failed to detect mouth"
    
    cv2.destroyWindow("preview")