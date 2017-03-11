"""
input: a loaded image; 
output: [[x,y],[width,height]] of the detected mouth area
"""

import cv

def findface(img):

  # INITIALIZE: loading the classifiers
  haarFace = cv.Load('haarcascade_frontalface_default.xml')
  # running the classifiers
  storage = cv.CreateMemStorage()
  detectedFace = cv.HaarDetectObjects(img, haarFace, storage)

  # FACE: find the largest detected face as detected face
  maxFaceSize = 0
  maxFace = 0
  if detectedFace:
   for face in detectedFace: # face: [0][0]: x; [0][1]: y; [0][2]: width; [0][3]: height 
    if face[0][3]* face[0][2] > maxFaceSize:
      maxFaceSize = face[0][3]* face[0][2]
      maxFace = face
  
  if maxFace == 0: # did not detect face
    return 2
  else:
    return maxFace

