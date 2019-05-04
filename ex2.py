import numpy
import cv2
from PIL import Image
from cv2 import dct
import time
import sys
import os

start_time = time.time()
#necessary initialization
vidcap = cv2.VideoCapture('vid.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MPEG') #four character code. identifies the format of the video
out = cv2.VideoWriter('out.mp4',fourcc,25,(1280,720))
success,image = vidcap.read()
images=[]
diff=[] #array of differences
QP=-1 #quantization parameter
while (QP<=0):
    QP=input("Please give the quantization parameter. Be sure that is greater than zero!\n")
    QP=int(QP)
success = True
number_of_frames=0
frame=[]
while success: #store the video as list
    try:
        success,image = vidcap.read()
        if number_of_frames==0:
            frame.append(image)
            count=2 #save the first frame as JPEG
        images.append(image.astype('int32')) #see "continue" at line 16
        number_of_frames = number_of_frames + 1
    except :
        continue #it stores the dtype for some reason and we don't want that

images=numpy.array(images) #convert list to numpy array
x,y,z,w = images.shape #number of frames + dimensions of the array

#make the array of differences, then quantize it
for i in range(1, x):
    images[i,:,:,:] = images[i,:,:,] - images[i-1,:,:,:] #make the array as we need it for the DPCM
    images[i,:,:,:] = numpy.rint(numpy.divide(images[i,:,:,:],QP))

#make the output video
for i in range (1, x-1):
    images[i,:,:,:] = images[i,:,:,:] + images[i+1,:,:,:]
    out.write((images[i,:,:,:]).astype('uint8'))

cv2.destroyAllWindows()
out.release()

before= os.path.getsize('./vid.mp4')
after = os.path.getsize('./out.mp4')
ratio = float(before/after)
print("The compress ration is: ", ratio)
print("Your video is now in the same folder as the program!")
print("---Program's execution: %s seconds  ---" % (time.time() - start_time))
