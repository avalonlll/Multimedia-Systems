from numpy import array
import cv2
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


def quantize(array):
    #quantize the image
    for i in range(width):
        for j in range(height):
            arr[i][j]=int(arr[i][j]/QP)
    return arr

def dequantize(array):
    for i in range(width):
        for j in range(height):
            arr[i][j]=int(arr[i][j]*QP)
    return arr

def RLE(array):
    #run length encoding
    #we will make two separated lists
    #one for the values of the pixels
    #and one for the times that each value is on the image
    #we are scanning the image row by row
    flag=arr[0][0]
    values=[]
    values.append(arr[0][0])
    times=[]
    comp=[]
    count=0
    for i in range(width):
        for j in range(height):
            if arr[i][j]!=flag :
                flag=arr[i][j]
                values.append(arr[i][j])
                times.append(count)
                count=1
            else:
                count=count+1
            if (i==width-1 and j==height-1): #last pixel of the image
                times.append(count)

    for i in range(len(values)):
        comp.append(str(values[i])+str(",")+str(times[i]))
    return(comp)

def compress_ratio(width, height, comp):
    #compress ratio
    ratio=width*height/len(comp)
    print("The compress ratio is: ", ratio)

def reconstruct(comp):
    loop=0
    n=0
    recon = [[]]
    for i in range(len(comp)):
        element = comp[i].split(",")
        loop = loop + int(element[1])
        if loop > width:
            n=n+1
            recon = recon + [[]]
            loop = int(element[1])
        for k in range(int(element[1])):
            recon[n] = recon[n] + [int(element[0])]
    recon = dequantize(recon)
    imgplot = plt.imshow(recon,cmap="gray")
    print("---Program's execution: %s seconds  ---" % (time.time() - start_time))
    plt.show()

                        ##main program##
QP=-1 #quantization parameter
while (QP<=0):
    QP=input("Please give the quantization parameter. Be sure that is greater than zero!\n")
    QP=int(QP)
try:
    image=(input("Please give the image's name!\n"))
    start_time = time.time()
    img = cv2.imread(str(image),0)
    arr=array(img)
    width, height = img.shape[:2]
    quantize(arr)
    comp=RLE(arr)
    compress_ratio(width, height, comp)
    reconstruct(comp)
except:
    print("Wrong input :( Try again!")
