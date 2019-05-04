import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def makeBlocks(windowsize_row, windowsize_col, array): #returns an array with the image divided into blocks
    blocks=[]
    #divide the window :)
    for r in range(0,array.shape[0] - windowsize_row+1, windowsize_row):#for each row of length(n))
        for c in range(0,array.shape[1] - windowsize_col+1, windowsize_col):#we get n columns of length(n
            window = array[r:r+windowsize_row,c:c+windowsize_col].astype('int16')#to create a block
            blocks.append(window)
    blocks=np.array(blocks)
    return blocks

def rnd_to_nxt_mul(size, i):
    return ((i - 1) // size + 1) * size

def add_black_lines_to(image):
    tmp_image = []
    height = image.shape[0]
    width = image.shape[1]
    black_pixels = np.array(black_pixel() * (rnd_to_nxt_mul(16, width) - width))
    # print(black_pixels)
    for row in image:
        # print(row)
        tmp_row = np.append(row, black_pixels, axis = 0)
        tmp_image.append(tmp_row)
    black_line = np.array(black_pixel() * rnd_to_nxt_mul(16, width))
    for j in range(rnd_to_nxt_mul(16, height) - height):
        tmp_image.append(black_line)
    return np.array(tmp_image)

def black_pixel():
    return [0]

def hierarchicalDivision(array):
    array=np.array(array) #ensure that we are handling a numpy array
    x, y = array.shape #it's an image, not a video. only 2 coordinates
    array2=[] #initialize the return array
    for i in range(0, x, 2):
        for j in range(0, y, 2):
            try:
                array2.append(array[i][j]) #make a singledimensional array
            except:
                continue #in the case of an array with even number of pixels
    array2=np.array(array2) #convert the array to numpy array
    array2=np.reshape(array2, (int(x/2), int(y/2))) #reshape the array to match the dimensions we want
    return(array2)

def motionEstimation(array):
    array=np.array(array) #ensure that we are handling a numpy array
    x, y = array.shape
    number_of_zeroes = x*y - np.count_nonzero(array) #count the number of zeroes in the given image
    if (number_of_zeroes >= 0.9 * x * y): #if the given array is at least 80% of zeroes
        #print("array almost full of zeroes")
        return(0)
    else:
        return(1)

def findTheDifferences(source, target):
    source=np.array(source)
    target=np.array(target)
    global hierimg1
    global hierimg2
    #we create 2 new images based on the original with lower resolutions to start searching on smaller blocks for faster discalification of blocks
    for i in range(2):
        hierimg1 = hierimg1 + [hierarchicalDivision(source)]
        source =hierarchicalDivision(source)
        hierimg2 = hierimg2 + [hierarchicalDivision(target)]
        target =hierarchicalDivision(target)

def ImageReconstrucrion(x, y,eikona2):
    r=1
    #we add lines of blocks on top of each other to create the image
    for i in range(x):
        #necessary initialisation to get dimensions
        output = np.array(eikona2[i*(y)])
        #we add blocks next to eaxh other to create lines of blocks
        for j in range(y-1):
            output = np.concatenate((output,eikona2[r]), axis=1)
            r= r +1
        r = r+1
        #necessary initialisation to get dimensions
        if(i==0):
            showim = output
        else:
            showim = np.concatenate((showim,output),axis=0)
    return(showim)#return reconstructed image

def Initializations(q):
    global hierimg1 , hierimg2
    hierimg1 = [images[0]]
    hierimg2 = [images[q]]
    findTheDifferences(images[0] ,images[q])#we take 2 images to compare by reducing their resolution for quicker calculations
    eikona1 = makeBlocks(4,4,hierimg1[2])#we divide the smallest image into 4x4 blocks
    eikona2 = makeBlocks(4,4,hierimg2[2])#we divide the smallest image into 4x4 blocks
    blockakia = []#whick blocks show movement
    for i in range(len(eikona1)):
        if motionEstimation(eikona1[i]-eikona2[i]):
            blockakia = blockakia + [i]#if movement then append the index of the block
    return (eikona1, eikona2, hierimg1, hierimg2, blockakia)
def FindMovingBlocks(blockakia,hierimg1,hierimg2):
    l = [8,16]
    for k in range(2):
        eikona1 = makeBlocks(l[k],l[k],hierimg1[1-k])#we divide the smallest image into lxl blocks
        eikona2 = makeBlocks(l[k],l[k],hierimg2[1-k])#we divide the smallest image into lxl blocks
        to_be_popped = []
        for i in range(len(blockakia)):#we will only check the blocks we saw movement in the previous hierarchical step
            if motionEstimation(eikona1[blockakia[i]]-eikona2[blockakia[i]]):
                continue #if there is still movement check next block
            else:
                to_be_popped = to_be_popped + [i]#if there is no movement then save the index of the block that will later be poppes from the list
        blockakia = [x for x in blockakia if x not in to_be_popped]#pop unwanted blocks
    return(eikona1, eikona2 ,blockakia)

def Program(eikona1,eikona2,hierimg1,hierimg2, images):
    #if the resolution cannot be divided perfectly by 16x16 blocks we add black pixels
    if(images[0].shape[0]%16 != 0 or images[0].shape[1]%16 !=0):
        for i in range(len(images)):
            images[i]=add_black_lines_to(images[i])
    images=np.array(images)
    for q in range(1,len(images)):
        #initialize arrays to get the required dimensions
        eikona1, eikona2, hierimg1, hierimg2, blockakia = Initializations(q)
        #calculate whick blocks are to be replaced
        eikona1, eikona2 ,blockakia = FindMovingBlocks(blockakia,hierimg1,hierimg2)

        #replace movement-blocks with backround-blocks
        for i in range(len(blockakia)):
            eikona2[blockakia[i]] = eikona1[blockakia[i]]

        y = int(images[0].shape[1]/16)
        x = int(images[0].shape[0]/16)
        #reconstuct image based on the original resolution
        images[q] = ImageReconstrucrion(x, y,eikona2)
        print ('frame ' + str(q) + ' Done' )
    return(images)

#necessary initializations
hierimg1 = []
hierimg2 = []
eikona1 = []
eikona2 = []
vidcap = cv2.VideoCapture('example.mp4')
success,image = vidcap.read()
images=[]
first=[]
success = True
number_of_frames=0


#read the video, convert each frame to grayscale. save the background as an individual frame.
while success: #store the video as list
    try:
        success,image = vidcap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert image to grayscale
        if number_of_frames==0:
            first.append(image) #first frame. only the background of the video
        images.append(image.astype('int16')) #see "continue" at line 16
        number_of_frames = number_of_frames + 1
    except :
        continue #it stores the dtype for some reason and we don't want that


#get the final images
images = np.array(Program(eikona1,eikona2,hierimg1,hierimg2, images), dtype=np.uint8)
#start a video file to write on
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (images[0].shape[1],images[0].shape[0]),False)
print('Creating video file...')
#write all the images in the video file
for i in range(len(images)):
    out.write(images[i])
print('The video "output.avi" ha been added/updated in your run folder.')
#release
out.release()
cv2.destroyAllWindows()
