import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.signal
import math

imgDir = '/Users/RJ/Desktop/Computer Vision/ha6/seq1';
startNumberOne = '000184'
startNumberTwo = '000185'
imgNameOne = imgDir + '/RGB_' + startNumberOne + '.jpg'
imgNameTwo = imgDir + '/RGB_' + startNumberTwo + '.jpg'

#Get image one to show in black and white
cupImgOneOriginal = cv2.imread(imgNameOne)
cupImgOneGray = cv2.cvtColor(cupImgOneOriginal, cv2.COLOR_BGR2GRAY)
# plt.imshow(cupImgOneGray, cmap='gray')
# plt.show()

#Get image two to show in black and white
cupImgTwoOriginal = cv2.imread(imgNameTwo)
cupImgTwoGray = cv2.cvtColor(cupImgTwoOriginal, cv2.COLOR_BGR2GRAY)
#plt.imshow(cupImgTwoGray, cmap='gray')
#plt.show()

#Load the mat file which has the labels for the coordinates of the bounding box
labelFile = loadmat('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Six/seq1.mat copy')
for i in labelFile:
    print(i)

gd= labelFile['gd']
print(np.shape(gd))



#index:0 => ImgOne
#index:1 => ImgTwo
#Get the ground truths from both images
gdOne = gd[0,0]
gdTwo = gd[0,1]

print(gdOne)
print(gdTwo)

#Crop the region
#crop imgOne with gdOne
croppedImg1 = cupImgOneGray[279:362, 226:331]
plt.imshow(croppedImg1, cmap='gray')
plt.show()

#crop imgTwo with gdOne
croppedImg2 = cupImgTwoGray[279:362, 226:331]
plt.imshow(croppedImg2, cmap='gray')
plt.show()

#Convert both images to np floats
cupImgOneFloat = np.float32(croppedImg1)
cupImgTwoFloat = np.float32(croppedImg2)

#Get the Harris Corner
cupImgOneHarrisCorner = cv2.cornerHarris(cupImgOneFloat, 2, 3, 0, 0.04)
cupImgTwoHarrisCorner = cv2.cornerHarris(cupImgTwoFloat, 2, 3, 0, 0.04)
#Display the Corners for Image One
plt.imshow(cupImgOneHarrisCorner, cmap='gray')
plt.show()
#Display the Corners for Image Two
plt.imshow(cupImgTwoHarrisCorner, cmap='gray')
plt.show()


#With the harris corners, do a local max surpress of the image
#Use a 3x3 region to scan for the highest value
#In the region make all other values 0

#Get the height and width of the cub image
def non_maxima(harris, width=3):
    harris_max = np.zeros_like(harris)
    for x in range(width, harris.shape[0] - width):
        for y in range(width, harris.shape[1] - width):
            pixel = harris[x,y]
            if pixel == np.amax(harris[x-width:x+width+1, y-width:y+width+1]):
                harris_max[x,y] = pixel

    return harris_max

#Return the reduced number of Harris corners for img one
non_max_One = non_maxima(cupImgOneHarrisCorner, width=3)
plt.imshow(non_max_One, cmap='gray')
plt.show()

#Return the reduced number of Harris corners for img two
non_max_Two = non_maxima(cupImgTwoHarrisCorner, width=3)
plt.imshow(non_max_Two, cmap='gray')
plt.show()

#Compare the maxing Harris Corners

#This function gets all of the Harris corners and sorts them by their values
#In other words returning a triple with 3 values => (Harris Value, height coordinate, wigth coordinate)
def getHarrisCoordinatesandValues(refinedHarris):
    harrisValuesandCoordinates = []
    for heightIndex in range(refinedHarris.shape[0]):
        for widthIndex in range(refinedHarris.shape[1]):
            harrisValue = refinedHarris[heightIndex][widthIndex]
            harrisValuesandCoordinates.append((harrisValue, heightIndex, widthIndex))

    harrisValuesandCoordinates = sorted(harrisValuesandCoordinates, key=lambda l:l[0], reverse=True)

    return harrisValuesandCoordinates

harrisCornerListOne = getHarrisCoordinatesandValues(non_max_One)
harrisCornerListTwo = getHarrisCoordinatesandValues(non_max_Two)


#Here is the HOG definition

def Hog(image):
    #Create a sliding window 3x3 via the x filter and y filter
    #kernel is another name for filter
    xkernal = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])
    ykernal = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])

    #This applies the xfilter which picks up the "vertical lines" i.e. gradients, the change in horizontal intensities (differences)
    xFilterdImage = scipy.signal.convolve2d(image, xkernal, "valid")

    #This applies the yfilter which picks up the "horizontal lines" i.e. gradients, the change in vertical intensities (differences)
    yFilterdImage = scipy.signal.convolve2d(image, ykernal, "valid")

    #Now we are going to extract the gradients to get the magnitudes (this requires both the x and y gradients)
    #We use this equation squareroot(x^2 + y^2)
    #First we square all of the gradients in the xFilterdImage and y FilteredImage
    xFilterdImageSquared = np.square(xFilterdImage)
    yFilterdImageSquared = np.square(yFilterdImage)

    #Then we add them together and take the squareroot
    #"filteredImageMagnitudes" has the magnitudes, we will need this later
    filteredImageMagnitudes = np.sqrt(xFilterdImageSquared + yFilterdImageSquared)
    # maxValue = np.amax(filteredImageMagnitudes)
    # filteredImageMagnitudes = filteredImageMagnitudes/maxValue
    # filteredImageMagnitudes = np.abs(filteredImageMagnitudes * 255)
    # filteredImageMagnitudes  = np.array(filteredImageMagnitudes,dtype='int16')


    #Now we are going to extract the gradients to get the angles (this requires both the x and y gradients)
    #We use this equation tan-1(y/x)
    #First we will create a duplicate image of the same size containing all 0's
    #And then populate within it the quotients/arctans of all the gradients in the yFilterdImage over xFilteredImage (iteration)
    #This is converted into degrees
    filteredImageDegrees = np.zeros_like(filteredImageMagnitudes)
    heightOfImage,widthOfImage = np.shape(filteredImageDegrees)

    for w in range(0, widthOfImage):
        for h in range(0, heightOfImage):
            x = xFilterdImage[h,w]
            y = yFilterdImage[h,w]
            if(x == 0):
                quotient = 0
            else:
                filteredImageDegrees[h,w] = 180 * np.arctan2(y,x) / math.pi

    #Now that we have both of the images via the magnitudes and the degrees, filteredImageMagnitudes and filteredImageDegrees
    #We can separate them into buckets
    #By buckets we mean a histogram like [0-45][46-90][90-135][135-180] and [0- -45][-46 - -90][-90 - -135][-135 - -180]
    #Let's make the histogram via HOG by first selecting the number of buckets (in this case it's 4 buckets per histogram)
    ############Be prepared to change "numberOfBuckets" on line 68 for assignment purposes###########

    #Because numberOfBuckets is 4 we have 4 elements which will be constantly changing in the histogram
    #histogram[0] is [-180 => -90], histogram[1] is [-89 => 0], histogram[2] is [0 => 90], histogram[3] is [91=>180]
    numberOfBins = 12
    binRange = 360 / numberOfBins
    histogram = [0] * numberOfBins

    #Now we can populate the histogram
    #First we take the degrees from the filteredImageDegrees
    #And then sort it into the histogram based on either magnitude or a simple index
    #Notice there are 4 if statements to help with the sorting of the histogram on lines 71/72
    ############Be prepared to change "magnitude" on line 82 for assignment purposes###########
    for w in range(0, widthOfImage):
        for h in range(0, heightOfImage):
            degree = filteredImageDegrees[h,w]
            magnitude = filteredImageMagnitudes[h,w]
            binIndex = np.int((degree + 180)/binRange)
            if binIndex == numberOfBins:
                binIndex = numberOfBins - 1
            histogram[binIndex] += magnitude

            # if -180 <= degree <= -90:
            #     histogram[0] += magnitude
            # elif -90 < degree <= 0:
            #     histogram[1] += magnitude
            # elif 0 < degree <= 90:
            #     histogram[2] += magnitude
            # elif 90 < degree <= 180:
            #     histogram[3] += magnitude

    maxValueHistorgram = max(histogram)
    if maxValueHistorgram != 0:
        histogram  = [i/maxValueHistorgram for i in histogram]

    return reorderHistogramLargestInFront(histogram)

################################################################################################################
#This function takes the largest element, moves it in front and then concatenates the rest to the end, returns numpy array
def reorderHistogramLargestInFront(histogram):
    histogram = np.array(histogram)

    maxIndex = np.argmax(histogram)

    ending = histogram[:maxIndex]
    beginning = histogram[maxIndex:]

    histogram = np.append(beginning,ending)
    return histogram



#Now we will take the coordinates and crop a region around the harris corner (top 10 of the list)
#We can use 16 x 16 because the cropped cup is about 100x100 from the 300x300 original image
#Get the height and width of the cub image
#This returns a new list
#a triple with 3 values => (height coordinate, wigth coordinate, hog histogram)
def harrisCornerRegion(harrisImg,harrisCornerList):
    harrisCoordinatesAndHistogram = []

    borderedImg = cv2.copyMakeBorder(harrisImg,8,8,8,8,cv2.BORDER_CONSTANT,value=0)

    for idx, i in enumerate(harrisCornerList):
        harrisCornerRegion = borderedImg[harrisCornerList[idx][1]-8:harrisCornerList[idx][1]+8, harrisCornerList[idx][2]-8:harrisCornerList[idx][2]+8]
        print(harrisCornerRegion.shape)
        plt.imshow(harrisCornerRegion, cmap='gray')
        plt.show()
        histogramOfHarrisRegion = reorderHistogramLargestInFront(Hog(harrisCornerRegion))
        harrisCoordinatesAndHistogram.append((harrisCornerList[idx][1], harrisCornerList[idx][2], histogramOfHarrisRegion))

    return harrisCoordinatesAndHistogram

harrisCoordinatesAndHistogramOne = harrisCornerRegion(non_max_One,harrisCornerListOne)
harrisCoordinatesAndHistogramTwo = harrisCornerRegion(non_max_Two,harrisCornerListTwo)

for i in range(9):
    print(harrisCornerListOne[i])


