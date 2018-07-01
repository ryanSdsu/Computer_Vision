import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal
import math

#Get image one to show in black and white
notreDameImgOneOriginal = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame.jpg')
notreDameImgOneGray = cv2.cvtColor(notreDameImgOneOriginal, cv2.COLOR_BGR2GRAY)
plt.imshow(notreDameImgOneGray, cmap='gray')
plt.show()
heightImageOne = np.size(notreDameImgOneGray, 0)
widthImageOne = np.size(notreDameImgOneGray,1)

#Get image two to show in black and white
notreDameImgTwoOriginal = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame2.jpg')
notreDameImgTwoGray = cv2.cvtColor(notreDameImgTwoOriginal, cv2.COLOR_BGR2GRAY)
plt.imshow(notreDameImgTwoGray, cmap='gray')
plt.show()
heightImageTwo = np.size(notreDameImgTwoGray, 0)
widthImageTwo = np.size(notreDameImgTwoGray,1)

#Convert both images to np floats
notreDameImgOneFloat = np.float32(notreDameImgOneGray)
notreDameImgTwoFloat = np.float32(notreDameImgTwoGray)

#Get the Harris Corner
notreDameImgOneHarrisCorner = cv2.cornerHarris(notreDameImgOneFloat, 2, 3, 0, 0.04)
notreDameImgTwoHarrisCorner = cv2.cornerHarris(notreDameImgTwoFloat, 2, 3, 0, 0.04)
#Display the Corners for Image One
plt.imshow(notreDameImgOneHarrisCorner, cmap='gray')
plt.show()
#Display the Corners for Image Two
plt.imshow(notreDameImgTwoHarrisCorner, cmap='gray')
plt.show()

#Gettng the coordinates of the Harris Corners
maxImageOne = np.amax(notreDameImgOneHarrisCorner)
imageOneHarrisCoordinates = []

for i in range(len(notreDameImgOneHarrisCorner[0])):
    for j in range(len(notreDameImgOneHarrisCorner)):
        if notreDameImgOneHarrisCorner[j][i] >= 0.2 * maxImageOne:
            imageOneHarrisCoordinates.append((j, i))
            print("Getting coordinates for Harris Corner Img One")

print(imageOneHarrisCoordinates)

maxImageTwo = np.amax(notreDameImgTwoHarrisCorner)
imageTwoHarrisCoordinates = []

for i in range(len(notreDameImgTwoHarrisCorner[0])):
    for j in range(len(notreDameImgTwoHarrisCorner)):
        if notreDameImgTwoHarrisCorner[j][i] >= 0.2 * maxImageTwo:
            imageTwoHarrisCoordinates.append((j, i))
            print("Getting coordinates for Harris Corner Img Two")

print(imageTwoHarrisCoordinates)


#Threshold the harris corner values so that everything is low won't be shown and the high values will be more highlighted
#Make the pixel that has a high value, make them red
#Anything values that are higher than 1 percent of maximum, label them as red
notreDameImgOneOriginal[notreDameImgOneHarrisCorner>0.01 * notreDameImgOneHarrisCorner.max()] = [0,0,255]
notreDameImgTwoOriginal[notreDameImgTwoHarrisCorner>0.01 * notreDameImgTwoHarrisCorner.max()] = [0,0, 255]

#Display the original pictures "with" the Harris Corners on top
plt.imshow(cv2.cvtColor(notreDameImgOneOriginal, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(notreDameImgTwoOriginal, cv2.COLOR_BGR2RGB))
plt.show()


#From here crop a region around the coordinates
#Then from each region transform into the histogram via HOG
#Sweep through each pixel and measure the change in gradients and then plot them into different bins for the histogram
#You need something that returns the gradients and the magnitude
#Explore np.gradient though this just gives you the sweeping histogram for x and y
#Explore skimage.feature import hog, http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

########Creating the HOG Begin############
#Use the imageOnceHarrisCoordinates
#We are not going to use padding, if the coordinates of the Harris corners are outside of the Harris Frame we disregard
#them

def hogToHistogram(image):
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
    maxValue = np.amax(filteredImageMagnitudes)
    filteredImageMagnitudes = filteredImageMagnitudes/maxValue
    filteredImageMagnitudes = np.abs(filteredImageMagnitudes * 255)
    filteredImageMagnitudes  = np.array(filteredImageMagnitudes,dtype='int16')

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
    numberOfBuckets = 4

    #Because numberOfBuckets is 4 we have 4 elements which will be constantly changing in the histogram
    #histogram[0] is [-180 => -90], histogram[1] is [-89 => 0], histogram[2] is [0 => 90], histogram[3] is [91=>180]
    histogram = [0,0,0,0]

    #Now we can populate the histogram
    #First we take the degrees from the filteredImageDegrees
    #And then sort it into the histogram based on either magnitude or a simple index
    #Notice there are 4 if statements to help with the sorting of the histogram on lines 71/72
    ############Be prepared to change "magnitude" on line 82 for assignment purposes###########
    for w in range(0, widthOfImage):
        for h in range(0, heightOfImage):
            degree = filteredImageDegrees[h,w]
            magnitude = filteredImageMagnitudes[h,w]
            if -180 <= degree <= -90:
                histogram[0] += magnitude
            elif -90 < degree <= 0:
                histogram[1] += magnitude
            elif 0 < degree <= 90:
                histogram[2] += magnitude
            elif 90 < degree <= 180:
                histogram[3] += magnitude

    #Now with the histogram, this is for the whole image
    return histogram




def cropOriginalImageIntoFourImageswithHarrisHistograms(image, harrisCoordinates):
    notreDameImgOneHog = cv2.imread(image)
    notreDameImgOneHogGray = cv2.cvtColor(notreDameImgOneHog, cv2.COLOR_BGR2GRAY)

    harrisFrame = np.zeros([32,32])
    hEye, wEye = harrisFrame.shape
    harrisAppendedHistograms = []

    #Iterate through each Harris Corner
    for harrisCorner in harrisCoordinates:
        #Checking for each Harris Corner bounds i.e. padding is the original picture
        if (hEye <= harrisCorner[0] <= heightImageOne - hEye):
            if (wEye <= harrisCorner[1] <= widthImageOne - wEye):
                histogramImage = []
                #Crop the original image
                notreDameImgOneHogGrayCropped = notreDameImgOneHogGray[harrisCorner[0]-hEye: harrisCorner[0] + hEye, harrisCorner[1]-wEye : wEye + harrisCorner[1]]

                #From each cropped image crop 4 more separate images
                notreDameImgOneHogGrayQ1 = notreDameImgOneHogGrayCropped[0: hEye, 0 : wEye]
                histogramOne = hogToHistogram(notreDameImgOneHogGrayQ1)
                for value in histogramOne:
                    histogramImage.append(value)

                notreDameImgOneHogGrayQ2 = notreDameImgOneHogGrayCropped[0: hEye, wEye : wEye + wEye]
                histogramTwo = hogToHistogram(notreDameImgOneHogGrayQ2)
                for value in histogramTwo:
                    histogramImage.append(value)

                notreDameImgOneHogGrayQ3 = notreDameImgOneHogGrayCropped[hEye: hEye + hEye, 0 : wEye]
                histogramThree = hogToHistogram(notreDameImgOneHogGrayQ3)
                for value in histogramThree:
                    histogramImage.append(value)

                notreDameImgOneHogGrayQ4 = notreDameImgOneHogGrayCropped[hEye: hEye + hEye, wEye : wEye + wEye]
                histogramFour = hogToHistogram(notreDameImgOneHogGrayQ4)
                for value in histogramFour:
                    histogramImage.append(value)

                harrisAppendedHistograms.append(histogramImage)

    return harrisAppendedHistograms

imageOneHarrisHistograms = cropOriginalImageIntoFourImageswithHarrisHistograms('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame.jpg' , imageOneHarrisCoordinates)
imageTwoHarrisHistograms = cropOriginalImageIntoFourImageswithHarrisHistograms('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame.jpg' , imageTwoHarrisCoordinates)

print(imageOneHarrisHistograms)
print(imageTwoHarrisHistograms)

########Creating the HOG END############






###Open CV has a way to "Create Feature Descriptors" as well as do the "Match Feature Descriptors" part of the assignment.
###This method does not encompass the use of the Harris Corners
###However this is a good reference
###You may need to create your HOG with the Harris Corners as well as the Match Feature Descriptors
###But at least we have a reference

###Create the SIFT detector via 500 points
sift = cv2.xfeatures2d.SIFT_create(500)
notreDameImgOneBinary = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame.jpg',0)
notreDameImgTwoBinary = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame2.jpg',0)
keyPointsImgOne, shapeDegreeAndMagnitudeImgOne = sift.detectAndCompute(notreDameImgOneBinary, None)
keyPointsImgTwo, shapeDegreeAndMagnitudeImgTwo = sift.detectAndCompute(notreDameImgTwoBinary, None)

###Gives the X/Y essentially Harris Corner Locations
print(keyPointsImgOne[0].pt)
###This is the HOG, the number of elements represents the degrees i.e. the length of the list and the values are the magnitudes
print(shapeDegreeAndMagnitudeImgOne[0])

###From here we can to the K-Nearest Neighbor Match
###We will use FLANN i.e. Fast Library for Approximate Nearest Neighbor
flannIndex = 0
indexParam = dict(algorithm = flannIndex, trees = 5)
searchParam = dict(checks=50)
#This creates the model
flann = cv2.FlannBasedMatcher(indexParam,searchParam)

###Now we need to feed the HOGs to the model
###k represents the number of nearest numbers
matches = flann.knnMatch(shapeDegreeAndMagnitudeImgOne,shapeDegreeAndMagnitudeImgTwo, k=2)

###Next loop through the matches and create an empty mask
matchesMask = [[0,0] for i in range(len(matches))]

###Now fill the mask with the match values based on a certain distance
###m and n represent the file descriptors (one of which represents the first image, the other represents the second)
for i,(m,n) in enumerate(matches):
    if m.distance < 0.4*n.distance:
        matchesMask[i]=[1,0]

###Now prep what it needed to draw everything
###This is saying that the Match Color is Green with a line
###And that Harris Corners are Red
drawingParameters = dict(matchColor = (0,255,0), singlePointColor = (0,0,255), matchesMask = matchesMask, flags = 0)

###Finally plot the two images into one
newImage = cv2.drawMatchesKnn(notreDameImgOneBinary, keyPointsImgOne, notreDameImgTwoBinary, keyPointsImgTwo, matches, None, **drawingParameters)

###Show the Image
plt.imshow(cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
plt.show()