import numpy as np
import cv2
import matplotlib.pyplot as plt

#Get image one to show in black and white
notreDameImgOneOriginal = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame.jpg')
notreDameImgOneGray = cv2.cvtColor(notreDameImgOneOriginal, cv2.COLOR_BGR2GRAY)
plt.imshow(notreDameImgOneGray, cmap='gray')
plt.show()

#Get image two to show in black and white
notreDameImgTwoOriginal = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame2.jpg')
notreDameImgTwoGray = cv2.cvtColor(notreDameImgTwoOriginal, cv2.COLOR_BGR2GRAY)
plt.imshow(notreDameImgTwoGray, cmap='gray')
plt.show()

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
imageOnceHarrisCoordinates = []
for i in range(len(notreDameImgOneHarrisCorner[0])):
    for j in range(len(notreDameImgOneHarrisCorner)):
        if notreDameImgOneHarrisCorner[j][i] >= 0.2 * maxImageOne:
            imageOnceHarrisCoordinates.append((j,i))

print(imageOnceHarrisCoordinates)
#From here crop a region around the coordinates
#Then from each region transform into the histogram via HOG
#Sweep through each pixel and measure the change in gradients and then plot them into different bins for the histogram
#You need something that returns the gradients and the magnitude
#Explore np.gradient though this just gives you the sweeping histogram for x and y
#Explore skimage.feature import hog, http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

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