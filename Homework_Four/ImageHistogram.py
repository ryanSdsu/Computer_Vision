import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.signal


#Get image one to show in black and white
notreDameImgOneOriginal = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Four/NotreDame.jpg')
notreDameImgOneGray = cv2.cvtColor(notreDameImgOneOriginal, cv2.COLOR_BGR2GRAY)
plt.imshow(notreDameImgOneGray, cmap='gray')
plt.show()

#Create a sliding window 3x3 via the x filter and y filter
#kernel is another name for filter
xkernal = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])
ykernal = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])

#This applies the xfilter which picks up the "vertical lines" i.e. gradients, the change in horizontal intensities (differences)
xFilterdImage = scipy.signal.convolve2d(notreDameImgOneGray, xkernal, "valid")
plt.imshow(xFilterdImage, cmap='gray')
plt.show()

#This applies the yfilter which picks up the "horizontal lines" i.e. gradients, the change in vertical intensities (differences)
yFilterdImage = scipy.signal.convolve2d(notreDameImgOneGray, ykernal, "valid")
plt.imshow(yFilterdImage, cmap='gray')
plt.show()

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

#Show the image via magnitudes
plt.imshow(filteredImageMagnitudes, cmap='gray')
plt.show()

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

print(np.amin(filteredImageDegrees))
print(np.amax(filteredImageDegrees))
#Now with the histogram, this is for the whole image
print(histogram)