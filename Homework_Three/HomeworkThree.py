import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats.stats import pearsonr

#Get image one to show in black and white
img1Original = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg', 0)
plt.imshow(img1Original, cmap='gray')
plt.show()

#Get image one w/ eye to show in black and white
img1Eye = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg', 0)
plt.imshow(img1Eye, cmap='gray')
plt.show()

#Get the height and width of the eye image as well as the original
hEye, wEye = img1Eye.shape
hOriginal, wOriginal = img1Original.shape

print("Size of cropped eye height: {}".format(hEye))
print("Size of cropped eye width: {}".format(wEye))

#This gets the cropped size of the original image and grabs via the eye
croppedImg1 = img1Original[0:hEye, 0:wEye]
plt.imshow(croppedImg1, cmap='gray')
plt.show()

#This is an auto built function which grabs the correlation through cv2 and is what we're tring to get
cor = cv2.matchTemplate(img1Original,img1Eye,cv2.TM_CCORR_NORMED)
minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(cor)
print("Location of cor eye height: {}".format(maxLocation[0]))
print("Location of cor eye width: {}".format(maxLocation[1]))
plt.imshow(cor, cmap='gray')
plt.show()

#Create a custom sliding window (i.e. do the correlation from scratch )
###
#This gets the cropped size of the image via Test check
croppedImgTest = img1Original[111:111+hEye, 105:105+wEye]
plt.imshow(croppedImgTest, cmap='gray')
plt.show()


dtype = [('score', float), ('height', int), ('width', int)]
maxList = []

#This does the "Correlation" i.e. Q1
#tempImage1 = np.zeros_like(img1Original)
tempImage1 = np.zeros_like(img1Original)

for heightIndex in range(hOriginal - hEye):
    for widthIndex in range(wOriginal - wEye):
        croppedImg1 = img1Original[heightIndex:hEye + heightIndex, widthIndex:wEye + widthIndex]
        flattenedCroppedImg1 = croppedImg1.flatten()
        flattenedImg1Eye = img1Eye.flatten()
        flattenedImg1Test = croppedImgTest.flatten()

        #This does the "Correlation" i.e. Q1
        correlationValue = pearsonr(flattenedCroppedImg1, flattenedImg1Eye)
        correlationValue = correlationValue[0] * 255


        # #This does the "SSD" i.e. Q2
        # correlationValue = ((flattenedCroppedImg1-flattenedImg1Eye)**2).sum()
        # correlationValue = 1 - float(correlationValue / 31046)
        # correlationValue = float(correlationValue * 255)


        tempImage1[heightIndex][widthIndex] = correlationValue
        maxList.append((correlationValue, heightIndex, widthIndex))


plt.imshow(tempImage1, cmap='gray')
plt.show()

a = np.array(maxList, dtype=dtype)
a = np.sort(a, order = "score")

print(a[-1])

