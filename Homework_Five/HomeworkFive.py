import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from random import shuffle
random.seed(4)
import math
import scipy.signal
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Get the images (as gray ones) into the folders into three different lists
FlowerOneList = []
FlowerTwoList = []
FlowerThreeList = []

################################################################################################################
#Create a function that reads in the flowers and stores them into different lists

def access_flower_folder(folder_path, flowerList):

    import os
    for file in os.listdir(folder_path):

        img_file_path = folder_path + "/" + file
        #Get image one to show in black and white
        ImgOneOriginal = cv2.imread(img_file_path)
        ImgOneGray = cv2.cvtColor(ImgOneOriginal, cv2.COLOR_BGR2GRAY)
        ImegOneGrayResized = cv2.resize(ImgOneGray, (256,256)) #Might need to fix the ratio of the images due to accruacy

        flowerList.append(ImegOneGrayResized)

    return

################################################################################################################
#Create a function that creates a list of cropped 16 x 16 images of the originals, returns as a numpy array

def sift(image):
    histogramFinal = []
    x=0
    xEnd = 0
    y=0
    yEnd = 0
    widthIncrement = 16
    heightIncrement = 16

    for x in range(0, 256 - widthIncrement, widthIncrement):
        for y in range(0, 256 - heightIncrement, heightIncrement):
            xEnd = x + widthIncrement*2
            yEnd = y + heightIncrement*2
            tmpImage = image[x:xEnd, y:yEnd]

            histogramFinal = np.append(histogramFinal,Hog(tmpImage))

    return histogramFinal



################################################################################################################
#Create a function that takes in an image and returns the gradient and degrees, returns as a numpy array

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


################################################################################################################
#This is the main

access_flower_folder("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Five/data/Flower1",FlowerOneList)
access_flower_folder("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Five/data/Flower2",FlowerTwoList)
access_flower_folder("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Five/data/Flower3",FlowerThreeList)



#Create training for the Master labels (Add the labels, 64 1's, 2's, 3's)
tmpLabelOne = [1] * 64
tmpLabelTwo = [2] * 64
tmpLabelThree = [3] * 64
FlowerMasterLabelListTraining = tmpLabelOne + tmpLabelTwo + tmpLabelThree
print("This is the FlowerMasterLabelListTraining: {}".format(FlowerMasterLabelListTraining))
print("This is the length of FlowerMasterLabelListTraining: {}".format(len(FlowerMasterLabelListTraining)))

#Create training for the Master images (Add the labels, 64 1's, 2's, 3's)
shuffle(FlowerOneList)
shuffle(FlowerTwoList)
shuffle(FlowerThreeList)
trainingInstancesLength = int(len(FlowerOneList) * 0.8)
FlowerMasterImageListTraining = FlowerOneList[0:trainingInstancesLength] + FlowerTwoList[0:trainingInstancesLength] + FlowerThreeList[0:trainingInstancesLength]
print("This is the length of FlowerMasterImageListTraining: {}".format(len(FlowerMasterImageListTraining)))

#Create test(s) for the Master labels (Add the labels, 64 1's, 2's, 3's)
tmpLabelOne = [1] * 16
tmpLabelTwo = [2] * 16
tmpLabelThree = [3] * 16
FlowerMasterLabelListTest = tmpLabelOne + tmpLabelTwo + tmpLabelThree
print("This is the FlowerMasterLabelListTest: {}".format(FlowerMasterLabelListTest))
print("This is the length of FlowerMasterLabelListTest: {}".format(len(FlowerMasterLabelListTest)))

#Create training for the Master images (Add the labels, 64 1's, 2's, 3's)
FlowerMasterImageListTest = FlowerOneList[trainingInstancesLength:] + FlowerTwoList[trainingInstancesLength:] + FlowerThreeList[trainingInstancesLength:]
print("This is the length of FlowerMasterImageListTest: {}".format(len(FlowerMasterImageListTest)))

################################################################################################################
#Create final image and label list for training
randomizedIndexes = list(range(0, 192))
shuffle(randomizedIndexes)
print("This is the randomizedTrainingIndexes for training: {}".format(randomizedIndexes))

FlowerMasterImageListTrainingFinal = [FlowerMasterImageListTraining[i] for i in randomizedIndexes]
FlowerMasterLabelListTrainingFinal = [FlowerMasterLabelListTraining[i] for i in randomizedIndexes]

################################################################################################################
#Create final image and label list for test
randomizedIndexes = list(range(0, 48))
shuffle(randomizedIndexes)
print("This is the randomizedTrainingIndexes for test(s): {}".format(randomizedIndexes))

FlowerMasterImageListTestFinal = [FlowerMasterImageListTest[i] for i in randomizedIndexes]
FlowerMasterLabelListTestFinal = [FlowerMasterLabelListTest[i] for i in randomizedIndexes]


################################################################################################################
#Attempt sift i.e. HOG for 16x16 crops of each image via the list for the training and test

siftImageTraining = []
for image in FlowerMasterImageListTrainingFinal:
    siftImageTraining.append(sift(image))

siftImageTest = []
for image in FlowerMasterImageListTestFinal:
    siftImageTest.append(sift(image))


################################################################################################################
#Create an SVM model with the training set
# try :clf = svm.SVC(decision_function_shape='ovo')
# try :clf = svm.SVC(decision_function_shape='ovr')
# try :clf = svm.Linear_svc()

clf = svm.LinearSVC()
clf.fit(siftImageTraining,FlowerMasterLabelListTrainingFinal)
testOutput = clf.predict(siftImageTest)
matrix = confusion_matrix(FlowerMasterLabelListTestFinal,testOutput)
print(matrix)

################################################################################################################
#Find the accuracy:
matrix = np.array(matrix)
accuracyScoreOne = np.sum(matrix.diagonal())/np.sum(matrix)
accuracyScoreTwo = accuracy_score(FlowerMasterLabelListTestFinal,testOutput)
print(accuracyScoreOne)
print(accuracyScoreTwo)


################################################################################################################
#Create an SVM model with the training set
# try :clf = svm.SVC(decision_function_shape='ovo')
# try :clf = svm.SVC(decision_function_shape='ovr')
# try :clf = svm.Linear_svc()

clf = svm.SVC() #Uses C=1
clf.fit(siftImageTraining,FlowerMasterLabelListTrainingFinal)
testOutput = clf.predict(siftImageTest)
matrix = confusion_matrix(FlowerMasterLabelListTestFinal,testOutput)
print(matrix)

################################################################################################################
#Find the accuracy:
matrix = np.array(matrix)
accuracyScoreOne = np.sum(matrix.diagonal())/np.sum(matrix)
accuracyScoreTwo = accuracy_score(FlowerMasterLabelListTestFinal,testOutput)
print(accuracyScoreOne)
print(accuracyScoreTwo)
