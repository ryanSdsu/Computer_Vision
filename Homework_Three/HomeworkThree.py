import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats.stats import pearsonr

# https://stackoverflow.com/questions/6991471/computing-cross-correlation-function
# Use the cv match template for the built in functions
# cv.MatchTemplate(templateCv, imageCv, resultCv, cv.CV_TM_CCORR_NORMED)

def builtInNormalizedCrossCorrelation(originalImage, croppedImage, scale):
    print("Data for builtInNormalizedCrossCorrelation... with a scale of: {}".format(scale))
    #Get image one to show in black and white
    img1Original = cv2.imread(originalImage, 0)
    plt.imshow(img1Original, cmap='gray')
    plt.show()

    #Get image one w/ eye to show in black and white
    img1Eye = cv2.imread(croppedImage, 0)
    hEye, wEye = img1Eye.shape
    hEye = int(hEye * scale)
    wEye = int(wEye * scale)
    img1Eye = cv2.resize(img1Eye, (int(wEye * scale), int(hEye * scale)))
    plt.imshow(img1Eye, cmap='gray')
    plt.show()

    #Get the height and width of the eye image as well as the original

    print("Size of cropped eye height: {}".format(hEye))
    print("Size of cropped eye width: {}".format(wEye))

    #This gets the cropped size of the original image and grabs via the eye
    croppedImg1 = img1Original[112:hEye + 112, 107:wEye + 107]
    plt.imshow(croppedImg1, cmap='gray')
    plt.show()

    #This is an auto built function which grabs the correlation through cv2 and is what we're tring to get
    cor = cv2.matchTemplate(img1Original,img1Eye,cv2.TM_CCORR_NORMED)
    minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(cor)
    print("Location of cor eye height: {}".format(maxLocation[0]))
    print("Location of cor eye width: {}".format(maxLocation[1]))
    print("Localization error: ({}, {})".format(abs(maxLocation[0] - 117), abs(maxLocation[1] - 117)))
    plt.imshow(cor, cmap='gray')
    plt.show()

    #This gets the cropped size of the original image and grabs via the eye
    croppedImg1 = cor[112-hEye: 112 + hEye, 107-wEye : wEye + 107]
    plt.imshow(croppedImg1, cmap='gray')
    plt.show()

def builtInSSQD(originalImage, croppedImage, scale):
    print("Data for builtInSSQD... with a scale of: {}".format(scale))
    #Get image one to show in black and white
    img1Original = cv2.imread(originalImage, 0)
    plt.imshow(img1Original, cmap='gray')
    plt.show()

    #Get image one w/ eye to show in black and white
    img1Eye = cv2.imread(croppedImage, 0)
    hEye, wEye = img1Eye.shape
    hEye = int(hEye * scale)
    wEye = int(wEye * scale)
    img1Eye = cv2.resize(img1Eye, (int(wEye * scale), int(hEye * scale)))
    plt.imshow(img1Eye, cmap='gray')
    plt.show()


    #Get the height and width of the eye image as well as the original
    print("Size of cropped eye height: {}".format(hEye))
    print("Size of cropped eye width: {}".format(wEye))

    #This gets the cropped size of the original image and grabs via the eye
    croppedImg1 = img1Original[112:hEye + 112, 107:wEye + 107]
    plt.imshow(croppedImg1, cmap='gray')
    plt.show()

    #This is an auto built function which grabs the correlation through cv2 and is what we're tring to get
    cor = cv2.matchTemplate(img1Original,img1Eye,cv2.TM_SQDIFF)
    minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(cor)
    print("Location of cor eye height: {}".format(maxLocation[0]))
    print("Location of cor eye width: {}".format(maxLocation[1]))
    print("Localization error: ({}, {})".format(abs(maxLocation[0] - 117), abs(maxLocation[1] - 117)))
    plt.imshow(cor, cmap='gray')
    plt.show()

    #This gets the cropped size of the original image and grabs via the eye
    croppedImg1 = cor[112-hEye: 112 + hEye, 107-wEye : wEye + 107]
    plt.imshow(croppedImg1, cmap='gray')
    plt.show()

def builtInZeroMeanCorrelation(originalImage, croppedImage, scale):
    print("Data for builtInZeroMeanCorrelation... with a scale of: {}".format(scale))
    #Get image one to show in black and white
    img1Original = cv2.imread(originalImage, 0)
    plt.imshow(img1Original, cmap='gray')
    plt.show()

    #Get image one w/ eye to show in black and white
    img1Eye = cv2.imread(croppedImage, 0)
    hEye, wEye = img1Eye.shape
    hEye = int(hEye * scale)
    wEye = int(wEye * scale)
    img1Eye = cv2.resize(img1Eye, (int(wEye * scale), int(hEye * scale)))
    plt.imshow(img1Eye, cmap='gray')
    plt.show()


    #Get the height and width of the eye image as well as the original
    print("Size of cropped eye height: {}".format(hEye))
    print("Size of cropped eye width: {}".format(wEye))

    #This gets the cropped size of the original image and grabs via the eye
    croppedImg1 = img1Original[112:hEye + 112, 107:wEye + 107]
    plt.imshow(croppedImg1, cmap='gray')
    plt.show()

    #This is an auto built function which grabs the correlation through cv2 and is what we're tring to get
    cor = cv2.matchTemplate(img1Original,img1Eye,cv2.TM_CCORR)
    minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(cor)
    print("Location of cor eye height: {}".format(maxLocation[0]))
    print("Location of cor eye width: {}".format(maxLocation[1]))
    print("Localization error: ({}, {})".format(abs(maxLocation[0] - 117), abs(maxLocation[1] - 117)))
    plt.imshow(cor, cmap='gray')
    plt.show()

    #This gets the cropped size of the original image and grabs via the eye
    croppedImg1 = cor[112-hEye: 112 + hEye, 107-wEye : wEye + 107]
    plt.imshow(croppedImg1, cmap='gray')
    plt.show()

def zeroMeanCorrelation(originalImage, croppedImage, scale):
    print("Data for zeroMeanCorrelation... with a scale of: {}".format(scale))
    #Get image one to show in black and white
    img1Original = cv2.imread(originalImage, 0)

    #Get image one w/ eye to show in black and white
    img1Eye = cv2.imread(croppedImage, 0)
    hEye, wEye = img1Eye.shape
    hEye = int(hEye * scale)
    wEye = int(wEye * scale)
    img1Eye = cv2.resize(img1Eye, (int(wEye * scale), int(hEye * scale)))
    plt.imshow(img1Eye, cmap='gray')
    plt.show()

    #Get the height and width of the eye image as well as the original
    hOriginal, wOriginal = img1Original.shape

    print("Size of cropped eye height: {}".format(hEye))
    print("Size of cropped eye width: {}".format(wEye))

    dtype = [('score', float), ('height', int), ('width', int)]
    maxList = []

    #This does the "Correlation" i.e. Q1
    tempImage1 = np.zeros_like(img1Original)

    max = 0
    for heightIndex in range(hOriginal - hEye):
        for widthIndex in range(wOriginal - wEye):
            croppedImg1 = img1Original[heightIndex:hEye + heightIndex, widthIndex:wEye + widthIndex]
            flattenedCroppedImg1 = croppedImg1.flatten()
            flattenedImg1Eye = img1Eye.flatten()

            #This does the "Correlation" i.e. Q1
            correlationValue = pearsonr(flattenedCroppedImg1, flattenedImg1Eye)
            correlationValue = correlationValue[0] * 255

            tempImage1[heightIndex][widthIndex] = correlationValue
            maxList.append((correlationValue, heightIndex, widthIndex))

    plt.imshow(tempImage1, cmap='gray')
    plt.show()

    #This gets the cropped size of the original image and grabs via the eye
    croppedImg1 = tempImage1[112-hEye: 112 + hEye, 107-wEye : wEye + 107]
    plt.imshow(croppedImg1, cmap='gray')
    plt.show()

    a = np.array(maxList, dtype=dtype)
    a = np.sort(a, order = "score")

    print(a[-1])
    print("Localization error: ({}, {})".format(abs(a[-1][1] - 117), abs(a[-1][2] - 117)))

def sumSquareDifference(originalImage, croppedImage, scale):
    print("Data for sumSquareDifference... with a scale of: {}".format(scale))
    #Get image one to show in black and white
    img1Original = cv2.imread(originalImage, 0)

    #Get image one w/ eye to show in black and white
    img1Eye = cv2.imread(croppedImage, 0)
    hEye, wEye = img1Eye.shape
    hEye = int(hEye * scale)
    wEye = int(wEye * scale)
    img1Eye = cv2.resize(img1Eye, (int(wEye * scale), int(hEye * scale)))
    plt.imshow(img1Eye, cmap='gray')
    plt.show()

    #Get the height and width of the eye image as well as the original
    hOriginal, wOriginal = img1Original.shape

    print("Size of cropped eye height: {}".format(hEye))
    print("Size of cropped eye width: {}".format(wEye))

    dtype = [('score', float), ('height', int), ('width', int)]
    maxList = []

    #This does the "Correlation" i.e. Q1
    tempImage1 = np.zeros_like(img1Original)

    max = 0
    for heightIndex in range(hOriginal - hEye):
        for widthIndex in range(wOriginal - wEye):
            croppedImg1 = img1Original[heightIndex:hEye + heightIndex, widthIndex:wEye + widthIndex]
            flattenedCroppedImg1 = croppedImg1.flatten()
            flattenedImg1Eye = img1Eye.flatten()

            # #This gets the max value for the "SSD" i.e. Q2
            correlationValue = ((flattenedCroppedImg1-flattenedImg1Eye)**2).sum()
            if correlationValue > max:
                max = correlationValue


    for heightIndex in range(hOriginal - hEye):
        for widthIndex in range(wOriginal - wEye):
            croppedImg1 = img1Original[heightIndex:hEye + heightIndex, widthIndex:wEye + widthIndex]
            flattenedCroppedImg1 = croppedImg1.flatten()
            flattenedImg1Eye = img1Eye.flatten()

            # #This does the "SSD" i.e. Q2
            correlationValue = ((flattenedCroppedImg1-flattenedImg1Eye)**2).sum()
            correlationValue = 1 - float(correlationValue / max)
            correlationValue = float(correlationValue * 255)


            tempImage1[heightIndex][widthIndex] = correlationValue
            maxList.append((correlationValue, heightIndex, widthIndex))

    plt.imshow(tempImage1, cmap='gray')
    plt.show()

    #This gets the cropped size of the original image and grabs via the eye
    croppedImg1 = tempImage1[112-hEye: 112 + hEye, 107-wEye : wEye + 107]
    plt.imshow(croppedImg1, cmap='gray')
    plt.show()

    a = np.array(maxList, dtype=dtype)
    a = np.sort(a, order = "score")

    print(a[-1])
    print("Localization error: ({}, {})".format(abs(a[-1][1] - 117), abs(a[-1][2] - 117)))


#Location of eye 117,117
builtInZeroMeanCorrelation("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
                           "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 0.5)

builtInZeroMeanCorrelation("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
                           "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 1)

builtInZeroMeanCorrelation("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
                           "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 2)

builtInNormalizedCrossCorrelation("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
            "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 0.5)

builtInNormalizedCrossCorrelation("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
            "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 1)

builtInNormalizedCrossCorrelation("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
            "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 2)

builtInSSQD("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
                "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 0.5)

builtInSSQD("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
            "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 1)

builtInSSQD("/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1.jpg",
            "/Users/RJ/PycharmProjects/Computer_Vision/Homework_Three/image1Eye.jpg", 2)
