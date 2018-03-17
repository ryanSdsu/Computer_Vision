import numpy as np
import cv2
import matplotlib.pyplot as plt

#Get image one to show in black and white
img1 = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Two/cat.bmp', 0)
plt.imshow(img1, cmap='gray')
plt.show()

#Get image two to show in black and white
img2 = cv2.imread('/Users/RJ/PycharmProjects/Computer_Vision/Homework_Two/dog.bmp', 0)
plt.imshow(img2, cmap='gray')
plt.show()

#FOURIER IMAGE ONE
#For image one we get the fourier image and display it
#GET FOURIER IMAGE* This is to get the fourier image
img1_f = np.fft.fft2(img1)
#DISPLAY ONLY* This is to visualize the fourier frequency first by centering the "0" in the middle
img1_shift = np.fft.fftshift(img1_f)
#DISPLAY ONLY* This is to "up" the values so you can visualize the frequency domain
magnitdue_spectrum_imageOne = 20*np.log(np.abs(img1_shift))
#DISPLAY ONLY* This is to display the fourier image
plt.imshow(magnitdue_spectrum_imageOne, cmap='gray')
plt.show()

#FOURIER IMAGE TWO
#For image two we get the fourier image and display it
#GET FOURIER IMAGE* This is to get the fourier image
img2_f = np.fft.fft2(img2)
#DISPLAY ONLY* This is to visualize the fourier frequency first by centering the "0" in the middle
img2_shift = np.fft.fftshift(img2_f)
#DISPLAY ONLY* This is to "up" the values so you can visualize the frequency domain
magnitdue_spectrum_imageTwo = 20*np.log(np.abs(img2_shift))
#DISPLAY ONLY* This is to display the fourier image
plt.imshow(magnitdue_spectrum_imageTwo, cmap='gray')
plt.show()


#Get the total # of rows/columns and the middle of both
totalRows = np.size(img1_f, 0)
totalColumns = np.size(img1_f,1)
centerRows = int(totalRows/2)
centerColumns = int(totalColumns/2)
#Play around with this number as it adjusts the size of the filter contribution of the Image One and Image Two
n = 20

#IMAGE ONE GET THE LOW FILTER
#From the fourier image one, edit it so that we can get a low frequency which is an image with a siloutte by blocking out the outside
#Make a completely black image
tempFrame = np.zeros_like(img1_shift)
#next block out everything but middle of the original fourier image i.e. crop the middle of the fourier
tempFrame[centerRows-n:centerRows+n, centerColumns-n:centerColumns+n] = img1_shift[centerRows - n:centerRows+n, centerColumns-n:centerColumns+n]
#Next display the new fourier
#DISPLAY ONLY* This is to "up" the values so you can visualize the frequency domain
tempFrameOne_magnitdue_spectrum = 20*np.log(np.abs(tempFrame))
#DISPLAY ONLY* This is to display the fourier image
plt.imshow(tempFrameOne_magnitdue_spectrum, cmap='gray')
plt.show()

#IMAGE ONE w/ THE LOW FILTER, return back to original
#Get the inverse now of the fourier image one to turn back to original image one (with now a siloutte)
inverse_tempframe_One = np.fft.ifft2(tempFrame)
inverse_tempframe_One_abs = np.abs(inverse_tempframe_One)
plt.imshow(inverse_tempframe_One_abs, cmap='gray')
plt.show()



#IMAGE TWO GET THE HIGH FILTER
#From the fourier image two, edit it so that we can get a low frequency which is an image with a siloutte by blocking out the outside
tempFrame_Two = np.copy(img2_shift)
tempFrame_Two[centerRows-n:centerRows+n, centerColumns-n:centerColumns+n] = 0
#Next display the new fourier
#DISPLAY ONLY* This is to "up" the values so you can visualize the frequency domain
tempFrameTwo_magnitdue_spectrum = 20*np.log(np.abs(tempFrame_Two))
#DISPLAY ONLY* This is to display the fourier image
plt.imshow(tempFrameTwo_magnitdue_spectrum, cmap='gray')
plt.show()

#IMAGE TWO w/ THE HIGH FILTER, return back to original
#Get the inverse now of the fourier image one to turn back to original image one (with now a siloutte)
inverse_tempframe_Two = np.fft.ifft2(tempFrame_Two)
inverse_tempframe_Two_abs = np.abs(inverse_tempframe_Two)
plt.imshow(inverse_tempframe_Two_abs, cmap='gray')
plt.show()


#Part 2
"""
Ok! So we got a black and white image both with a low filter and high filter. Great job!
The next step is to do it with RGB.
Do the same exact thing but pass in 3 seperate images via img1R, img1G, img1B
all through one filter and then do the same thing with the 2nd image via img2R, img2G, img2B. 

Here you should have 6 images. 3 in one low filter via RBG and 3 in a high filter via RGB.

We can now also blend the pictures by adding them together so: img1B + img2B and then element-wise dividie by 2.
cont:
(img1B + img2B) / 2 = finalImgB
(img1G + img2G) / 2 = finalImgG
(img1R + img2R) / 2 = finalImgR

To simplify the addition look for and use "np.add" which adds both img1B and img2B => "finalImgB = np.add(img1B, img2B)"
and then just divide by 2 i.e. "finalImgB = finalImgB / 2" 

From here merge them together (finalImgB, finalImgG, finalImgR) to create a blended image!
"""