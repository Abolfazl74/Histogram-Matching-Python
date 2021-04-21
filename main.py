# @Author: Ashiyan3@gmail.com
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as pimg

img1 = cv.imread("test1.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("test2.jpg", cv.IMREAD_GRAYSCALE)

row1, col1 = img1.shape[:2]
row2, col2 = img2.shape[:2]

def df(img, deb=""):
    values = np.zeros((256))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[round(img[i,j])]+=1
                
    return values


def cdf(hist):  
    cdf = np.zeros((256))
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i]= cdf[i-1]+hist[i]
        
    cdf = [ele*255/cdf[-1] for ele in cdf]     
    return cdf

def equalize_image(image):
    equa = cdf(df(image))
    image_equalized = np.zeros_like(image)
    #image_equalized = np.interp(x=image, xp=range(0,256), fp=equa)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x, y]
            image_equalized[x, y] = equa[pixel_val]
            
    return image_equalized

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
	
	
	
plt.figure(0)
plt.title('Input image')
plt.imshow(img1, cmap=cm.gray, vmin=0, vmax=256)
plt.figure(1)
plt.title('Histogram of default image')
plt.plot(df(img1))
plt.figure(2)
plt.title('Template image')
plt.imshow(img2, cmap=cm.gray, vmin=0, vmax=256)
plt.figure(3)
plt.title('Histogram of template image')
plt.plot(df(img2))
plt.show()

eq1 = equalize_image(img1)
eqHist1 = cdf(df(eq1))
hist1 = df(eq1)
eq2 = equalize_image(img2)
eqHist2 = cdf(df(eq2))
hist2 = df(eq2)

plt.figure(4)
plt.title('Equalized input image')
plt.imshow(eq1, cmap=cm.gray, vmin=0, vmax=256)
plt.figure(5)
plt.title('Histogram of equalized input image')
plt.plot(hist1)
plt.figure(6)
plt.title('Equalized template image')
plt.imshow(eq2, cmap=cm.gray, vmin=0, vmax=256)
plt.figure(7)
plt.title('Histogram of equalized template image')
plt.plot(hist2)
plt.show()

mappedHist = np.zeros_like(hist1)
matched_image = np.zeros_like(img1)

for i in range(1, 256):
    if(eqHist2[i] != 0):
        idx = find_nearest(eqHist1, eqHist2[i])
        mappedHist[i] = hist1[idx]
        
match_equa = cdf(mappedHist)
#matched_image = np.interp(x=img1, xp=range(0,256), fp=mappedHist)
#cv.imwrite('newimg.jpg', matched_image)
for x in range(matched_image.shape[0]):
    for y in range(matched_image.shape[1]):
        pixel_val = img1[x, y]
        matched_image[x, y] = match_equa[pixel_val]
		
plt.figure(9)
plt.title('Matched new image Histogram')
plt.plot(df(matched_image))
plt.figure(10)
plt.title('Matched new image')
plt.imshow(matched_image, cmap=cm.gray)
plt.show()
