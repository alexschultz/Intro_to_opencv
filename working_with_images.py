import os
import cv2

# read in an image
lenaImage = cv2.imread(os.path.join('images', 'lena.png'), cv2.IMREAD_COLOR)
cv2.imshow('lena', lenaImage)
print('image shape {}'.format(lenaImage.shape))
# wait for any key to be pressed
cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()

# read in an image in grayscale
lenaImage = cv2.imread(os.path.join('images', 'lena.png'), cv2.IMREAD_GRAYSCALE)
cv2.imshow('lena in gray', lenaImage)
# notice that when we switch to gray scale, the image size changes
print('gray image shape {}'.format(lenaImage.shape))
cv2.waitKey(0)
cv2.destroyAllWindows()

# filters
# https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
lenaImage = cv2.imread(os.path.join('images', 'lena.png'), cv2.IMREAD_COLOR)
blur = cv2.GaussianBlur(lenaImage, (9, 9), 0)
# blur the image
cv2.imshow('lena in blurred', blur)
cv2.imshow('lena raw', lenaImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# edge detection
# https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
lenaImage = cv2.imread(os.path.join('images', 'lena.png'), cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(lenaImage, 100, 200)
cv2.imshow('lena edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# show individual color spaces

# histogram and histogram equalization

# resize image

# deal with roi in image

# draw on image

# write image to disk