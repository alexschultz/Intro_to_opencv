import os
import cv2
from matplotlib import pyplot as plt

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

# show individual color channels
lenaImage = cv2.imread(os.path.join('images', 'lena.png'), cv2.IMREAD_COLOR)
(channel_b, channel_g, channel_r) = cv2.split(lenaImage)
# (channel_b, channel_g, channel_r) = (lenaImage[:, :, 0], lenaImage[:, :, 1], lenaImage[:, :, 2])

cv2.imshow('lena blue', channel_b)
cv2.imshow('lena green', channel_g)
cv2.imshow('lena red', channel_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
# img = cv2.merge((b,g,r))


# histograms and histogram equalization
cv2.imshow('lena', lenaImage)
plt.hist(lenaImage.ravel(), 256, [0, 256])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# show individual color values in histogram
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([lenaImage], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
plt.xlim([0, 256])
plt.show()

lenaImage = cv2.imread(os.path.join('images', 'lena.png'), cv2.IMREAD_GRAYSCALE)
lenaEqualized = cv2.equalizeHist(lenaImage)
cv2.imshow('lena normal', lenaImage)
cv2.imshow('lena equalized', lenaEqualized)

plt.hist(lenaEqualized.ravel(), 256, [0, 256])
plt.hist(lenaImage.ravel(), 256, [0, 256])
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
img = cv2.imread(os.path.join('images', 'clahe.png'), 0)
equ = cv2.equalizeHist(img)
cv2.imshow('normal', img)
cv2.imshow('global equalized', equ)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(os.path.join('images', 'clahe.png'), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
cv2.imshow('normal', img)
cv2.imshow(' adaptive equalized', cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# deal with roi in image
lenaImage = cv2.imread(os.path.join('images', 'lena.png'), cv2.IMREAD_COLOR)
# [y1:y2, x1:x2]
roi = lenaImage[200:395, 206:365]
cv2.imshow('normal', lenaImage)
cv2.imshow('ROI', roi)

# resize image
largerRoi = cv2.resize(roi, None, fx=3, fy=3)
cv2.imshow('Larger ROI', largerRoi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img[273:333, 100:160] = ball

# draw on image
lenaImage = cv2.imread(os.path.join('images', 'lena.png'), cv2.IMREAD_COLOR)


def click_and_crop(event, x, y, flags, param):
    print(x, y)


cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

cv2.rectangle(lenaImage, (206, 200), (365, 395), (0, 255, 0), 1)
cv2.circle(lenaImage, (264, 271), 20, (255, 0, 0), 1)
cv2.circle(lenaImage, (337, 272), 20, (255, 0, 0), 1)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(lenaImage, 'Lena', (10, 500), font, 4, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('image', lenaImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# write image to disk
