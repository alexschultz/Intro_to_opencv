import cv2
import os


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

# wait for any key to be pressed
cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
