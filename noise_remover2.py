import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


SOURCE_PATH_1 = '../test/tahe1_11_0_0_255.png'
SOURCE_PATH_2 = '../test/tahe1_11_0_250_255.png'
DESTINATION_PATH = '../test_out/tahe1_11_0_250_255.png'

kernel_3x3 = np.ones((3,3), np.uint8)

mask = cv2.imread(SOURCE_PATH_1, cv2.IMREAD_GRAYSCALE)
mask = cv2.dilate(mask, kernel_3x3, iterations=4)
mask = cv2.erode(mask, kernel_3x3, iterations=10)
mask = (255-mask)
img = cv2.imread(SOURCE_PATH_1, cv2.IMREAD_GRAYSCALE)
img = cv2.bitwise_and(img, mask)
cv2.imwrite(DESTINATION_PATH, img)
