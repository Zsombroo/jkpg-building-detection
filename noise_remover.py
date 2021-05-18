import cv2
import os
import numpy as np


def remove_noise(img, threshold=3, kernel=(3, 3)):
    out = np.copy(img)
    np.copy
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            pixel_value = int(
                sum(
                    img[i-int(kernel[0]/2.0):i+int(kernel[0]/2.0)+1, 
                        j-int(kernel[1]/2.0):j+int(kernel[1]/2.0)+1
                    ].flatten()
                )/255.0)
            out[i, j] = int(pixel_value * 255/9.0)
            continue
            if pixel_value < threshold:
                out[i, j] = 0
            else:
                out[i, j] = 255
    return out


if __name__=='__main__':
    SOURCE_PATH_1 = '../test/tahe1_11_0_0_255.png'
    SOURCE_PATH_2 = '../test/tahe1_11_0_250_255.png'
    DESTINATION_PATH = '../test_out/tahe1_11_0_250_255.png'
    LOOPS = 1
    THRESHOLD = 3
    KERNEL = (5, 5)

    mask = cv2.imread(SOURCE_PATH_2, cv2.IMREAD_GRAYSCALE)
    for i in range(LOOPS):
        mask = remove_noise(mask, THRESHOLD, KERNEL)
    mask = remove_noise(mask, 5, (5, 5))
    img = cv2.imread(SOURCE_PATH_1, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_and(img, img, mask=mask)
    mask = remove_noise(img)
    mask = remove_noise(mask)
    img2 = cv2.imread(SOURCE_PATH_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.bitwise_and(img2, img2, mask=mask)
    img2 = cv2.bitwise_and(cv2.imread(SOURCE_PATH_1, cv2.IMREAD_GRAYSCALE), img2)
    cv2.imwrite(DESTINATION_PATH, img2)
