import cv2
import os


if __name__=='__main__':
    SOURCE_PATH = '../sliced_raw'
    DESTINATION_PATH = '../canny_edges/low_th_170'

    for image_file in sorted(os.listdir(SOURCE_PATH)):
        img = cv2.imread('{}/{}'.format(SOURCE_PATH, image_file),
                         cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 170, 256)
        cv2.imwrite('{}/{}'.format(DESTINATION_PATH, image_file), edges)
