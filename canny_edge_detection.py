import cv2
import os


if __name__=='__main__':
    SOURCE_PATH = '../sliced_raw'
    DESTINATION_PATH = '../canny_edges'
    LOW_THRESHOLD, HIGH_THRESHOLD = 250, 255

    for image_file in sorted(os.listdir(SOURCE_PATH)):
        img = cv2.imread('{}/{}'.format(SOURCE_PATH, image_file),
                         cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, LOW_THRESHOLD, HIGH_THRESHOLD)
        cv2.imwrite('{}/{}_{}_{}.png'.format(DESTINATION_PATH, image_file[:-4],
                    LOW_THRESHOLD,
                    HIGH_THRESHOLD),
                    edges)
