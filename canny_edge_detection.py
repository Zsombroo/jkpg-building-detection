import configparser
import cv2
import os


if __name__=='__main__':
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    slicer_condig = config['CANNY_EDGE_DETECTION']

    source_path = slicer_condig['source_path']
    destination_path = slicer_condig['destination_path_without_param']
    low_threshold = int(slicer_condig['low_threshold'])
    high_threshold = int(slicer_condig['high_threshold'])
    destination_path = '_'.join((destination_path,
                                 str(low_threshold),
                                 str(high_threshold)))
    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)
        
    for image_file in sorted(os.listdir(source_path)):
        img = cv2.imread('{}/{}'.format(source_path, image_file),
                         cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, low_threshold, high_threshold)
        cv2.imwrite('{}/{}'.format(destination_path, image_file), edges)
