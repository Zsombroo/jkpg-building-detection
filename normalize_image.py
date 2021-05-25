import configparser
import cv2
import os
from tqdm import tqdm


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    slicer_condig = config['NORMALIZATION']

    source_path = slicer_condig['source_path']
    destination_path = slicer_condig['destination_path_without_param']

    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)

    for image_file in tqdm(sorted(os.listdir(source_path))):
        img = cv2.imread('{}/{}'.format(source_path, image_file),
                         cv2.IMREAD_GRAYSCALE)
        img = img / 255
        cv2.imwrite('{}/{}'.format(destination_path, image_file), img)
