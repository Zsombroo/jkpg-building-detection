import configparser
import cv2
import os
from tqdm import tqdm


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    slicer_config = config['GRAY_SCALE']

    source_path_raw = slicer_config['source_path_raw']
    destination_path = slicer_config['destination_path_without_param']

    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)

    for image_file in tqdm(sorted(os.listdir(source_path_raw))):
        gray = cv2.imread('{}/{}'.format(source_path_raw, image_file),
                         cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('{}/{}'.format(destination_path, image_file), gray)
