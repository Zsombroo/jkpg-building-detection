import configparser
import cv2
import os
from tqdm import tqdm


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    slicer_condig = config['COMBINE_RAW_WITH_EDGES']

    source_path_raw = slicer_condig['source_path_raw']
    source_path_edge = slicer_condig['source_path_edge']
    destination_path = slicer_condig['destination_path_without_param']

    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)

    for image_file in tqdm(sorted(os.listdir(source_path_raw))):
        raw = cv2.imread('{}/{}'.format(source_path_raw, image_file),
                         cv2.IMREAD_ANYCOLOR)
        edge = cv2.imread('{}/{}'.format(source_path_edge, image_file),
                         cv2.IMREAD_ANYCOLOR)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        combined = cv2.add(raw, edge)
        cv2.imwrite('{}/{}'.format(destination_path, image_file), combined)
