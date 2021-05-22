import cv2
import numpy as np
import os
from tqdm import tqdm
import configparser


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    noise_remover_config = config['NOISE_REMOVER']

    source_path = noise_remover_config['source_path']
    destination_path = noise_remover_config['destination_path_without_param']
    dilate_iterations = int(noise_remover_config['dilate_iterations'])
    erode_iterations = int(noise_remover_config['erode_iterations'])
    destination_path = '_'.join((destination_path,
                                 str(dilate_iterations),
                                 str(erode_iterations)))

    kernel_3x3 = np.ones((3, 3), np.uint8)

    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)

    for image_file in tqdm(sorted(os.listdir(source_path))):
        mask = cv2.imread('{}/{}'.format(source_path, image_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.dilate(mask, kernel_3x3, iterations=4)
        mask = cv2.erode(mask, kernel_3x3, iterations=10)
        mask = (255-mask)
        img = cv2.imread('{}/{}'.format(source_path, image_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_and(img, mask)
        cv2.imwrite('{}/{}'.format(destination_path, image_file), img)
