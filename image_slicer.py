import configparser
import cv2
import numpy as np
import os
import re


def slice_image(
    image: np.ndarray,
    image_prefix: str, 
    slice_size: int,
    offset_h_w: tuple = (0, 0)
    ) -> dict:

    out = dict()
    h, w = image.shape[:2]
    for slice_y in range(int((h-offset_h_w[0])/slice_size)):
        for slice_x in range(int((w-offset_h_w[1])/slice_size)):
            out['{}_{}_{}'.format(image_prefix, slice_y, slice_x)] = \
                image[offset_h_w[0]+slice_y*slice_size:
                      offset_h_w[0]+slice_y*slice_size+slice_size,
                      offset_h_w[1]+slice_x*slice_size:
                      offset_h_w[1]+slice_x*slice_size+slice_size]
    return out
    

if __name__=='__main__':
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    slicer_condig = config['IMAGE_SLICER']

    source_path = slicer_condig['source_path']
    destination_path = slicer_condig['destination_path_without_size']
    slice_size = int(slicer_condig['size'])
    destination_path = '_'.join((destination_path, str(slice_size)))

    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)

    pattern = re.compile('.*\.tif$')
    for file in sorted(os.listdir(source_path)):
        if pattern.match(file):
            slices = slice_image(cv2.imread('{}/{}'.format(source_path, file)),
                                 file[:-4],
                                 slice_size)
            for k, v in slices.items():
                cv2.imwrite('{}/{}.png'.format(destination_path, k), v)