import cv2
import numpy as np
import os
import re


def slice_image(
    image: np.ndarray,
    image_prefix: str, 
    slice_size: tuple = (0, 0),
    offset_h_w: tuple = (0, 0)
    ) -> dict:

    out = dict()
    h, w = image.shape[:2]
    for slice_y in range(int((h-offset_h_w[0])/slice_size[0])):
        for slice_x in range(int((w-offset_h_w[1])/slice_size[1])):
            out['{}_{}_{}'.format(image_prefix, slice_y, slice_x)] = \
                image[offset_h_w[0]+slice_y*slice_size[0]:
                      offset_h_w[0]+slice_y*slice_size[0]+slice_size[0],
                      offset_h_w[1]+slice_x*slice_size[1]:
                      offset_h_w[1]+slice_x*slice_size[1]+slice_size[1]]
    return out
    

if __name__=='__main__':
    SOURCE_PATH = '../ortofoto'

    pattern = re.compile('.*\.tif$')
    for slice_one_side in [500, 640, 1024]:
        DESTINATION_PATH = '../sliced_raw_{}_{}'.format(slice_one_side, slice_one_side)
        SLICE_SIZE = (slice_one_side, slice_one_side)
        
        for file in sorted(os.listdir(SOURCE_PATH)):
            if pattern.match(file):
                slices = slice_image(cv2.imread('{}/{}'.format(SOURCE_PATH, file)),
                                    file[:-4],
                                    SLICE_SIZE)
                                    
                for k, v in slices.items():
                    cv2.imwrite('{}/{}.png'.format(DESTINATION_PATH, k), v)