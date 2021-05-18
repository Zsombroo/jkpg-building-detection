import configparser
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    slicer_condig = config['IMAGE_HISTOGRAM']

    source_path = slicer_condig['source_path']
    destination_path = slicer_condig['destination_path_without_param']
                                 
    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)

    for image_file in tqdm(sorted(os.listdir(source_path))):
        img_color = cv2.imread('{}/{}'.format(source_path, image_file))
        img_gray = cv2.imread('{}/{}'.format(source_path, image_file), 0)

        plt.figure(figsize=(12,4));
        plt.subplot(1,2,1);
        plt.imshow(img_color[:,:,::-1]);
        plt.xticks([]); plt.yticks([])

        # Calculate and plot histograms of each color channel (b,g,r)
        color = ('b','g','r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img_color], [i], None, [256], [0,256])
            plt.subplot(1,2,2);
            plt.plot(hist, color = col)
            plt.xlim([0,256])
        
        # Calculate and plot histogram for the gray scale image
        hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0,256])
        plt.subplot(1,2,2);
        plt.plot(hist_gray, color = 'k')
        plt.xlim([0,256])

        plt.tight_layout(pad=3.0)
        plt.savefig('{}/{}'.format(destination_path, image_file), format='png')