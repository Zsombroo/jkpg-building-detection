import cv2
import os
import matplotlib.pyplot as plt


if __name__=='__main__':
    SOURCE_PATH = '../sliced_raw'
    DESTINATION_PATH = '../hist'

    for image_file in sorted(os.listdir(SOURCE_PATH))[869:]:
        img_color = cv2.imread('{}/{}'.format(SOURCE_PATH, image_file))

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

        plt.tight_layout(pad=3.0)
        plt.savefig('{}/{}'.format(DESTINATION_PATH, image_file), format='png')