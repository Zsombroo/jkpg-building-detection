import argparse
import configparser
import cv2
import numpy as np
import os


class OrthophotoPreprocessingPipeline(object):

    def __init__(self, args):
        self.args = args
        self.config = configparser.ConfigParser()
        self.config.read('preprocessing.config')

    def run(self):
        for root, _, files in os.walk(self.args.source_path):
            for name in files:
                image = cv2.imread(os.path.join(root, name))
                
                if self.args.n:
                    image = self._normalize(image)
                if self.args.e:
                    edges = self._edge_detection(image)
                if self.args.r:
                    denoised = self._denoise_edges(edges)
                if self.args.c:
                    if not self.args.n:
                        combined = self._combine_raw_with_edge(image, edges)
                    else:
                        combined = self._combine_raw_with_edge(image, denoised)
                
                if self.args.e and not self.args.n and not self.args.c:
                    cv2.imwrite(os.path.join(self.args.destination_path, name),
                                edges)
                if self.args.e and self.args.n and not self.args.c:
                    cv2.imwrite(os.path.join(self.args.destination_path, name),
                                denoised)
                if self.args.e and self.args.n and self.args.c:
                    cv2.imwrite(os.path.join(self.args.destination_path, name),
                                combined)

    def _normalize(self, image):
        return image/255

    def _edge_detection(self, image):
        canny_edge_config = self.config['CANNY_EDGE_DETECTION']
        low_threshold = int(canny_edge_config['low_threshold'])
        high_threshold = int(canny_edge_config['high_threshold'])
        out = cv2.Canny(image, low_threshold, high_threshold)
        return out
    
    def _denoise_edges(self, image):
        noise_remover_config = self.config['NOISE_REMOVER']
        dilate_iterations = int(noise_remover_config['dilate_iterations'])
        erode_iterations = int(noise_remover_config['erode_iterations'])
        kernel_3x3 = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(image, kernel_3x3, iterations=dilate_iterations)
        mask = cv2.erode(mask, kernel_3x3, iterations=erode_iterations)
        mask = (255-mask)
        out = cv2.bitwise_and(image, mask)
        return out
    
    def _combine_raw_with_edge(self, raw, edge):
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        out = cv2.add(raw, edge)
        return out


if __name__=='__main__':

    argument_parser = argparse.ArgumentParser(
        description='Run images through the orthophoto preprocessing pipeline.')

    argument_parser.add_argument('-SP', '--source_path', 
        help='Path to the folder that contains the input images.')
    argument_parser.add_argument('-DP', '--destination_path',
        help='Path to the folder that recieves the processed images.')
    argument_parser.add_argument('-CF', '--config_file',
        default='preprocessing.config',
        help='Path of the config file to use in the pipeline.')
    argument_parser.add_argument('-n', action='store_true', 
        help='Run normalization on the images.')
    argument_parser.add_argument('-e', action='store_true', 
        help='Run Canny edge detection on the images.')
    argument_parser.add_argument('-r', action='store_true', 
        help='Remove noise after edge detection.')
    argument_parser.add_argument('-c', action='store_true', 
        help='Combine raw and edge images.')
    input_arguments = argument_parser.parse_args()

    assert input_arguments.source_path, \
        '--source_path needs to be provided.'
    assert input_arguments.destination_path, \
        '--destination_path needs to be provided.'
    assert os.path.exists(input_arguments.source_path), \
        'The given source folder does not exist, aborting.'

    OrthophotoPreprocessingPipeline(input_arguments).run()
