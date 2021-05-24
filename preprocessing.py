import argparse
import cv2
import os


class OrthophotoPreprocessingPipeline(object):

    def __init__(self, args):
        self.args = args

    def run(self):
        for root, _, files in os.walk(self.args.source_path):
            for name in files:
                image = cv2.imread(os.path.join(root, name))
                if self.args.e:
                    pass
                if self.args.n:
                    pass
                if self.args.c:
                    pass
                cv2.imwrite(os.path.join(self.args.destination_path, name), \
                    image)


if __name__=='__main__':

    argument_parser = argparse.ArgumentParser(
        description='Run images through the orthophoto preprocessing pipeline.')

    argument_parser.add_argument('-SP', '--source_path', 
        help='Path to the folder that contains the input images.')
    argument_parser.add_argument('-DP', '--destination_path', 
        help='Path to the folder that recieves the processed images.')
    argument_parser.add_argument('-e', action='store_true', 
        help='Run Canny edge detection on the images.')
    argument_parser.add_argument('-n', action='store_true', 
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
