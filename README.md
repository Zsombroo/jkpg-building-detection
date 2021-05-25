# jkpg-building-detection / Object detection in aerial footage
## Installation of the environment
Sources and inspiration when setting up the environment:
[Gilbert Tanner github](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model)
[Tensorflow 2 Object detection, official github](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

### After following the official Tensorflow Object Detection API installation instructions, these changes were made:
 * Uninstall of duplicate (different versions) of tensorflow (TF). TF 2.4.1 is used.
 * When creating the docker container, a volume is mounted and ports for ssh, tensorboard and juypter notebook is added.
 * pip install opencv-contrib-python-headless
 * pip install scipy
 * Other packages and libaries that is installed but not necessary is openssh-server, nano, net-tools
 

### Code changes needed
Due to the hardware we are running on, the following code i necessary to use the Tensorflow Object Detection API:
´´´bash
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
´´´

## Process of training
 * Crop images to size 640x640
 * Choose images to label. To help the model in not misclassifying we are introducing two "extra" categories. In total the following categories are labelled: building, small_building, car.
 * Apply filter to images (canny edge + denoising filter)
 * Create k folds with 80% training set and 20% test set
 * Serialize images, bounding box labels and labelmap per fold and train/test set
 * Run training of the model on the train set
 * Evaluate the model on the test set

## Inference on novel data
 * Export the model into saved model format
 * Run inference on image(s)
