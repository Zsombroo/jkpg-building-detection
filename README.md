# jkpg-building-detection / Object detection in aerial footage
## Installation of the environment
Sources and inspiration when setting up the environment:
[Gilbert Tanner github](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model)
[Tensorflow 2 Object detection, official github](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

### After following the official Tensorflow Object Detection API installation instructions, these changes were made:
 * Inside the docker container, uninstall of duplicates (different versions) of tensorflow (TF) installations. TF 2.4.1 is used.
 * When creating the docker container, a volume is mounted and ports for ssh, tensorboard and juypter notebook is added to the configuration.
 * Inside the docker container: pip install opencv-contrib-python-headless
 * Inside the docker container: pip install scipy
 * Other packages and libaries that is installed but not necessary is openssh-server, nano, net-tools
 
### Code changes needed
Due to the hardware we are running on, the following code is necessary in the scripts that will use the GPU with Tensorflow Object Detection API:
```
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
```

### Script to create tf-records
Before training of the model, images and ground truth data (xml) is serialized through protobuf. Two scripts from [Gilbert Tanner github](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model) is refactored and merged to make the process more easy. The merged script is found in this repository (create_tfrecords.py).

### Tensorboard
Tensorboard is used to visualize the result from training and evaluating the models created.
```
tensorboard --logdir=<folder where log data is stored>
```

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
