# jkpg-building-detection / Object detection in aerial footage
## Installation of the environment
Inspiration: [Gilbert Tanner github](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model)
Inspiration: [Tensorflow 2 Object detection, official github](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

Changes made from default installation/extra additions

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
