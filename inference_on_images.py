"""
Code is based on code from "Inference from saved model tf2 colab"
(https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_from_saved_model_tf2_colab.ipynb)
and from Gilbert Tanner,
https://colab.research.google.com/github/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model/blob/master/Tensorflow_2_Object_Detection_Train_model.ipynb#scrollTo=5tGVwzpLxvSv

"""

import os
import numpy as np
import glob
from IPython.display import display
from six import BytesIO
from PIL import Image
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse

############ 
# The folloing configuration is to handle known bugs related to GPU hardware.
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
############



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path (this can be local or on colossus)

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
        for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def input_sanity_check(inputs):
    """
    Sanity check on the input.
    Returns the given input.
    """

    inputs.add_argument('-if', '--input_folder', 
        help='Path to folder that contains images.')
    inputs.add_argument('-im', '--input_model', 
        help='Path to model.')
    
    input_arguments = inputs.parse_args()
    assert input_arguments.input_folder, '--input_folder needs to be provided.'
    assert input_arguments.input_model, '--input_model needs to be provided.'

    return input_arguments.input_folder, input_arguments.input_model


if __name__ == '__main__':
    # Check input
    inputs = argparse.ArgumentParser()
    input_folder, input_model = input_sanity_check(inputs)

    if not os.path.isdir(input_folder+'/'+'inference'):
        os.makedirs(input_folder+'/'+'inference')

    labelmap_path = '../models/research/object_detection/training/label_map_3.pbtxt'
    tf.keras.backend.clear_session()
    model = tf.saved_model.load(input_model+'/saved_model')
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

    for image_path in glob.glob(input_folder+'/*.png'):
        print("Inference on ", image_path)
        image_np = load_image_into_numpy_array(image_path)
        output_dict = run_inference_for_single_image(model, image_np)
        
        im_test = image_np.copy()
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=3)

        Image.fromarray(image_np).save(input_folder+'/'+'inference/'+image_path.split("/")[-1])
