"""

Creates tf records from image and xml-input.
Input:  input_folder (both images and xml-files), output_folder, output_filename
Output: .record-files

input-syntax example: --input_folder "../serialize_raw_640" 
    --output_folder "../serialized" 
    --output_filename train_raw_640.record

Code based on:
https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model
https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from os import path
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
import configparser
import io

from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


def input_sanity_check(inputs, tfrecords_config):
    """
    Sanity check on the input, checks if given folders and target file exist.
    Returns the given input.
    """

    inputs.add_argument('-if', '--input_folder', 
        help='Path to folder that contains images and xml-files.')
    inputs.add_argument('-of', '--output_folder', 
        help='Path to folder where .record-file will be output to.')
    inputs.add_argument('-on', '--output_filename', 
        help='Filename for the output .record-file (e.g., train_raw_640.record).')
    inputs.add_argument('-cn', '--classnames', 
        help='Classnames used for the tfrecords, list in configparser.')
    input_arguments = inputs.parse_args()

    assert input_arguments.input_folder, '--input_folder needs to be provided.'
    assert input_arguments.output_folder, '--output_folder needs to be provided.'
    assert input_arguments.output_filename, '--output_filename needs to be provided.'
    assert input_arguments.classnames, '--classnames needs to be provided.'
    assert (path.exists(input_arguments.input_folder) 
        and path.exists(input_arguments.output_folder)
        ), 'The given folders does not exist, aborting.'
    assert not path.exists(input_arguments.output_folder 
        + '/' 
        + input_arguments.output_filename
        ), 'The given output file already exist, aborting.'
    assert tfrecords_config[input_arguments.classnames], 'The given classname identifier does not exist, aborting.'

    return input_arguments.input_folder, \
        input_arguments.output_folder, \
        input_arguments.output_filename, \
        tfrecords_config[input_arguments.classnames]

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def class_text_to_int(row_label, classnames):
    for idx, classname in enumerate(classnames.split(',')):
        if row_label == classname:
            return idx+1
    
    # If no value match is found, return None
    return None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, classnames):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], classnames))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


if __name__ == '__main__':
    # Fetch list of classes from configuration
    config = configparser.ConfigParser()
    config.read('preprocessing.config')
    tfrecords_config = config['TF_RECORDS_CLASSES']
    
    # Check input
    inputs = argparse.ArgumentParser()
    input_folder, output_folder, output_filename, classnames = input_sanity_check(inputs, tfrecords_config)
    
    # Create csv-file from input data
    xml_df = xml_to_csv(input_folder)
    csv_filename = output_folder+'/'+output_filename.split('.')[0]+'_labels.csv'
    xml_df.to_csv((csv_filename), index=None)
    if xml_df.size == 0:
        print('It seems that there were no data in the folder, size 0 is written to csv.')
    else:
        print('Xml converted to csv, size ', xml_df.size)

    # Create tfrecords
    writer = tf.python_io.TFRecordWriter(output_folder+'/'+output_filename)
    path = os.path.join(input_folder)
    examples = pd.read_csv(csv_filename)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, classnames)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created tf records: {}'.format(output_folder+'/'+output_filename))