#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
import six
import collections
from vgg import vgg_16


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_boolean('is_train', True, "train or predic")
flags.DEFINE_string('image_name', 'fcn_val', 'predict image name')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}

def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': _int64_list_feature(height),
        'image/width': _int64_list_feature(width),
        'image/filename': _bytes_list_feature(os.path.basename(data)[:-4]),
        'image/encoded': _bytes_list_feature(encoded_data),
        'image/label': _bytes_list_feature(encoded_label),
        'image/format': _bytes_list_feature(_IMAGE_FORMAT_MAP[os.path.basename(data)[-3:]]),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example



def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
      return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def create_tf_record(output_filename, file_pars):
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for (data, label) in file_pars:
            print(data)
            example = dict_to_tf_example(data, label)
            if not example is None:
                tfrecord_writer.write(example.SerializeToString())
        
    


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    if FLAGS.is_train:
        train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
        train_files = read_images_names(FLAGS.data_dir, True)
        create_tf_record(train_output_path, train_files)
    
    val_output_path = os.path.join(FLAGS.output_dir, FLAGS.image_name + '.record')
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
