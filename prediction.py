#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import time
from PIL import Image

import cv2
import numpy as np
import tensorflow as tf

import pydensecrf.densecrf as dcrf
import vgg
from dataset import (input_tensor, lable_input)
from pydensecrf.utils import (create_pairwise_bilateral,
                              create_pairwise_gaussian, unary_from_softmax)
from utils import (bilinear_upsample_weights, grayscale_to_voc_impl)

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

# 命令行解析参数
def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='/content/fcn/output/train')
    parser.add_argument('--output_image_dir', type=str, default='../server/static/result')
    parser.add_argument('--image', type=str)
    parser.add_argument('--number_of_classes', type=int, default=21)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


FLAGS, unparsed = parse_args()



class FcnModel(object):

    def __init__(self, ckpt, number_of_classes = 21, upsample_factor = 16):
        self.ckpt = ckpt
        self.number_of_classes = number_of_classes
        self.upsample_factor = upsample_factor
        self.sess, self.pred, self.orig_img_tensor, self.probabilities = self.init_model()


    def init_model(self):
        slim = tf.contrib.slim

        # 初始化
        tf.reset_default_graph()

        # 输入图片
        image_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='image_tensor')  
        orig_img_tensor = tf.placeholder(tf.uint8, shape=(1, None, None, 3), name='orig_img_tensor')  


        # 初始化模型
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, end_points = vgg.vgg_16(image_tensor,
                                            num_classes=self.number_of_classes,
                                            is_training=False,
                                            spatial_squeeze=False,
                                            fc_conv_padding='SAME')

        downsampled_logits_shape = tf.shape(logits)

        img_shape = tf.shape(image_tensor)

        # Calculate the ouput size of the upsampled tensor
        # The shape should be batch_size X width X height X num_classes
        upsampled_logits_shape = tf.stack([
                                          downsampled_logits_shape[0],
                                          img_shape[1],
                                          img_shape[2],
                                          downsampled_logits_shape[3]
                                          ])


        pool4_feature = end_points['vgg_16/pool4']
        with tf.variable_scope('vgg_16/fc8'):
            aux_logits_16s = slim.conv2d(pool4_feature, self.number_of_classes, [1, 1],
                                         activation_fn=None,
                                         weights_initializer=tf.zeros_initializer,
                                         scope='conv_pool4')

        # Perform the upsampling
        upsample_filter_np_x2 = bilinear_upsample_weights(2,  # upsample_factor,
                                                          self.number_of_classes)

        upsample_filter_tensor_x2 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2')

        upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x2,
                                                  output_shape=tf.shape(aux_logits_16s),
                                                  strides=[1, 2, 2, 1],
                                                  padding='SAME')


        upsampled_logits = upsampled_logits + aux_logits_16s

        upsample_filter_np_x16 = bilinear_upsample_weights(self.upsample_factor,
                                                           self.number_of_classes)

        upsample_filter_tensor_x16 = tf.Variable(upsample_filter_np_x16, name='vgg_16/fc8/t_conv_x16')
        upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x16,
                                                  output_shape=upsampled_logits_shape,
                                                  strides=[1, self.upsample_factor, self.upsample_factor, 1],
                                                  padding='SAME')


        # Tensor to get the final prediction for each pixel -- pay
        # attention that we don't need softmax in this case because
        # we only need the final decision. If we also need the respective
        # probabilities we will have to apply softmax.
        pred = tf.argmax(upsampled_logits, axis=3, name='predictions')

        probabilities = tf.nn.softmax(upsampled_logits, name='probabilities')


        # 恢复模型，从训练好的模型中恢复参数
        checkpoint_path = tf.train.latest_checkpoint(self.ckpt)
        assert checkpoint_path, "no checkpoint exist, cant perform predict."
        variables_to_restore = slim.get_model_variables()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()

        saver = tf.train.Saver(max_to_keep=1)
        # Run the initializers.
        sess.run(init_op)
        sess.run(init_local_op)
        saver.restore(sess, checkpoint_path)
        logging.debug('checkpoint restored from [{0}]'.format(checkpoint_path))

        return sess, pred, orig_img_tensor, probabilities


    def perform_crf(self, image, probabilities):
        image = image.squeeze()
        softmax = probabilities.squeeze().transpose((2, 0, 1))
    
        # The input should be the negative of the logarithm of probability values
        # Look up the definition of the softmax_to_unary for more information
        unary = unary_from_softmax(softmax)
    
        # The inputs should be C-continious -- we are using Cython wrapper
        unary = np.ascontiguousarray(unary)
    
        d = dcrf.DenseCRF(image.shape[0] * image.shape[1], self.number_of_classes)
    
        d.setUnaryEnergy(unary)
    
        # This potential penalizes small pieces of segmentation that are
        # spatially isolated -- enforces more spatially consistent segmentations
        feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])
    
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    
        # This creates the color-dependent features --
        # because the segmentation that we get from CNN are too coarse
        # and we can use local color features to refine them
        feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                          img=image, chdim=2)
    
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)
    
        res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
        return res


    def predict(self, image_path):  
        image_name = os.path.basename(image_path)
        logging.debug("predicting %s..."%image_name)
        img, orig_img = input_tensor(image_path)

        with tf.Session() as sess2:
            img = sess2.run(img)
            orig_img = sess2.run(orig_img)

        image_feed_dict = {'image_tensor:0': img, 'orig_img_tensor:0': orig_img}
        val_pred, val_orig_image, val_poss = self.sess.run([self.pred, self.orig_img_tensor, self.probabilities], feed_dict=image_feed_dict)  
        crf_ed = self.perform_crf(val_orig_image, val_poss)
        
        return val_orig_image, val_pred, crf_ed
    
    def eval(self, image_path, lable_path):
        image_name = os.path.basename(image_path)
        logging.debug("evaling %s..."%image_name)
        img, orig_img = input_tensor(image_path)
        lable = lable_input(lable_path)

        with tf.Session() as sess2:
            img, orig_img = sess2.run([img, orig_img])
            lable = sess2.run(lable)

        image_feed_dict = {'image_tensor:0': img, 'orig_img_tensor:0': orig_img}
        val_pred, val_orig_image, val_poss = self.sess.run([self.pred, self.orig_img_tensor, self.probabilities], feed_dict=image_feed_dict)  
        crf_ed = self.perform_crf(val_orig_image, val_poss)
        
        lable = np.reshape(lable, [lable.shape[1], lable.shape[2]])
        crfed_tensor = tf.constant(crf_ed)
        lable = tf.constant(lable)
        miou, mat = tf.metrics.mean_iou(lable, crfed_tensor, 21)

        with tf.Session() as sess3:
            sess3.run(tf.local_variables_initializer())
            mat = sess3.run(mat)
            miou = sess3.run(miou)

        
        return miou

def save_predict_img(image_name, orig_img, pred_img, crfed_img, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
    orig_path = os.path.join(output_dir, '{0}_orig.jpg'.format(image_name))
    pred_path = os.path.join(output_dir, '{0}_pred.jpg'.format(image_name))
    crfed_path = os.path.join(output_dir, '{0}_pred_crfed.jpg'.format(image_name))
    overlay_path = os.path.join(output_dir, '{0}_overlay.jpg'.format(image_name))

    cv2.imwrite(orig_path, cv2.cvtColor(np.squeeze(orig_img), cv2.COLOR_RGB2BGR))
    cv2.imwrite(pred_path, cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(pred_img)), cv2.COLOR_RGB2BGR))  
    cv2.imwrite(crfed_path, cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crfed_img)), cv2.COLOR_RGB2BGR))  
    overlay = cv2.addWeighted(cv2.cvtColor(np.squeeze(orig_img), cv2.COLOR_RGB2BGR), 1, cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crfed_img)), cv2.COLOR_RGB2BGR), 0.8, 0)
    cv2.imwrite(overlay_path, overlay)


def main():
    model = FcnModel(FLAGS.ckpt, FLAGS.number_of_classes)
    val_orig_image, val_pred, crf_ed = model.predict(FLAGS.image)
    save_predict_img(os.path.basename(FLAGS.image)[:-4], val_orig_image, val_pred, crf_ed,  FLAGS.output_image_dir)

if __name__ == '__main__':
    main()
