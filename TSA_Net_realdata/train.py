from __future__ import absolute_import

import tensorflow as tf
import yaml
import os

from Model.Decoder_Handler import Decoder_Handler


config_filename = './Model/Config.yaml'

def main():
    with open(config_filename) as handle:
        model_config = yaml.load(handle,Loader=yaml.FullLoader)
    data_dir = []
    data_dir.append(model_config['category_train'])
    data_dir.append(model_config['category_valid'])
    data_dir.append(model_config['category_test'])
    mask_dir = model_config['category_mask']
        
    dataset_dir = (data_dir,mask_dir)
    
    tf_config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        Cube_Decoder = Decoder_Handler(dataset_dir=dataset_dir, model_config=model_config, sess = sess, is_training=True)
        Cube_Decoder.train()

if __name__ == '__main__':
    main()

