from __future__ import absolute_import

import tensorflow as tf
import yaml
import os
import h5py

from Model.Decoder_Handler import Decoder_Handler

config_filename = './Model/Config.yaml'

def main():
    ave_folder, ave_config = 'TSA-Model-Real','config_107.yaml'
    
    folder_id, config_id = ave_folder,ave_config
    with open(config_filename) as handle:
        model_config = yaml.load(handle,Loader=yaml.FullLoader)  
    data_dir = []
    data_dir.append(model_config['category_train'])
    data_dir.append(model_config['category_valid'])
    data_dir.append(model_config['category_test'])
    
    mask_dir = model_config['category_mask']
    
    log_dir = os.path.join(os.path.abspath('.'),model_config['result_dir'],model_config['result_model'], folder_id)

    with open(os.path.join(log_dir, config_id)) as handle:
        model_config = yaml.load(handle,Loader=yaml.FullLoader)
    
    dataset_dir = (data_dir,mask_dir)
    
    tf_config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        Cube_Decoder = Decoder_Handler(dataset_dir=dataset_dir, model_config=model_config, sess = sess, is_training=False)
        Cube_Decoder.test()

if __name__ == '__main__':
    main()
    
