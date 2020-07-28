import tensorflow as tf
import numpy as np
import yaml
import os
import h5py
import time
import sys
import math

from Lib.Model_Logger import *

class Basement_Handler(object):
    def __init__(self, sess, model_config, is_training):
        
        # Initialization for model configure and training history logging
        self.sess = sess
        self.model_config = model_config
        self.max_grad_norm = float(model_config.get('max_grad_norm', 5.0))
        self.init_logging(is_training)
        self.logger.info(model_config)
        
    def init_logging(self, is_training):
        if is_training is not True:
            folder_dir = self.model_config.get('result_data')
            log_dir = os.path.join(self.model_config.get('result_dir'), folder_dir)
        else:
            base_dir = self.model_config.get('result_model')
            folder_dir = generate_folder_id(self.model_config)
            log_dir = os.path.join(self.model_config.get('result_dir'), base_dir, folder_dir)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_dir = log_dir
        self.logger = get_logger(self.log_dir, folder_dir)
        self.writer = tf.summary.FileWriter(self.log_dir)

    def trainable_parameter_info(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            total_parameters += np.product([x.value for x in variable.get_shape()])

        self.logger.info('Total number of trainable parameters: %d' % total_parameters)
        for var in tf.global_variables():
            self.logger.debug('%s, %s' % (var.name, var.get_shape()))
    
    def summary_logging(self, global_step, names, values):
        for name, value in zip(names, values):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, global_step)

    def save_model(self, saver, epoch, val_loss):
        config_filename = 'config_%02d.yaml' % epoch
        config = dict(self.model_config)
        global_step = self.sess.run(tf.train.get_or_create_global_step())
        config['epoch'] = epoch
        config['log_dir'] = self.log_dir
        config['model_filename'] = saver.save(self.sess, os.path.join(self.log_dir, 'models-%.4f' % val_loss),
                                              global_step=global_step, write_meta_graph=False)
        with open(os.path.join(self.log_dir, config_filename), 'w') as f:
            yaml.dump(config, f)
        return config['model_filename']
    
    def restore(self):
        
        config = dict(self.model_config)
        model_filename = config['model_filename']
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, model_filename)
    