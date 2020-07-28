import tensorflow as tf
import numpy as np
import yaml
import os
import time
import sys
import math
import scipy.io as sio

from Lib.Data_Processing import *
from Lib.Utility import *
from Model.Decoder_Model import Depth_Decoder
from Model.Base_Handler import Basement_Handler


class Decoder_Handler(Basement_Handler):
    def __init__(self, dataset_dir, model_config, sess, is_training=True):
        
        # Initialization of Configuration, Parameter and Datasets
        super(Decoder_Handler, self).__init__(sess=sess, model_config=model_config, is_training=is_training)
        self.initial_parameter()
        self.data_assignment(dataset_dir,is_training)

        # Data Generator
        
        self.gen_train = Data_Generator_File(self.data_dir, self.training_list, self.sense_mask, self.batch_size, is_training=True, is_valid=False, is_testing=False)
        self.gen_valid = Data_Generator_File(self.data_dir, self.valid_list, self.sense_mask, self.batch_size, is_training=False, is_valid=True, is_testing=False)
        self.gen_test = Data_Generator_File(self.data_dir, self.testing_list, self.sense_mask, self.batch_size, is_training=False, is_valid=False, is_testing=True)
        
        # Define the general model and the corresponding input
        
        shape_meas = (self.batch_size, self.H, self.W ,self.nC)      # PhiTy
        shape_mask = (self.H, self.W, self.nC)
        shape_truth = (self.batch_size, self.H, self.W ,self.nC)
        self.meas_sample = tf.placeholder(tf.float32, shape=shape_meas, name='input_meas')
        self.sense_matrix = tf.placeholder(tf.float32, shape=shape_mask, name='input_mask')
        self.truth = tf.placeholder(tf.float32, shape=shape_truth, name='output_truth')
        
        # Initialization for the model training procedure.
        self.learning_rate = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(self.lr_init),trainable=False)
        self.lr_new = tf.placeholder(tf.float32, shape=(), name='lr_new')
        self.lr_update = tf.assign(self.learning_rate, self.lr_new, name='lr_update')
        self.train_test_valid_assignment()
        self.trainable_parameter_info()
        self.saver = tf.train.Saver(tf.global_variables())

    def initial_parameter(self):
        # Configuration Set
        config = self.model_config
        
        # Model Input Initialization
        self.batch_size = int(config.get('batch_size',1))
        self.W = int(config.get('num_width', 660))
        self.H = int(config.get('num_height', 660))
        self.nC = int(config.get('num_channel', 24))
        
        # Initialization for Training Controler
        self.epochs = int(config.get('epochs',100))
        self.lr_init = float(config.get('learning_rate',0.001))
        self.lr_decay_coe = float(config.get('lr_decay',0.1))
        self.lr_decay_epoch = int(config.get('lr_decay_epoch',20))
        self.lr_decay_interval = int(config.get('lr_decay_interval',10))

    def data_assignment(self, dataset_dir, is_training=True):
        # Division for train, test and validation
        model_config = self.model_config
        self.data_dir, self.training_list, self.valid_list, self.testing_list, self.sense_mask = Data_Division(dataset_dir)
        
        # The value of the position is normalized (the value of lat and lon are all limited in range(0,1))
        disp_train = len(self.training_list)
        disp_valid = len(self.valid_list)
        disp_test = len(self.testing_list)

        self.train_size = int(np.ceil(float(disp_train)/self.batch_size))
        self.test_size  = int(np.ceil(float(disp_test)/self.batch_size))
        self.valid_size = int(np.ceil(float(disp_valid)/self.batch_size))
        
        # Display the data structure of Training/Testing/Validation Dataset
        print('Available samples (batch) train %d(%d), valid %d(%d), test %d(%d)' % (
            disp_train,self.train_size,disp_valid,self.valid_size,disp_test,self.test_size))
        
    def train_test_valid_assignment(self):#, is_training = True, reuse = False
        value_set = (self.meas_sample,
                     tf.expand_dims(self.sense_matrix, 0),
                     self.truth)
        
        with tf.name_scope('Train'):
            with tf.variable_scope('Depth_Decoder', reuse=False):
                self.Decoder_train = Depth_Decoder(value_set, self.learning_rate, self.sess, self.model_config, is_training=True)
        with tf.name_scope('Val'):
            with tf.variable_scope('Depth_Decoder', reuse=True):
                self.Decoder_valid = Depth_Decoder(value_set, self.learning_rate, self.sess, self.model_config, is_training=False)
        with tf.name_scope('Test'):
            with tf.variable_scope('Depth_Decoder', reuse=True):
                self.Decoder_test = Depth_Decoder(value_set, self.learning_rate, self.sess, self.model_config, is_training=False)
                
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        print('Training Started')
        # restore pretrained network if needed
        if self.model_config['model_filename'] is not None:
            self.restore()
            print('Pretrained Model Downloaded')
            
        epoch_cnt, min_val_loss, max_val_psnr = 0, float('inf'), 0
        while epoch_cnt <= self.epochs:
            start_time = time.time()
            cur_lr = self.calculate_scheduled_lr(epoch_cnt)
            train_fetches = {'global_step': tf.train.get_or_create_global_step(), 
                             'train_op': self.Decoder_train.train_op,
                             'metrics': self.Decoder_train.metrics,
                             'pred_orig': self.Decoder_train.decoded_image,
                             'loss': self.Decoder_train.loss}
            valid_fetches = {'global_step': tf.train.get_or_create_global_step(),
                            'pred_orig': self.Decoder_valid.decoded_image,
                             'metrics': self.Decoder_valid.metrics,
                            'loss': self.Decoder_valid.loss}
            Tresults,Vresults = {"loss":[], "psnr":[], "ssim":[]},{"loss":[], "psnr":[], "ssim":[]}
            
            # Training 
            for trained_batch in range(0,self.train_size):
                (measure_train, mask_train, ground_train) = self.gen_train.next()
                feed_dict_train = {self.meas_sample: measure_train, 
                                   self.sense_matrix: mask_train,
                                   self.truth: ground_train}
                train_output = self.sess.run(train_fetches, feed_dict=feed_dict_train)
                Tresults["loss"].append(train_output['loss'])
                Tresults["psnr"].append(train_output['metrics'][0])
                Tresults["ssim"].append(train_output['metrics'][1])
                    
                if trained_batch%100 == 0 and trained_batch != 0:
                    Train_loss = np.mean(np.asarray(Tresults["loss"][-100:]))
                    Train_psnr = np.mean(np.asarray(Tresults["psnr"][-100:]))
                    message = "Train Epoch [%2d/%2d] Batch [%d/%d] lr: %.4f, loss: %.8f psnr: %.4f" % (epoch_cnt, self.epochs, trained_batch, self.train_size, cur_lr, Train_loss, Train_psnr)
                    print(message)

                
            # Validation 
            list_truth,list_pred = [],[]
            for valided_batch in range(0,self.valid_size):
                (measure_valid,mask_valid,ground_valid) = self.gen_valid.next()
                feed_dict_valid = {self.meas_sample: measure_valid,
                                   self.sense_matrix: mask_valid,
                                   self.truth: ground_valid}
                valid_output = self.sess.run(valid_fetches,feed_dict=feed_dict_valid)
                Vresults["loss"].append(valid_output['loss'])
                Vresults["psnr"].append(valid_output['metrics'][0])
                Vresults["ssim"].append(valid_output['metrics'][1])

                list_truth.append(ground_valid)
                list_pred.append(valid_output['pred_orig'])
            Vpsnr = np.mean(Vresults["psnr"])

            # Summary
            Tloss, Vloss = np.mean(Tresults["loss"]), np.mean(Vresults["loss"])
            train_psnr, valid_psnr = np.mean(Tresults["psnr"]), np.mean(Vresults["psnr"])
            train_ssim, valid_ssim = np.mean(Tresults["ssim"]), np.mean(Vresults["ssim"])
            summary_format = ['loss/train_loss','loss/valid_loss','metric/train_psnr','metric/train_ssim',
                              'metric/valid_psnr','metric/valid_ssim']
            summary_data = [Tloss, Vloss, train_psnr, train_ssim, valid_psnr, valid_ssim]
            self.summary_logging(train_output['global_step'], summary_format, summary_data)
            end_time = time.time()
            message = 'Epoch [%3d/%3d] Train(Valid) loss: %.4f(%.4f), psnr: %2f(%2f)' % (
                epoch_cnt, self.epochs, Tloss, Vloss, train_psnr, valid_psnr)
            self.logger.info(message)

            if Vloss <= min_val_loss:
                model_filename = self.save_model(self.saver, epoch_cnt, Vloss)
                self.logger.info('Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss,Vloss, model_filename))
                min_val_loss = Vloss
            
            epoch_cnt += 1
            sys.stdout.flush()

    def test(self):
        
        print("Testing Started")
        self.restore()
        start_time = time.time()
        test_fetches = {'global_step': tf.train.get_or_create_global_step(),
                        'pred_orig':   self.Decoder_test.decoded_image,
                        'metrics':     self.Decoder_test.metrics,
                        'loss':        self.Decoder_test.loss}
        
        for tested_batch in range(self.test_size):
            (measure_test,mask_train,ground_test) = self.gen_test.next()
            feed_dict_test = {self.meas_sample: measure_test,self.sense_matrix: mask_train,self.truth: ground_test}
            test_output = self.sess.run(test_fetches,feed_dict=feed_dict_test)
        ## save recon
        for i in range(len(self.testing_list)):
            sio.savemat('./'+self.log_dir+'/Recon_'+self.testing_list[i], {'recon': np.float32(np.squeeze(test_output['pred_orig'][i,:,:,:]))} )
        
    def calculate_scheduled_lr(self, epoch, min_lr=1e-8):
        decay_factor = int(math.ceil((epoch - self.lr_decay_epoch) / float(self.lr_decay_interval)))
        new_lr = self.lr_init * (self.lr_decay_coe ** max(0, decay_factor))
        new_lr = max(min_lr, new_lr)
        
        self.logger.info('Current learning rate to: %.6f' % new_lr)
        sys.stdout.flush()
        
        self.sess.run(self.lr_update, feed_dict={self.lr_new: new_lr})
        self.Decoder_train.set_lr(self.learning_rate) 
        return new_lr
