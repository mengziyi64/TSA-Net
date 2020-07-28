import os
import tensorflow as tf

class Basement_TFModel(object):
    '''Define and Initialize the basic/necessary element of a tensorflow model '''
    def __init__(self, sess, config, learning_rate, is_training):

        # Initialization of General value for model building
        # For all the other component (Hyperparameter for model structure), Please refer to the function "initial_parameter"
        self.sess = sess
        self.config = config
        self.is_training = is_training

        # Model training SetUp
        self.learning_rate = learning_rate
        self.max_grad_norm = float(config.get('max_grad_norm', 5.0))
        
        # Loss function & Metric
        self.loss = None
        self.loss_func = config.get('loss_func','RMSE')

    def set_lr(self,new_learning_rate):
        self.learning_rate = new_learning_rate