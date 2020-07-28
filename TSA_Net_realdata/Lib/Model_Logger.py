from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging
import os
import numpy as np
import time
import scipy.sparse as sp
import tensorflow as tf

def generate_folder_id(model_config):
    # Introduce the Standard parameter
    model_name = model_config.get('model_name')
    learning_rate_init = float(model_config.get('learning_rate'))
    loss_func = model_config.get('loss_func')
    # Generate the folder name
    folder_id = '%s-T%s-L%.3f-%s/' % (
        model_name, time.strftime('%m%d%H%M%S'),  learning_rate_init, loss_func)
    
    return folder_id

def get_logger(log_dir, name):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    
    return logger
