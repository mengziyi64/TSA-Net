import tensorflow as tf
import numpy as np
from Lib.ms_ssim import *
    
def loss_mse(decoded, ground):
    loss_pixel = tf.square(tf.subtract(decoded, ground))
    return tf.reduce_mean(loss_pixel)
    
def loss_rmse(decoded, ground):
    loss_pixel = tf.square(tf.subtract(decoded, ground))
    loss_pixel = tf.sqrt(tf.reduce_mean(loss_pixel,axis=(1,2,3)))
    return tf.reduce_mean(loss_pixel)

def loss_spec(decoded, ground):
    grad_ground = tf.subtract(ground[:,:,:,1:],ground[:,:,:,:-1])
    grad_decode = tf.subtract(decoded[:,:,:,1:],decoded[:,:,:,:-1])
    loss_pixel = tf.reduce_mean(tf.square(tf.subtract(grad_ground, grad_decode)),axis=(1,2,3))
    return tf.reduce_mean(tf.sqrt(loss_pixel)) #tf.reduce_mean(loss_pixel)

def loss_SSIM(decoded,ground):
    return MultiScaleSSIM(decoded,ground)    
    
def tensor_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return  tf.divide(numerator,denominator)

def metric_psnr(decoded, ground):
    loss_pixel = tf.reduce_mean(tf.square(tf.subtract(decoded, ground)),axis=(1,2,3))
    psnr_s = tf.constant(10.0)*tensor_log10(tf.square(tf.reduce_max(ground,axis=(1,2,3)))/loss_pixel)
    return tf.reduce_mean(psnr_s)

def metric_ssim(decoded, ground):
    loss_pixel = tf.abs(tf.subtract(decoded, ground))
    return tf.reduce_mean(loss_pixel)

def calculate_metrics(decoded, ground):
    psnr = metric_psnr(decoded, ground)
    ssim = metric_ssim(decoded, ground)
    return psnr, ssim