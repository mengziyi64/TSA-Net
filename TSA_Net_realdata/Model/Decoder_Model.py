import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import os
import json

from Lib.Utility import *
from Model.Base_TFModel import Basement_TFModel

class Depth_Decoder(Basement_TFModel):
    
    def __init__(self, value_sets, init_learning_rate, sess, config, is_training=True, *args, **kwargs):
        
        super(Depth_Decoder, self).__init__(sess=sess, config=config, learning_rate=init_learning_rate,is_training=is_training)
        (measurement, mat_sense, truth) = value_sets
        self.depth = mat_sense.get_shape().as_list()[-1]
        self.batch_size = truth.get_shape().as_list()[0]
                
        self.decoded_image = self.encdec_handler(measurement, mat_sense)
        self.metric_opt(self.decoded_image, truth)
        
    def encdec_handler(self, measurement, mat_sense):

        self.hyper_structure = [(3,64,3,3),(3,128,3,2),(3,256,3,2),(3,512,3,2),(3,1024,3,2)]
        self.end_encoder = (3,1280,3)
        
        encoder_in = measurement
        #print(encoder_in.get_shape().as_list())
        output = self.inference(encoder_in,0.8,phase_train = True)#self.is_training
        return output
    
    def inference(self, images, keep_probability,phase_train=True, bottleneck_layer_size=128, weight_decay=0.0005, reuse=None):
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            'scale':True,
            'is_training':phase_train,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],}
        
        with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.conv2d_transpose],
                            weights_initializer=slim.initializers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params):
            return self.encoder_decoder(images, is_training=phase_train,dropout_keep_prob=keep_probability,reuse=reuse)
    
    def Decoder_reshape(self, decoder, dif, module_ind):
        kdepth = decoder.get_shape().as_list()[-1]
        decoder = slim.conv2d(decoder, kdepth, (1+dif,1+dif), padding='VALID', scope='conv_reshape_%d'%(module_ind))
        return decoder

    def EncConv_module(self, net, module_ind, hyper_struc, flag_res=True, PoolValid=True):
        (lnum,knum,ksize,pstr) = hyper_struc
        for layer_cnt in range(1,1+lnum):
            if layer_cnt == 2 and flag_res == True:
                net = self.module_res2net(net,module_ind,4)
            else:
                net = slim.conv2d(net, knum, ksize, stride=1, padding='SAME',scope='en_%d_%d'%(module_ind,layer_cnt))
        self.end_points['encode_%d'%(module_ind)] = net
        #print(net.get_shape().as_list())
        if PoolValid is True:
            return slim.max_pool2d(net,pstr,stride=pstr,padding='SAME',scope='Pool%d'%(module_ind))
        else:
            return net

    def DecConv_module(self, net, module_ind, hyper_struc, flag_res=True, PoolValid=True, dif=None):
        (lnum,knum,ksize,pstr) = hyper_struc
        if PoolValid is True:
            net = slim.conv2d_transpose(net, knum, pstr, pstr, padding='SAME')
        if dif is not None:
            net = self.Decoder_reshape(net,dif,module_ind)
        #print(net.get_shape().as_list())
        net=tf.concat([net,self.end_points['encode_%d'%(module_ind)]],3)
        for layer_cnt in range(1,1+lnum):
            if layer_cnt == 2 and flag_res == True:
                net = self.module_res2net(net,module_ind,4,scope='Dec')
            else:
                net = slim.conv2d(net, knum, ksize, stride=1, padding='SAME',scope='de_%d_%d'%(module_ind,layer_cnt))
        return net
            
    def encoder_decoder(self, inputs, is_training=True, dropout_keep_prob=0.8, reuse=None, scope='generator'):
        self.end_points = {}
        with tf.variable_scope(scope, 'generator', [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],stride=1, padding='SAME'):
                    ############################# encoder ##############################################
                    net = self.EncConv_module(inputs,1,self.hyper_structure[0],False)
                    net = self.EncConv_module(net,2,self.hyper_structure[1],False)
                    net = self.EncConv_module(net,3,self.hyper_structure[2])
                    net = self.EncConv_module(net,4,self.hyper_structure[3])

                    with tf.variable_scope('share',reuse=tf.AUTO_REUSE): #tf.compat.AUTO_REUSE
                        for recur_ind in range(1):
                            net = self.EncConv_module(net,5,self.hyper_structure[4])
                            (lnum,knum,ksize) = self.end_encoder
                            net = slim.conv2d(net, knum, ksize, stride=1, padding='SAME', scope='en_6')
                            net = slim.conv2d(net, knum, ksize, stride=1, padding='SAME', scope='en_7')
                            net = self.DecConv_module(net, 5, self.hyper_structure[4])

                    ############################# decoder ##############################################
                    net = self.DecConv_module(net,4,self.hyper_structure[3], dif=1)
                    net = self.DecConv_module(net,3,self.hyper_structure[2])
                    dim_lambda,att_unit,num_head = self.hyper_structure[2][1],(64, 64, 80, 3),8
                    net = self.Transform_layer(net,1,dim_lambda,att_unit,num_head,[0,0])
                    net = self.DecConv_module(net,2,self.hyper_structure[1],False)
                    dim_lambda,att_unit,num_head = self.hyper_structure[1][1],(64, 64, 40, 3),8
                    net = self.Transform_layer(net,2,dim_lambda,att_unit,num_head,[1,0])
                    net = self.DecConv_module(net,1,self.hyper_structure[0],False)
                    dim_lambda,att_unit,num_head = 28,(48, 48, 30, 3),4
                    spectral_mask = self.Spectral_Mask(dim_lambda)
                    net = self.Transform_layer(net,3,dim_lambda,att_unit,num_head,[1,1],spectral_mask)
                    net=slim.conv2d(net,self.depth,1,stride=1,padding='SAME',activation_fn=tf.nn.sigmoid, scope='Final_recon')
        return net
    
    def metric_opt(self, model_output, ground_truth):
        
        if self.loss_func == 'MSE':
            self.loss = loss_mse(model_output, ground_truth)
        elif self.loss_func == 'RMSE':
            self.loss = loss_rmse(model_output, ground_truth)+ loss_spec(model_output, ground_truth)
        elif self.loss_func == 'SSIM':
            self.loss = loss_SSIM(model_output, ground_truth)
        else:
            self.loss = loss_rmse(model_output, ground_truth)
            
        self.metrics = calculate_metrics(model_output, ground_truth)
        global_step = tf.train.get_or_create_global_step()
            
        if self.is_training:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
        self.info_merge = tf.summary.merge_all()

    def module_res2net(self, net, module_ind=0, subsets=4,scope='Enc'):
        with tf.variable_scope(scope+'module_res_%d'%(module_ind)):
            (batch_size,height,width,num_feature) = net.get_shape().as_list()
            #print (batch_size,height,width,num_feature)
            size_set = int(num_feature/subsets)
            output = net[:,:,:,:size_set]
            cube = slim.conv2d(net[:, :, :, size_set:2*size_set],size_set,3,stride=1,padding='SAME',scope='res_cube_1')
            output = tf.concat([output,cube],-1)
            for i in range(2,subsets):
                cube = output[:,:,:,-size_set:]+net[:, :, :, i*size_set:(i+1)*size_set]
                cube = slim.conv2d(cube, size_set, 3, stride=1,padding='SAME',scope='res_cube_%d'%(i))
                output = tf.concat([output, cube], -1)
        return output

    def Transform_layer(self, att_input, module_idx, dim_lambda, att_unit, num_heads, mode=[1,1], mask=None, flat_attention=False):
        net = slim.conv2d(att_input, dim_lambda, 3, stride=1, padding='SAME',activation_fn=None, scope='Chan_Dim_%d'%(module_idx))
        net_CDSA = self.CDSA_Module(net, module_idx, dim_lambda, att_unit, num_heads, mode, mask, flat_attention)
        net = slim.layer_norm(net_CDSA+net, scope='CDSA_Norm_%d'%(module_idx))
        #print('CDSA', module_idx, net_CDSA.get_shape().as_list())
        return net

    def CDSA_Module(self, att_input, module_idx, dim_lambda, att_unit, num_heads, mode, mask=None, flag_attention=False):
        (dim_wgt_x, dim_wgt_y, dim_wgt_lambda, dim_value) = att_unit

        Q_x = self.space_fea(att_input,num_heads,dim_wgt_x,'Q_%d_x_'%(module_idx),mode)
        Q_y = self.space_fea(att_input,num_heads,dim_wgt_y,'Q_%d_y_'%(module_idx),mode,'Y')
        K_x = self.space_fea(att_input,num_heads,dim_wgt_x,'K_%d_x_'%(module_idx),mode)
        K_y = self.space_fea(att_input,num_heads,dim_wgt_y,'K_%d_y_'%(module_idx),mode,'Y')
        Q_lambda = self.spectral_fea(att_input,num_heads,dim_wgt_lambda,'Q_%d_lambda'%(module_idx))
        K_lambda = self.spectral_fea(att_input,num_heads,dim_wgt_lambda,'K_%d_lambda'%(module_idx))
        
        Value = slim.conv2d(att_input, num_heads*dim_lambda, 1, stride=1,padding='SAME',scope='Value_Head_' + str(module_idx))

        Q_x = tf.squeeze(tf.concat(tf.split(Q_x, num_heads, axis=3), axis=0),-1)
        Q_y = tf.squeeze(tf.concat(tf.split(Q_y, num_heads, axis=3), axis=0),-1)
        K_x = tf.squeeze(tf.concat(tf.split(K_x, num_heads, axis=3), axis=0),-1)
        K_y = tf.squeeze(tf.concat(tf.split(K_y, num_heads, axis=3), axis=0),-1)
        Q_lambda = tf.concat(tf.split(Q_lambda, num_heads, axis=2), axis=0)
        K_lambda = tf.concat(tf.split(K_lambda, num_heads, axis=2), axis=0)
        V_headbatch = tf.concat(tf.split(Value, num_heads, axis=3), axis=0)

        Q_headbatch = (Q_x,Q_y,Q_lambda)
        K_headbatch = (K_x,K_y,K_lambda)
        AM_x, AM_y, AM_lambda = self.Attention_softmax(Q_headbatch, K_headbatch, att_unit[1:], mask)

        if flag_attention == False:
            out = self.Attention_Value(AM_x, AM_y, AM_lambda, V_headbatch, num_heads, mode, module_idx)
        else:
            out = (AM_x, AM_y, AM_lambda)
        return out

    def space_fea(self, att_in, num_heads, dim_feature, scope_root, mode, direct='X'):
        ker = [(1,5),(3,5)]
        std = [(1,2),(2,2)]
        if direct == 'Y':
            att_in == tf.transpose(att_in,[0,2,1,3])
        feature = slim.conv2d(att_in, num_heads, ker[mode[0]], stride=std[mode[0]], padding='SAME',scope=scope_root+'Conv_1')
        feature = slim.conv2d(feature, num_heads, ker[mode[1]], stride=std[mode[1]], padding='SAME',scope=scope_root+'Conv_2')
        feature = slim.fully_connected(tf.transpose(feature,[0,1,3,2]), dim_feature, scope=scope_root+'FC')
        return tf.transpose(feature,[0,1,3,2])

    def spectral_fea(self, att_in, num_heads, dim_feature, scope_root):
        dim_lambda = att_in.get_shape().as_list()[-1]
        feature = slim.conv2d(att_in, dim_lambda, 5, stride=2, padding='VALID',scope=scope_root+'Conv_1')
        feature = slim.conv2d(feature, dim_lambda, 5, stride=2, padding='VALID',scope=scope_root+'Conv_2')
        feature = tf.reshape(tf.transpose(feature,[0,3,1,2]), [self.batch_size, dim_lambda, -1])
        feature = slim.fully_connected(feature, num_heads*dim_feature, scope=scope_root+'FC')
        return feature

    def Attention_softmax(self,Q_headbatch, K_headbatch, att_unit, mask=None):
        '''mask is applied before the softmax layer, no dropout is applied, '''
        (Q_x,Q_y,Q_lambda) = Q_headbatch
        (K_x,K_y,K_lambda) = K_headbatch
        (space_unit, spec_unit, V_units) = att_unit
        
        # Check the dimension consistency of the combined matrix
        assert Q_x.get_shape().as_list() == K_x.get_shape().as_list()
        assert Q_y.get_shape().as_list() == K_y.get_shape().as_list()
        assert Q_lambda.get_shape().as_list() == K_lambda.get_shape().as_list()
        #print('K_x',K_x.get_shape().as_list(),'K_y',K_y.get_shape().as_list(),'K_lambda',K_lambda.get_shape().as_list())

        AM_x = tf.matmul(Q_x, tf.transpose(K_x, [0, 2, 1])) / tf.sqrt(tf.cast(space_unit, tf.float32))
        AM_y = tf.matmul(Q_y, tf.transpose(K_y, [0, 2, 1])) / tf.sqrt(tf.cast(space_unit, tf.float32))
        AM_lambda = tf.matmul(Q_lambda, tf.transpose(K_lambda, [0, 2, 1])) / tf.sqrt(tf.cast(spec_unit, tf.float32))        

        AM_x = tf.nn.softmax(AM_x, 2)
        AM_y = tf.nn.softmax(AM_y, 2)
        AM_lambda = tf.nn.softmax(AM_lambda, 2)
        #print(AM_x.get_shape().as_list())
        #print(AM_y.get_shape().as_list())

        if mask is not None:
            AM_lambda = (AM_lambda + mask )/ tf.sqrt(tf.cast(1.1, tf.float32))
        return (AM_x, AM_y, AM_lambda)
        

    def Attention_Value(self,AM_x, AM_y, AM_lambda, V_headbatch, num_heads, mode, module_idx):

        [headbatch, dim_x, dim_y, dim_lambda] = V_headbatch.get_shape().as_list()
        scale=1
        for i in range(sum(mode)):
            scale *= 2
            V_headbatch = slim.conv2d(V_headbatch, dim_lambda*scale, 3, stride=2, padding='SAME',scope='Conv_tran_%d_%d'%(i,module_idx))
        #print(V_headbatch.get_shape().as_list())
        shape_x = [headbatch, int(dim_x/scale), int(dim_y/scale), dim_lambda*scale]
        shape_y = [headbatch, int(dim_y/scale), int(dim_x/scale), dim_lambda*scale]
        shape_lambda = [headbatch, dim_lambda, dim_x, dim_y]
        #print(shape_x,shape_y,shape_lambda)

        out = tf.reshape(tf.matmul(AM_x, tf.reshape(V_headbatch,[headbatch, shape_x[1], -1])), shape_x)
        out = tf.transpose(out,perm=[0,2,1,3])
        out = tf.reshape(tf.matmul(AM_y, tf.reshape(out,[headbatch, shape_y[1], -1])), shape_y)
        out = tf.transpose(out,perm=[0,2,1,3])

        if sum(mode) > 0:
            out = slim.conv2d_transpose(out, dim_lambda, scale, stride=scale, padding='SAME')
        #print('Deconv', out.get_shape().as_list())

        out = tf.reshape(tf.matmul(AM_lambda, tf.reshape(tf.transpose(out,perm=[0,3,1,2]),[headbatch, dim_lambda, -1])), shape_lambda)
        out = tf.concat(tf.split(tf.transpose(out,perm=[0,2,3,1]), num_heads, axis=0), axis=3)
        out = slim.fully_connected(out, dim_lambda, activation_fn=None, scope='Head_fuse_%d'%(module_idx))
        return out

    def Spectral_Mask(self,dim_lambda):
        '''After put the available data into the model, we use this mask to avoid outputting the estimation of itself.'''
        orig = (np.cos(np.linspace(-1, 1, num=2*dim_lambda-1)*np.pi)+1.0)/2.0
        att = np.zeros((dim_lambda,dim_lambda))
        for i in range(dim_lambda):
            att[i,:] = orig[dim_lambda-1-i:2*dim_lambda-1-i]
        AM_Mask = tf.expand_dims(tf.convert_to_tensor(att, dtype=tf.float32),0)
        return AM_Mask
    
    def kronecker_product_tf(self, mat1, mat2):
        m1, n1 = mat1.get_shape().as_list()
        mat1_rsh = tf.reshape(mat1, [1, m1, 1, n1, 1])
        bs, m2, n2 = mat2.get_shape().as_list()
        mat2_rsh = tf.reshape(mat2, [bs, 1, m2, 1, n2])
        return tf.reshape(mat1_rsh * mat2_rsh, [bs, m1 * m2, n1 * n2])
    
 