#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
num_keep_radio = 0.8
#define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    #label=-1 --> label=0 net_factory
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)

    num_cls_prob = tf.size(cls_prob)

    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)

    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))


    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    valid_inds = tf.where(label < zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #set 0 to invalid sample
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)

def bbox_ohem(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    #(batch,)
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1) #�������
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def landmark_ohem(landmark_pred,landmark_target,label):
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)

    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
    
def cal_accuracy(cls_prob,label):

    # [0.2,0.8]  label=1
    pred = tf.argmax(cls_prob,axis=1)  # pred=1
    label_int = tf.cast(label,tf.int64)# label=1
    cond = tf.where(tf.greater_equal(label_int,0)) #cond=0

    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op
#construct Pnet
def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    #define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005), 
                        padding='valid'):
       
        net = slim.conv2d(inputs, num_outputs=10, kernel_size=[3,3], stride=1,scope='conv1')
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1')
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')

        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
   
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
 
        landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)

        if training:
            
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            cls_loss = cls_ohem(cls_prob,label)
            
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            
            landmark_pred = tf.squeeze(landmark_pred,[1,2],name="landmark_pred")
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)

            accuracy = cal_accuracy(cls_prob,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy 
        
        else:
            #batch_size = 1 so axios=0
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
            return cls_pro_test,bbox_pred_test,landmark_pred_test
        
def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
        
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1") 
        net = slim.max_pool2d(net, kernel_size=[2,2],stride=2,scope="pool1")
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1")
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
    
def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):
     
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
     
        net = slim.max_pool2d(net, kernel_size=[2,2],stride=2,scope="pool1")
        
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
      
        net = slim.max_pool2d(net,kernel_size=[2,2],stride=2,scope="pool2")
       
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
      
        net = slim.max_pool2d(net,kernel_size=[2,2],stride=2,scope="pool3")
     
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
                     
        fc_flatten = slim.flatten(net)
       
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
   
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
      
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
     
        landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)    
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            return cls_prob,bbox_pred,landmark_pred
            
        
        
                                                                  
