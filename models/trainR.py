#coding:utf-8
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import sys
sys.path.append("../utils")

from utils import config
from mtcnn_model import R_Net
import random
import cv2
def train_model(base_lr, loss, data_num):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    #LR_EPOCH [8,14]
    #boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    #lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    #control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)

    return train_op, lr_op

def train(net_factory, prefix, end_epoch, base_dir,
          display=200, base_lr=0.01):
    """
    train RNet
    :param net
    :param prefix:
    :param end_epoch:16
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """
    net = prefix.split('/')[-1]
    #label file
    label_file = os.path.join(base_dir,'train_%s_landmark.txt' % net)
    #label_file = os.path.join(base_dir,'landmark_12_few.txt')
    print label_file 
    f = open(label_file, 'r')
    num = len(f.readlines())
    print("Total datasets is: ", num)
    print prefix

    pos_dir = os.path.join(base_dir,'pos_landmark.tfrecord_shuffle')
    part_dir = os.path.join(base_dir,'part_landmark.tfrecord_shuffle')
    neg_dir = os.path.join(base_dir,'neg_landmark.tfrecord_shuffle')
    landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
    dataset_dirs = [pos_dir,part_dir,neg_dir,landmark_dir]
    pos_radio = 1.0/6;part_radio = 1.0/6;landmark_radio=1.0/6;neg_radio=3.0/6
    pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))
    assert pos_batch_size != 0,"Batch Size Error "
    part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
    assert part_batch_size != 0,"Batch Size Error "        
    neg_batch_size = int(np.ceil(config.BATCH_SIZE*neg_radio))
    assert neg_batch_size != 0,"Batch Size Error "
    landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))
    assert landmark_batch_size != 0,"Batch Size Error "
    batch_sizes = [pos_batch_size,part_batch_size,neg_batch_size,landmark_batch_size]
    image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net)        
    
    if net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    #define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,10],name='landmark_target')
    #class,regression
    cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op = net_factory(input_image, label, bbox_target,landmark_target,training=True)
    #train,update learning rate(3 loss)
    train_op, lr_op = train_model(base_lr, radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_landmark_loss*landmark_loss_op + L2_loss_op, num)
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    #save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)
    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    summary_op = tf.summary.merge_all()
    logs_dir = "../logs/%s" %(net)
    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()    
    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            
            if (step+1) % display == 0:
                #acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})                
                print("%s : Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f, landmark loss: %4f,L2 loss: %4f,lr:%f " % (
                datetime.now(), step+1, acc, cls_loss, bbox_loss, landmark_loss, L2_loss, lr))
            #save every two epochs
            if i * config.BATCH_SIZE > num*2:
                epoch = epoch + 1
                i = 0
                saver.save(sess, prefix, global_step=epoch*2)
            writer.add_summary(summary,global_step=step)
    except tf.errors.OutOfRangeError:
        print("finish")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    base_dir = '../data/imglists/RNet'
    model_path = '../models_check/RNet'  
    prefix = model_path
    end_epoch = 22
    display = 1000
    lr = 0.01
    train(R_Net, prefix, end_epoch, base_dir, display=display, base_lr=lr)