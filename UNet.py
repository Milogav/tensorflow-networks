import tensorflow as tf
import os
import numpy as np

def scriptPath():
    return os.path.realpath(__file__)


def convUnit(layer,num_filters,name,dropout=0,isTraining=True):
    with tf.name_scope(name):
        layer = tf.layers.conv2d(layer,num_filters,kernel_size=(3,3),padding='same')
        # layer = tf.layers.batch_normalization(layer,training=isTraining)
        layer = tf.nn.relu(layer)
        if dropout:
            layer = tf.layers.dropout(layer,rate=dropout,training=isTraining)
        return layer
    
def asppUnit(layer,num_filters,name,dropout = 0,isTraining=True):

    l1 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=1,padding='same')
    # l1 = tf.layers.batch_normalization(l1,training=isTraining)
    l1 = tf.nn.relu(l1)

    l2 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=4,padding='same')
    # l2 = tf.layers.batch_normalization(l2,training=isTraining)
    l2 = tf.nn.relu(l2)

    l3 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=8,padding='same')
    # l3 = tf.layers.batch_normalization(l3,training=isTraining)
    l3 = tf.nn.relu(l3)

    l4 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=16,padding='same')
    # l4 = tf.layers.batch_normalization(l4,training=isTraining)
    l4 = tf.nn.relu(l4)

    pool = tf.layers.max_pooling2d(layer,2,2)
    l5 = tf.image.resize_bilinear(pool,l1.shape[1:3])

    lCat = tf.concat([l1,l2,l3,l4,l5],3)

    out = tf.layers.conv2d(lCat,num_filters,kernel_size=1,padding='same')
    # out = tf.layers.batch_normalization(out,training=isTraining)
    out = tf.nn.relu(out)

    if dropout:
            out = tf.layers.dropout(out,rate=dropout,training=isTraining) 
    
    return out

def transposeConvUnit(layer,num_filters,name,dropout=0,isTraining=True):
    with tf.name_scope(name):
        layer = tf.layers.conv2d_transpose(layer,num_filters,kernel_size=(3,3),strides=2,padding='same')
        # layer = tf.layers.batch_normalization(layer,training=isTraining)
        layer = tf.nn.relu(layer)
        if dropout:
            layer = tf.layers.dropout(layer,rate=dropout,training=isTraining)
        return layer

def fit_tensor_shapes(refTensor,tensor):
    _,h,w,_ = refTensor.shape
    tensor = tf.image.resize_image_with_crop_or_pad(tensor,int(h),int(w))
    return tensor

    
def Network(inputs,isTraining=True):

    with tf.name_scope('Network'):
            
        conv1 = convUnit(inputs,48,'Conv_1',dropout=0,isTraining=isTraining)
        mp1 = tf.layers.max_pooling2d(conv1, pool_size =(2,2),strides = 2,padding = 'same',name='MaxPool_1')
        
        conv2 = convUnit(mp1,48,'Conv_2',dropout=0,isTraining=isTraining)
        mp2 = tf.layers.max_pooling2d(conv2, pool_size =(2,2),strides = 2,padding = 'same',name='MaxPool_2')
        
        conv3 = convUnit(mp2,64,'Conv_3',dropout=0,isTraining=isTraining)
        mp3 = tf.layers.max_pooling2d(conv3, pool_size =(2,2),strides = 2,padding = 'same',name='MaxPool_3')
        
        conv4 = convUnit(mp3,64,'Conv_4',dropout=0,isTraining=isTraining)
        mp4 = tf.layers.max_pooling2d(conv4, pool_size =(2,2),strides = 2,padding = 'same',name='MaxPool_4')
        
        conv5 = convUnit(mp4,64,'Conv_5',dropout=0.5,isTraining=isTraining)
        upconv5 = transposeConvUnit(conv5,64,'UpConv5',dropout=0.5,isTraining=isTraining)

        upconv5r = fit_tensor_shapes(conv4, upconv5)
        cat6 = tf.concat([conv4, upconv5r], 3)
        upconv6 = transposeConvUnit(cat6,64,'UpConv6',dropout=0,isTraining=isTraining)

        upconv6r = fit_tensor_shapes(conv3, upconv6)
        cat7 = tf.concat([conv3, upconv6r], 3)
        upconv7 = transposeConvUnit(cat7,64,'UpConv6',dropout=0,isTraining=isTraining)

        upconv7r = fit_tensor_shapes(conv2, upconv7)
        cat8 = tf.concat([conv2, upconv7r], 3)
        upconv8 = transposeConvUnit(cat8,48,'UpConv6',dropout=0,isTraining=isTraining)

        upconv8r = fit_tensor_shapes(conv1, upconv8)
        cat9 = tf.concat([conv1, upconv8r], 3)
        conv9 = convUnit(cat9,48,'Conv_9',isTraining=isTraining)

        conv10 = tf.layers.conv2d(conv9,1,kernel_size=(3,3),padding='same',name='Conv_10')
        
    return conv10
