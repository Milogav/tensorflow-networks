import tensorflow as tf
import os
import numpy as np

def scriptPath():
    return os.path.realpath(__file__)

def convUnit(layer,num_filters,name,kernel_size=3,dropout=0,stride=1,addRelu=True,isTraining=True):
    with tf.name_scope(name):
        layer = tf.layers.conv2d(layer,num_filters,kernel_size=kernel_size,padding='same',strides = stride)
        #layer = tf.layers.batch_normalization(layer,training=isTraining)
        if addRelu:
            layer = tf.nn.relu(layer)
        if dropout:
            layer = tf.layers.dropout(layer,rate=dropout,training=isTraining)
        return layer


def asppUnit(layer,num_filters,name,isTraining):

    l1 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=1,padding='same')
    l1 = tf.layers.batch_normalization(l1,training=isTraining)
    l1 = tf.nn.relu(l1)

    l2 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=4,padding='same')
    l2 = tf.layers.batch_normalization(l2,training=isTraining)
    l2 = tf.nn.relu(l2)

    l3 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=8,padding='same')
    l3 = tf.layers.batch_normalization(l3,training=isTraining)
    l3 = tf.nn.relu(l3)

    l4 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=16,padding='same')
    l4 = tf.layers.batch_normalization(l4,training=isTraining)
    l4 = tf.nn.relu(l4)

    pool = tf.layers.max_pooling2d(layer,2,2)
    l5 = tf.image.resize_bilinear(pool,l1.shape[1:3])

    lCat = tf.concat([l1,l2,l3,l4,l5],3)

    out = tf.layers.conv2d(lCat,num_filters,kernel_size=1,padding='same')
    out = tf.layers.batch_normalization(out,training=isTraining)
    out = tf.nn.relu(out) 
    
    return out


def transposeConvUnit(layer,num_filters,name,dropout=0,kernel_size = 3,isTraining=True):
    with tf.name_scope(name):
        layer = tf.layers.conv2d_transpose(layer,num_filters,kernel_size=kernel_size,strides=2,padding='same')
        #layer = tf.layers.batch_normalization(layer,training=isTraining)
        layer = tf.nn.relu(layer)
        if dropout:
            layer = tf.layers.dropout(layer,rate=dropout,training=isTraining)
        return layer

def fit_tensor_shapes(refTensor,tensor):
    _,h,w,_ = refTensor.shape
    tensor = tf.image.resize_image_with_crop_or_pad(tensor,int(h),int(w))
    return tensor

def resBlock(layer,num_filters,name,kernel_size=3,dropout=0,isTraining=True):
    y = convUnit(layer,num_filters,name+'-conv1',3,dropout=dropout,isTraining = isTraining)
    y = convUnit(y,num_filters,name+'-conv2',3,dropout=dropout,isTraining = isTraining)
    out = tf.add(layer,y)
   # out = convUnit(out,num_filters,name+'-out',1,isTraining = isTraining)
    return out
    
def Network(inputs,isTraining=True):
    with tf.name_scope('Network'):

        x = convUnit(inputs,32,'conv1',kernel_size = 3,isTraining = isTraining)

        x = convUnit(x,32,'conv2',kernel_size = 3,stride = 2,isTraining = isTraining)
        for i in range(0,5):
            x = resBlock(x,32,'resBlock'+str(i),isTraining=isTraining)
        
        x = convUnit(x,32,'conv3',kernel_size = 3,stride = 2,isTraining = isTraining)
        for i in range(5,10):
            x = resBlock(x,32,'resBlock'+str(i),isTraining=isTraining)

        x = convUnit(x,64,'conv4',kernel_size = 3,stride = 2,isTraining = isTraining)
        for i in range(10,15):
            x = resBlock(x,64,'resBlock'+str(i),dropout=0.5,isTraining=isTraining)
        
        x = transposeConvUnit(x,64,'trconv1',kernel_size = 3,isTraining = isTraining)
        for i in range(15,20):
            x = resBlock(x,64,'resBlock'+str(i),dropout=0.5,isTraining=isTraining)

        x = transposeConvUnit(x,32,'trconv2',kernel_size = 3,isTraining = isTraining)
        for i in range(20,25):
            x = resBlock(x,32,'resBlock'+str(i),isTraining=isTraining)
        
        x = transposeConvUnit(x,32,'trconv2',kernel_size = 3,isTraining = isTraining)
        x = convUnit(x,32,'conv5',kernel_size = 3,isTraining = isTraining)
        x = tf.layers.conv2d(x,1,kernel_size = 3,padding='same')
        x = fit_tensor_shapes(inputs,x)
    return x
