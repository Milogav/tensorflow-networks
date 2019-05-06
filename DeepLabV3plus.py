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

def sepConvUnit(layer,num_filters,name,kernel_size=3,dropout=0,stride=1,addRelu=True,isTraining=True):
    with tf.name_scope(name):
        layer = tf.layers.separable_conv2d(layer,num_filters,kernel_size,strides = stride,padding='same')
        #layer = tf.layers.batch_normalization(layer,training=isTraining)
        if addRelu:
            layer = tf.nn.relu(layer)
        if dropout:
            layer = tf.layers.dropout(layer,rate=dropout,training=isTraining)
        return layer

def middleBlock(layer,name,num_filters,dropout=0):
    with tf.name_scope(name):
        x = tf.nn.relu(layer)
        sepConv1 = sepConvUnit(x,num_filters,'SepConv_1',dropout=dropout,addRelu=True,isTraining=True)
        sepConv2 = sepConvUnit(sepConv1,num_filters,'SepConv_2',addRelu=True,isTraining=True)
        sepConv3 = sepConvUnit(sepConv2,num_filters,'SepConv_3',addRelu=False,isTraining=True)

        add = tf.add(layer,sepConv3)
        return add

def asppUnit(layer,num_filters,name,isTraining):

    l1 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=1,padding='same')
    #l1 = tf.layers.batch_normalization(l1,training=isTraining)
    l1 = tf.nn.relu(l1)

    l2 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=4,padding='same')
    #l2 = tf.layers.batch_normalization(l2,training=isTraining)
    l2 = tf.nn.relu(l2)

    l3 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=8,padding='same')
    #l3 = tf.layers.batch_normalization(l3,training=isTraining)
    l3 = tf.nn.relu(l3)

    l4 = tf.layers.conv2d(layer,num_filters,kernel_size=3,dilation_rate=16,padding='same')
    #l4 = tf.layers.batch_normalization(l4,training=isTraining)
    l4 = tf.nn.relu(l4)

    pool = tf.layers.max_pooling2d(layer,2,2)
    l5 = tf.image.resize_bilinear(pool,l1.shape[1:3])

    lCat = tf.concat([l1,l2,l3,l4,l5],3)

    out = tf.layers.conv2d(lCat,num_filters,kernel_size=1,padding='same')
    #out = tf.layers.batch_normalization(out,training=isTraining)
    out = tf.nn.relu(out) 
    
    return out


def transposeConvUnit(layer,num_filters,name,dropout=0,addRelu=True,isTraining=True):
    with tf.name_scope(name):
        layer = tf.layers.conv2d_transpose(layer,num_filters,kernel_size=(3,3),strides=2,padding='same')
        #layer = tf.layers.batch_normalization(layer,training=isTraining)
        if addRelu:
            layer = tf.nn.relu(layer)
        if dropout:
            layer = tf.layers.dropout(layer,rate=dropout,training=isTraining)
        return layer

def fit_tensor_shapes(refTensor,tensor):
    _,h,w,_ = refTensor.shape
    tensor = tf.image.resize_image_with_crop_or_pad(tensor,int(h),int(w))
    return tensor

    
def Network(inputs,isTraining=True):
###Deeplabv3+ with Xception architecture
    with tf.name_scope('Network'):
        
       ##### ENTRY FLOW#####  
        conv1 = convUnit(inputs,48,'Conv_1', addRelu = True,isTraining=isTraining)
        conv2 = convUnit(conv1,48,'Conv_2',stride = 2, addRelu = True, isTraining=isTraining)

        conv1x1_1 = convUnit(conv2,48,'Conv_1x1_1',kernel_size=1,stride = 2,addRelu=False, isTraining=isTraining)
        
        sepConv1 = sepConvUnit(conv2,48,'SepConv_1',addRelu=True,isTraining=isTraining)
        sepConv2 = sepConvUnit(sepConv1,48,'SepConv_2',addRelu=True,isTraining=isTraining)
        sepConv3 = sepConvUnit(sepConv2,48,'SepConv_3',addRelu=False,stride=2,isTraining=isTraining)

        add1 = tf.add(conv1x1_1,sepConv3,name='Add_1')

        conv1x1_2 = convUnit(add1,48,'Conv_1x1_2',kernel_size=1,dropout=0,stride = 2,addRelu=False, isTraining=isTraining)

        radd1 = tf.nn.relu(add1)
        sepConv4 = sepConvUnit(radd1,48,'SepConv_4',addRelu=True,isTraining=isTraining)
        sepConv5 = sepConvUnit(sepConv4,48,'SepConv_5',addRelu=True,isTraining=isTraining)
        sepConv6 = sepConvUnit(sepConv5,48,'SepConv_6',addRelu=False,stride=2,isTraining=isTraining)

        add2 = tf.add(conv1x1_2,sepConv6,name='Add_2')

        conv1x1_3 = convUnit(add2,48,'Conv_1x1_3',kernel_size=1,dropout=0,stride = 2,addRelu=False, isTraining=isTraining)
  
        radd2 = tf.nn.relu(add2)
        sepConv7 = sepConvUnit(radd2,48,'SepConv_7',addRelu=True,isTraining=isTraining)
        sepConv8 = sepConvUnit(sepConv7,48,'SepConv_8',addRelu=True,isTraining=isTraining)
        sepConv9 = sepConvUnit(sepConv8,48,'SepConv_9',addRelu=False,stride=2,isTraining=isTraining)
        
        y = tf.add(conv1x1_3,sepConv9,name='Add_3')

       ##### MIDDLE BLOCKS #####  
        for i in range(20):
            y = middleBlock(y,'midBlock_'+str(i),num_filters=48,dropout=0.5)

       ##### EXIT FLOW #### 

        conv1x1_4 = convUnit(y,48,'Conv_1x1_4',kernel_size=1,stride = 2,addRelu=False, isTraining=isTraining,dropout=0.5)

        radd3 = tf.nn.relu(y)
        sepConv10 = sepConvUnit(radd3,48,'SepConv_10',addRelu=True,isTraining=isTraining)
        sepConv11 = sepConvUnit(sepConv10,48,'SepConv_11',addRelu=True,isTraining=isTraining)
        sepConv12 = sepConvUnit(sepConv11,48,'SepConv_12',addRelu=False,stride=2,isTraining=isTraining)

        add4 = tf.add(conv1x1_4,sepConv12)

        sepConv13 = sepConvUnit(add4,48,'SepConv_13',addRelu=True,isTraining=isTraining)
        sepConv14 = sepConvUnit(sepConv13,48,'SepConv_14',addRelu=True,isTraining=isTraining)
        sepConv15 = sepConvUnit(sepConv14,48,'SepConv_15',addRelu=True,isTraining=isTraining)

        aspp = asppUnit(sepConv15,48,name='ASPP',isTraining=isTraining)
        up1 = tf.image.resize_bilinear(aspp,add2.shape[1:3])
        cat1 = tf.concat([up1,add2],3)
        up2 = tf.image.resize_bilinear(cat1,conv2.shape[1:3])
        cat2 = tf.concat([up2,conv2],3)

        output = transposeConvUnit(cat2,48,name='upConv1',dropout=0,addRelu=True,isTraining=True)
        output = tf.image.resize_bilinear(output,inputs.shape[1:3])
        output = tf.layers.conv2d(output,filters = 1,kernel_size=3,padding='same')


    return output
