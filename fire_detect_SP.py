import cv2
import os
import sys
import math
import numpy as np
import argparse
import time
################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

################################################################################

#from inceptionVxOnFire import construct_inceptionv1onfire, construct_inceptionv3onfire, construct_inceptionv4onfire
def construct_inceptionv3onfire(x,y, training=False, enable_batch_norm=True):

    # build network as per architecture

    network = input_data(shape=[None, y, x, 3])

    conv1_3_3 = conv_2d(network, 32, 3, strides=2, activation='relu', name = 'conv1_3_3',padding='valid')
    conv2_3_3 = conv_2d(conv1_3_3, 32, 3, strides=1, activation='relu', name = 'conv2_3_3',padding='valid')
    conv3_3_3 = conv_2d(conv2_3_3, 64, 3, strides=2, activation='relu', name = 'conv3_3_3')

    pool1_3_3 = max_pool_2d(conv3_3_3, 3,strides=2)
    if enable_batch_norm:
        pool1_3_3 = batch_normalization(pool1_3_3)
    conv1_7_7 = conv_2d(pool1_3_3, 80,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    conv2_7_7 = conv_2d(conv1_7_7, 96,3, strides=1, activation='relu', name='conv2_7_7_s2',padding='valid')
    pool2_3_3= max_pool_2d(conv2_7_7,3,strides=2)

    inception_3a_1_1 = conv_2d(pool2_3_3,64, filter_size=1, activation='relu', name='inception_3a_1_1')

    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 48, filter_size=1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 64, filter_size=[5,5],  activation='relu',name='inception_3a_3_3')


    inception_3a_5_5_reduce = conv_2d(pool2_3_3, 64, filter_size=1, activation='relu', name = 'inception_3a_5_5_reduce')
    inception_3a_5_5_asym_1 = conv_2d(inception_3a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_3a_5_5_asym_1')
    inception_3a_5_5 = conv_2d(inception_3a_5_5_asym_1, 96, filter_size=[3,3],  name = 'inception_3a_5_5')


    inception_3a_pool = avg_pool_2d(pool2_3_3, kernel_size=3, strides=1,  name='inception_3a_pool')
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a

    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3, name='inception_3a_output')


    inception_5a_1_1 = conv_2d(inception_3a_output, 96, 1, activation='relu', name='inception_5a_1_1')

    inception_5a_3_3_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3_asym_1 = conv_2d(inception_5a_3_3_reduce, 64, filter_size=[1,7],  activation='relu',name='inception_5a_3_3_asym_1')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_asym_1,96, filter_size=[7,1],  activation='relu',name='inception_5a_3_3')


    inception_5a_5_5_reduce = conv_2d(inception_3a_output, 64, filter_size=1, activation='relu', name = 'inception_5a_5_5_reduce')
    inception_5a_5_5_asym_1 = conv_2d(inception_5a_5_5_reduce, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_1')
    inception_5a_5_5_asym_2 = conv_2d(inception_5a_5_5_asym_1, 64, filter_size=[1,7],  name = 'inception_5a_5_5_asym_2')
    inception_5a_5_5_asym_3 = conv_2d(inception_5a_5_5_asym_2, 64, filter_size=[7,1],  name = 'inception_5a_5_5_asym_3')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_asym_3, 96, filter_size=[1,7],  name = 'inception_5a_5_5')


    inception_5a_pool = avg_pool_2d(inception_3a_output, kernel_size=3, strides=1 )
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 96, filter_size=1, activation='relu', name='inception_5a_pool_1_1')

    # merge the inception_5a__
    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], mode='concat', axis=3)



    inception_7a_1_1 = conv_2d(inception_5a_output, 80, 1, activation='relu', name='inception_7a_1_1')
    inception_7a_3_3_reduce = conv_2d(inception_5a_output, 96, filter_size=1, activation='relu', name='inception_7a_3_3_reduce')
    inception_7a_3_3_asym_1 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[1,3],  activation='relu',name='inception_7a_3_3_asym_1')
    inception_7a_3_3_asym_2 = conv_2d(inception_7a_3_3_reduce, 96, filter_size=[3,1],  activation='relu',name='inception_7a_3_3_asym_2')
    inception_7a_3_3=merge([inception_7a_3_3_asym_1,inception_7a_3_3_asym_2],mode='concat',axis=3)

    inception_7a_5_5_reduce = conv_2d(inception_5a_output, 66, filter_size=1, activation='relu', name = 'inception_7a_5_5_reduce')
    inception_7a_5_5_asym_1 = conv_2d(inception_7a_5_5_reduce, 96, filter_size=[3,3],  name = 'inception_7a_5_5_asym_1')
    inception_7a_5_5_asym_2 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[1,3],  activation='relu',name='inception_7a_5_5_asym_2')
    inception_7a_5_5_asym_3 = conv_2d(inception_7a_3_3_asym_1, 96, filter_size=[3,1],  activation='relu',name='inception_7a_5_5_asym_3')
    inception_7a_5_5=merge([inception_7a_5_5_asym_2,inception_7a_5_5_asym_3],mode='concat',axis=3)


    inception_7a_pool = avg_pool_2d(inception_5a_output, kernel_size=3, strides=1 )
    inception_7a_pool_1_1 = conv_2d(inception_7a_pool, 96, filter_size=1, activation='relu', name='inception_7a_pool_1_1')

    # merge the inception_7a__
    inception_7a_output = merge([inception_7a_1_1, inception_7a_3_3, inception_7a_5_5, inception_7a_pool_1_1], mode='concat', axis=3)



    pool5_7_7=global_avg_pool(inception_7a_output)
    if(training):
        pool5_7_7=dropout(pool5_7_7,0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv3',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model

################################################################################

# InceptionV4 : definition of inception_block_a

def inception_block_a(input_a):

    inception_a_conv1_1_1 = conv_2d(input_a,96,1,activation='relu',name='inception_a_conv1_1_1')

    inception_a_conv1_3_3_reduce = conv_2d(input_a,64,1,activation='relu',name='inception_a_conv1_3_3_reduce')
    inception_a_conv1_3_3 = conv_2d(inception_a_conv1_3_3_reduce,96,3,activation='relu',name='inception_a_conv1_3_3')

    inception_a_conv2_3_3_reduce = conv_2d(input_a,64,1,activation='relu',name='inception_a_conv2_3_3_reduce')
    inception_a_conv2_3_3_sym_1 = conv_2d(inception_a_conv2_3_3_reduce,96,3,activation='relu',name='inception_a_conv2_3_3')
    inception_a_conv2_3_3 = conv_2d(inception_a_conv2_3_3_sym_1,96,3,activation='relu',name='inception_a_conv2_3_3')

    inception_a_pool = avg_pool_2d(input_a,kernel_size=3,name='inception_a_pool',strides=1)
    inception_a_pool_1_1 = conv_2d(inception_a_pool,96,1,activation='relu',name='inception_a_pool_1_1')

    # merge inception_a

    inception_a = merge([inception_a_conv1_1_1,inception_a_conv1_3_3,inception_a_conv2_3_3,inception_a_pool_1_1],mode='concat',axis=3)

    return inception_a


################################################################################

# InceptionV4 : definition of reduction_block_a

def reduction_block_a(reduction_input_a):

    reduction_a_conv1_1_1 = conv_2d(reduction_input_a,384,3,strides=2,padding='valid',activation='relu',name='reduction_a_conv1_1_1')

    reduction_a_conv2_1_1 = conv_2d(reduction_input_a,192,1,activation='relu',name='reduction_a_conv2_1_1')
    reduction_a_conv2_3_3 = conv_2d(reduction_a_conv2_1_1,224,3,activation='relu',name='reduction_a_conv2_3_3')
    reduction_a_conv2_3_3_s2 = conv_2d(reduction_a_conv2_3_3,256,3,strides=2,padding='valid',activation='relu',name='reduction_a_conv2_3_3_s2')

    reduction_a_pool = max_pool_2d(reduction_input_a,strides=2,padding='valid',kernel_size=3,name='reduction_a_pool')

    # merge reduction_a

    reduction_a = merge([reduction_a_conv1_1_1,reduction_a_conv2_3_3_s2,reduction_a_pool],mode='concat',axis=3)

    return reduction_a

################################################################################

# InceptionV4 : definition of inception_block_b

def inception_block_b(input_b):

    inception_b_1_1 = conv_2d(input_b, 384, 1, activation='relu', name='inception_b_1_1')

    inception_b_3_3_reduce = conv_2d(input_b, 192, filter_size=1, activation='relu', name='inception_b_3_3_reduce')
    inception_b_3_3_asym_1 = conv_2d(inception_b_3_3_reduce, 224, filter_size=[1,7],  activation='relu',name='inception_b_3_3_asym_1')
    inception_b_3_3 = conv_2d(inception_b_3_3_asym_1, 256, filter_size=[7,1],  activation='relu',name='inception_b_3_3')


    inception_b_5_5_reduce = conv_2d(input_b, 192, filter_size=1, activation='relu', name = 'inception_b_5_5_reduce')
    inception_b_5_5_asym_1 = conv_2d(inception_b_5_5_reduce, 192, filter_size=[7,1],  name = 'inception_b_5_5_asym_1')
    inception_b_5_5_asym_2 = conv_2d(inception_b_5_5_asym_1, 224, filter_size=[1,7],  name = 'inception_b_5_5_asym_2')
    inception_b_5_5_asym_3 = conv_2d(inception_b_5_5_asym_2, 224, filter_size=[7,1],  name = 'inception_b_5_5_asym_3')
    inception_b_5_5 = conv_2d(inception_b_5_5_asym_3, 256, filter_size=[1,7],  name = 'inception_b_5_5')


    inception_b_pool = avg_pool_2d(input_b, kernel_size=3, strides=1 )
    inception_b_pool_1_1 = conv_2d(inception_b_pool, 128, filter_size=1, activation='relu', name='inception_b_pool_1_1')

    # merge the inception_b

    inception_b_output = merge([inception_b_1_1, inception_b_3_3, inception_b_5_5, inception_b_pool_1_1], mode='concat', axis=3)

    return inception_b_output

################################################################################

# InceptionV4 : definition of reduction_block_b

def reduction_block_b(reduction_input_b):

    reduction_b_1_1 = conv_2d(reduction_input_b,192,1,activation='relu',name='reduction_b_1_1')
    reduction_b_1_3 = conv_2d(reduction_b_1_1,192,3,strides=2,padding='valid',name='reduction_b_1_3')

    reduction_b_3_3_reduce = conv_2d(reduction_input_b, 256, filter_size=1, activation='relu', name='reduction_b_3_3_reduce')
    reduction_b_3_3_asym_1 = conv_2d(reduction_b_3_3_reduce, 256, filter_size=[1,7],  activation='relu',name='reduction_b_3_3_asym_1')
    reduction_b_3_3_asym_2 = conv_2d(reduction_b_3_3_asym_1, 320, filter_size=[7,1],  activation='relu',name='reduction_b_3_3_asym_2')
    reduction_b_3_3=conv_2d(reduction_b_3_3_asym_2,320,3,strides=2,activation='relu',padding='valid',name='reduction_b_3_3')

    reduction_b_pool = max_pool_2d(reduction_input_b,kernel_size=3,strides=2,padding='valid')

    # merge the reduction_b

    reduction_b_output = merge([reduction_b_1_3,reduction_b_3_3,reduction_b_pool],mode='concat',axis=3)

    return reduction_b_output

################################################################################

# InceptionV4 : defintion of inception_block_c

def inception_block_c(input_c):
    inception_c_1_1 = conv_2d(input_c, 256, 1, activation='relu', name='inception_c_1_1')
    inception_c_3_3_reduce = conv_2d(input_c, 384, filter_size=1, activation='relu', name='inception_c_3_3_reduce')
    inception_c_3_3_asym_1 = conv_2d(inception_c_3_3_reduce, 256, filter_size=[1,3],  activation='relu',name='inception_c_3_3_asym_1')
    inception_c_3_3_asym_2 = conv_2d(inception_c_3_3_reduce, 256, filter_size=[3,1],  activation='relu',name='inception_c_3_3_asym_2')
    inception_c_3_3=merge([inception_c_3_3_asym_1,inception_c_3_3_asym_2],mode='concat',axis=3)

    inception_c_5_5_reduce = conv_2d(input_c, 384, filter_size=1, activation='relu', name = 'inception_c_5_5_reduce')
    inception_c_5_5_asym_1 = conv_2d(inception_c_5_5_reduce, 448, filter_size=[1,3],  name = 'inception_c_5_5_asym_1')
    inception_c_5_5_asym_2 = conv_2d(inception_c_5_5_asym_1, 512, filter_size=[3,1],  activation='relu',name='inception_c_5_5_asym_2')
    inception_c_5_5_asym_3 = conv_2d(inception_c_5_5_asym_2, 256, filter_size=[1,3],  activation='relu',name='inception_c_5_5_asym_3')

    inception_c_5_5_asym_4 = conv_2d(inception_c_5_5_asym_2, 256, filter_size=[3,1],  activation='relu',name='inception_c_5_5_asym_4')
    inception_c_5_5=merge([inception_c_5_5_asym_4,inception_c_5_5_asym_3],mode='concat',axis=3)


    inception_c_pool = avg_pool_2d(input_c, kernel_size=3, strides=1 )
    inception_c_pool_1_1 = conv_2d(inception_c_pool, 256, filter_size=1, activation='relu', name='inception_c_pool_1_1')

    # merge the inception_c

    inception_c_output = merge([inception_c_1_1, inception_c_3_3, inception_c_5_5, inception_c_pool_1_1], mode='concat', axis=3)

    return inception_c_output

################################################################################

def construct_inceptionv4onfire(x,y, training=True, enable_batch_norm=True):

    network = input_data(shape=[None, y, x, 3])

    #stem of inceptionV4

    conv1_3_3 = conv_2d(network,32,3,strides=2,activation='relu',name='conv1_3_3_s2',padding='valid')
    conv2_3_3 = conv_2d(conv1_3_3,32,3,activation='relu',name='conv2_3_3')
    conv3_3_3 = conv_2d(conv2_3_3,64,3,activation='relu',name='conv3_3_3')
    b_conv_1_pool = max_pool_2d(conv3_3_3,kernel_size=3,strides=2,padding='valid',name='b_conv_1_pool')
    if enable_batch_norm:
        b_conv_1_pool = batch_normalization(b_conv_1_pool)
    b_conv_1_conv = conv_2d(conv3_3_3,96,3,strides=2,padding='valid',activation='relu',name='b_conv_1_conv')
    b_conv_1 = merge([b_conv_1_conv,b_conv_1_pool],mode='concat',axis=3)

    b_conv4_1_1 = conv_2d(b_conv_1,64,1,activation='relu',name='conv4_3_3')
    b_conv4_3_3 = conv_2d(b_conv4_1_1,96,3,padding='valid',activation='relu',name='conv5_3_3')

    b_conv4_1_1_reduce = conv_2d(b_conv_1,64,1,activation='relu',name='b_conv4_1_1_reduce')
    b_conv4_1_7 = conv_2d(b_conv4_1_1_reduce,64,[1,7],activation='relu',name='b_conv4_1_7')
    b_conv4_7_1 = conv_2d(b_conv4_1_7,64,[7,1],activation='relu',name='b_conv4_7_1')
    b_conv4_3_3_v = conv_2d(b_conv4_7_1,96,3,padding='valid',name='b_conv4_3_3_v')
    b_conv_4 = merge([b_conv4_3_3_v, b_conv4_3_3],mode='concat',axis=3)

    b_conv5_3_3 = conv_2d(b_conv_4,192,3,padding='valid',activation='relu',name='b_conv5_3_3',strides=2)
    b_pool5_3_3 = max_pool_2d(b_conv_4,kernel_size=3,padding='valid',strides=2,name='b_pool5_3_3')
    if enable_batch_norm:
        b_pool5_3_3 = batch_normalization(b_pool5_3_3)
    b_conv_5 = merge([b_conv5_3_3,b_pool5_3_3],mode='concat',axis=3)
    net = b_conv_5

    # inceptionV4 modules

    net=inception_block_a(net)

    net=inception_block_b(net)

    net=inception_block_c(net)

    pool5_7_7=global_avg_pool(net)
    if(training):
        pool5_7_7=dropout(pool5_7_7,0.4)
    loss = fully_connected(pool5_7_7, 2,activation='softmax')

    if(training):
        network = regression(loss, optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
    else:
        network=loss

    model = tflearn.DNN(network, checkpoint_path='inceptionv4onfire',
                        max_checkpoints=1, tensorboard_verbose=0)

    return model

################################################################################

################################################################################

# extract non-zero region of interest (ROI) in an otherwise zero'd image

def extract_bounded_nonzero(input):

    # take the first channel only (for speed)

    gray = input[:, :, 0];

    # find bounding rectangle of a non-zero region in an numpy array
    # credit: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    rows = np.any(gray, axis=1)
    cols = np.any(gray, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # cropping the non zero image

    return input[cmin:cmax,rmin:rmax]

################################################################################

# pad a supplied multi-channel image to the required [X,Y,C] size

def pad_image(image, new_width, new_height, pad_value = 0):

    # create an image of zeros, the same size as padding target size

    padded = np.zeros((new_width, new_height, image.shape[2]), dtype=np.uint8)

    # compute where our input image will go to centre it within the padded image

    pos_x = int(np.round((new_width / 2) - (image.shape[1] / 2)))
    pos_y = int(np.round((new_height / 2) - (image.shape[0] / 2)))

    # copy across the data from the input to the position centred within the padded image

    padded[pos_y:image.shape[0]+pos_y,pos_x:image.shape[1]+pos_x] = image

    return padded

################################################################################

# parse command line arguments

parser = argparse.ArgumentParser(description='Perform superpixel based InceptionV1/V3/V4 fire detection on incoming video')
parser.add_argument("-m", "--model_to_use", type=int, help="specify model to use", default=4, choices={ 3, 4})
#parser.add_argument('video_file', metavar='video_file', type=str, help='specify video file')
args = parser.parse_args()

#   construct and display model

print("Constructing SP-InceptionV" + str(args.model_to_use) + "-OnFire ...")


if (args.model_to_use == 3):

    # use InceptionV3-OnFire CNN model -  [Samarth/Bhowmik/Breckon, 2019]
    # N.B. weights_only=False as we are using Batch Normalization, and need those weights loaded also

    model = construct_inceptionv3onfire (224, 224, training=False)
    model.load(os.path.join("models/SP-InceptionV3-OnFire", "sp-inceptionv3onfire"),weights_only=False)

elif (args.model_to_use == 4):

    # use InceptionV4-OnFire CNN model -  [Samarth/Bhowmik/Breckon, 2019]
    # N.B. weights_only=False as we are using Batch Normalization, and need those weights loaded also

    model = construct_inceptionv4onfire (224, 224, training=False)
    model.load(os.path.join("models/SP-InceptionV4-OnFire", "sp-inceptionv4onfire"),weights_only=False)

print("Loaded CNN network weights ...")

################################################################################

# network input sizes

rows = 224
cols = 224

# display and loop settings

windowName = "Live Fire Detection - Superpixels with SP-InceptionV" + str(args.model_to_use) + "-OnFire"
keepProcessing = True

################################################################################

# load video file from first command line argument

video = cv2.VideoCapture(0)
time.sleep(0.1)
print("Loaded video ...")



cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

#width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#ffps = video.get(cv2.CAP_PROP_FPS)
#frame_time = round(1/fps)

while (keepProcessing):
    start_t = cv2.getTickCount()
    ret, frame = video.read()
    if not ret:
        print("... end of video file reached")
        break
    small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
    slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
    slic.iterate(10)

    segments = slic.getLabels()
    for (i, segVal) in enumerate(np.unique(segments)):
        mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)
        if ((args.model_to_use == 3) or (args.model_to_use == 4)):
            superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2RGB)
            superpixel = pad_image(extract_bounded_nonzero(superpixel), 224, 224)
        output = model.predict([superpixel])

        if round(output[0][0]) == 1:
            cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)

        else:
            cv2.drawContours(small_frame, contours, -1, (0,0,255), 1)

    stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency())
    cv2.imshow(windowName, small_frame)
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('x')):
        keepProcessing = False
    elif (key == ord('f')):
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)