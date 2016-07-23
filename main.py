import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import time
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import pdb
import random
from keras.utils import np_utils 
random.seed(2016)

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
import tensorflow as tf
sess = tf.InteractiveSession()

def imgPaths(train=1):
    imgPaths = []
    if train == 1:
        folder = "train/"
    else:
        folder = "test/"
    for f in glob.glob(folder+"*/*"):
        imgPaths.append(f)
    return imgPaths

def splitTrainTest(paths,splitPerc=0.9):
    nObs = len(paths)
    splitPoint = int(math.floor(splitPerc*nObs))
    print("Total size of training set = ",nObs)
    train, test = paths[:splitPoint], paths[splitPoint:]
    print("Holding out %d for testing, training on %d" % (len(test),len(train)))
    return train, test


def showImg(img,title="Image"):
    fig = plt.figure()
    plt.imshow(img,cmap=cm.gray)
    fig.suptitle(title,fontsize=20)
    plt.show()

def loadImg(path,channels=1,w=640,h=480):
    if channels == 1:
        img = cv2.imread(path,0)
    else: 
        img = cv2.imread(path)
    resized = cv2.resize(img,(h,w),interpolation=cv2.INTER_LINEAR)
    return resized

def getLabelName(label):
    labels = {0:"normal driving",1:"texting R",2:"talking R",3:"texting L",4:"talking L",5:"radio",6:"drinking",7:"reaching behind",8:"hair/makeup", 9: "talking to passenger"}
    return labels[label]

def oneHot(y):
    return np.eye(10)[y]

def dataIterator(paths,channels=1,w=640,h=480):
    # get data paths
    batchIdx = 0

    while True:
        # Shuffle - to do
        random.shuffle(paths)
        batchSize = 8 
        for batchIdx in range(0, len(paths), batchSize):
            xTrainBatch = []
            yTrainBatch = []
            for i in range(batchIdx,batchIdx+batchSize):
                xTrainBatch.append(loadImg(paths[i],channels,w,h))
                yTrainBatch.append(oneHot(int(paths[i].split("/")[1][1])))
            xTrainBatch = np.array(xTrainBatch)
            yTrainBatch = np.array(yTrainBatch)
            yield xTrainBatch, yTrainBatch


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def weightVariable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

if __name__ == "__main__":
    nClasses = 10
    w,h = 640/2, 480/2
    train = 1
    crossValidation = 1
    trainPaths = imgPaths(train=train)
    if crossValidation == 1:
        trainPaths, testPaths = splitTrainTest(trainPaths)
    iter_ = dataIterator(trainPaths,channels=1,w=w,h=h)

    
    x = tf.placeholder(tf.float32,shape = [None,w,h])
    y_ = tf.placeholder(tf.float32,shape = [None,10])

    W_conv1 = weightVariable([3,3,1,32])
    b_conv1 = biasVariable([32])

    xImage = tf.reshape(x,[-1,w,h,1])
    h_conv1 = tf.nn.relu(conv2d(xImage,W_conv1) + b_conv1)
    h_pool1 = maxPool(h_conv1)

    W_conv2 = weightVariable([3,3,32,64])
    b_conv2 = biasVariable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = maxPool(h_conv2)

    W_conv3 = weightVariable([3,3,64,72])
    b_conv3 = biasVariable([72])
    h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
    h_pool3 = maxPool(h_conv3)
        
    W_conv4 = weightVariable([3,3,72,84])
    b_conv4 = biasVariable([84])
    h_conv4 = tf.nn.relu(conv2d(h_pool3,W_conv4) + b_conv4)
    h_pool4 = maxPool(h_conv4)

    W_conv5 = weightVariable([3,3,84,96])
    b_conv5 = biasVariable([96])
    h_conv5 = tf.nn.relu(conv2d(h_pool4,W_conv5) + b_conv5)
    h_pool5 = maxPool(h_conv5)

    W_conv6 = weightVariable([3,3,96,108])
    b_conv6 = biasVariable([108])
    h_conv6 = tf.nn.relu(conv2d(h_pool5,W_conv6) + b_conv6)
    h_pool6 = maxPool(h_conv6)

    W_fc1 = weightVariable([5*4*108,128])
    b_fc1 = biasVariable([128])

    h_pool6_flat = tf.reshape(h_pool6,[-1,5*4*108])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat,W_fc1) + b_fc1)

    W_fc2 = weightVariable([128,10])
    b_fc2 = biasVariable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    sess.run(tf.initialize_all_variables())

    for i in range(100):
        batch = iter_.next()
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})
            print("step %d, training accuracy %g"%(i,train_accuracy))
            train_step.run(feed_dict={x:batch[0],y_:batch[1]})
            


                        


    


    
    
    

        
   
    
