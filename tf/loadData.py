import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import time
from tqdm import tqdm

#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import pdb
import random
from keras.utils import np_utils 
from bn import batchNorm
random.seed(2016)

def showImg(img,ohTruth,sleep=0):
    truth = np.where(ohTruth==1)[0][0]
    title = getLabelName(truth) 
    fig = plt.figure()
    plt.imshow(img,cmap=cm.gray)
    fig.suptitle(title,fontsize=20)
    plt.show()
    time.sleep(sleep)
    plt.close()

def getLabelName(label):
    labels = {0:"normal driving",1:"texting R",2:"talking R",3:"texting L",4:"talking L",5:"radio",6:"drinking",7:"reaching behind",8:"hair/makeup", 9: "talking to passenger"}
    return labels[label]

class dataLoader():
    def __init__(self,splitPerc = 0.9, width=160, height = 120, channels = 3):
        imgPaths = []
        for f in glob.glob("../train/*/*_x.jpg"):
            imgPaths.append(f)
        nObs = len(imgPaths)
        splitPoint = int(math.floor(splitPerc*nObs))
        print("Total size of training set = ",nObs)
        self.trainPaths, self.testPaths = imgPaths[:splitPoint], imgPaths[splitPoint:]
        self.batchIdxTrain = 0
        self.batchIdxTest = 0
        self.finished = 0
	self.w = width
	self.h = height 
	self.c = channels 
	print(" h, w, c = {%d,%d,%d}" % (self.w,self.h,self.c))
        print("Holding out %d for testing, training on %d" % (len(self.testPaths),len(self.trainPaths)))


    def loadImg(self, path, channels, w, h):
        if channels == 1:
            img = cv2.imread(path,0)
        else: 
            img = cv2.imread(path)
        return cv2.resize(img,(h,w),interpolation=cv2.INTER_LINEAR)

    def oneHot(self,y):
        return np.eye(10)[y]

    def getBatch(self,trainOrTest, batchSize = 5, channels=3,w=160,h=120):
        # get data paths

            # Shuffle - to do



            if trainOrTest == "train":
                random.shuffle(self.trainPaths)

                while True:

                    xBatch = []
                    yBatch = []
                    for i in range(self.batchIdxTrain,min(self.batchIdxTrain+batchSize,len(self.trainPaths))):
                        xBatch.append(self.loadImg(self.trainPaths[i],channels=self.c,w=self.w,h=self.h))
                        yBatch.append(self.oneHot(int(self.trainPaths[i].split("/")[2][1])))

                    xBatch = np.array(xBatch)
                    yBatch = np.array(yBatch)
                    if self.batchIdxTrain + batchSize >= len(self.trainPaths):
                        self.batchIdxTrain = 0
                        self.finished = 1
                        print("Finished epoch")
                    else:
                        self.batchIdxTrain += batchSize
                    yield xBatch, yBatch

            else:
                random.shuffle(self.testPaths)

                while True:

                    xBatch = []
                    yBatch = []
                    for i in range(self.batchIdxTest,min(self.batchIdxTest+batchSize,len(self.testPaths))):
                        xBatch.append(self.loadImg(self.testPaths[i],channels=self.c,w=self.w,h=self.h))
                        yBatch.append(self.oneHot(int(self.testPaths[i].split("/")[2][1])))

                    xBatch = np.array(xBatch)
                    yBatch = np.array(yBatch)

                    if self.batchIdxTest + batchSize >= len(self.testPaths):
                        self.batchIdxTest = 0
                        self.finished = 1
                    else:
                        self.batchIdxTest += batchSize
                    yield xBatch, yBatch



if __name__ == "__main__":
    nClasses = 10
    w,h = 640/4, 480/4
    train = 1
    dataLoad = dataLoader()
    train = dataLoad.getBatch("train",batchSize=10)
    test = dataLoad.getBatch("test",batchSize=10)
    while dataLoad.finished == False:
        x, y = test.next()


            


                        


    


    

        
   
    
