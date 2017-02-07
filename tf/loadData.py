import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import cv2, glob, os, sys
sys.path.append("/home/msmith/misc/py")
from removeFiles import removeFiles
from model import imSum
import pdb, time
from tqdm import tqdm

def brightness(img,max_delta=30):
    return tf.image.random_brightness(img,max_delta=max_delta)

def contrast(img,lower=0.8,upper=1.2):
    return tf.image.random_contrast(img,lower=lower,upper=upper)

def aug(img,inSize):
    e = np.random.normal(0.95,0.01,1)[0]
    cropSizeX = int(e*inSize[0])
    cropSizeY = int(e*inSize[1])
    img = tf.random_crop(img,[cropSizeX,cropSizeY,inSize[2]])
    img = tf.image.resize_images(img,inSize[:2])
    #img = contrast(img)
    return img

def oneHot(idx,nClasses=10):
    oh = tf.sparse_to_dense(idx,output_shape = [nClasses], sparse_values = 1.0)
    return oh

def makeCsv():
    for i in ["train"]:
        mask = glob.glob("../{0}/*/*_mask.jpg".format(i))
        path = [x.replace("_mask","") for x in mask]
        label = [int(x.split("/")[2][1]) for x in path]
        csv = pd.DataFrame({"path":path,"pathMask":mask,"label":label})
        nObs = csv.shape[0]
        rIdx = np.random.permutation(nObs)
        csv = csv.reindex(rIdx)
        csv.reset_index(drop=1,inplace=1)
        csv.to_csv("{0}.csv".format(i),index=0)

    for i in ["test"]:
        mask = glob.glob("../{0}/*_mask.jpg".format(i))
        path = [x.replace("_mask","") for x in mask]
        csv = pd.DataFrame({"path":path,"pathMask":mask,"label":0})
        csv.to_csv("{0}.csv".format(i),index=0)

def augment():
    train = pd.read_csv("trainCV.csv")
    savePath = "augmented/"
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    count = 0
    newWidth = newHeight = 224
    maxShift = int(0.05*newWidth)
    nObs = train.shape[0]
    df = []
    for i in tqdm(range(nObs)):
        row = train.ix[i]
        im, mask = [cv2.imread(x) for x in [row.path, row.pathMask]]
        im, mask = [cv2.resize(x,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC) for x in [im,mask]]
        for j in xrange(20):
            imC = im.copy()
            maskC = mask.copy()
            if j != 0:
                #aug
                angle = np.random.uniform(-6,6)
                scale = np.random.normal(1.0,0.1)
                M = cv2.getRotationMatrix2D((newWidth/2,newHeight/2),angle,scale=scale)
                M[0,1] += np.random.normal(0,0.1)
                M[:,2] = np.random.normal(0,maxShift,2)
                imC, maskC = [cv2.warpAffine(x,M,(newWidth,newHeight),borderMode = 0,flags=cv2.INTER_CUBIC) for x in [imC,maskC]]
            wp = savePath + str(count) + ".jpg"
            wp = os.path.abspath(wp)
            wpMask = wp.replace(".jpg","_mask.jpg")
            cv2.imwrite(wp,imC)
            cv2.imwrite(wpMask,maskC)
            df.append([row.label,wp,wpMask])
            count += 1
            #time.sleep(1)
    df = pd.DataFrame(df)
    df.columns = ["label","path","pathMask"]
    nObs = df.shape[0]
    rIdx = np.random.permutation(nObs)
    df= df.reindex(rIdx)
    df.reset_index(drop=1,inplace=1)
    df.to_csv("trainAugCV.csv",index=0)


def show(img):
    plt.imshow(img,cmap=cm.gray)
    plt.show()
    plt.close()

def showBatch(batchX,batchXM, figsize=(15,15)):
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w,c)
    batchXM = batchXM.reshape(n*h,w)
    plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.imshow(batchX)
    plt.subplot(122)
    plt.imshow(batchXM,cmap=cm.gray)
    plt.show()

def getImg(path,size):
    imageBytes = tf.read_file(path)
    decodedImg = tf.image.decode_jpeg(imageBytes)
    decodedImg = tf.image.resize_images(decodedImg,size)
    decodedImg = tf.cast(decodedImg,tf.float32)
    decodedImg = tf.mul(decodedImg,1/255.0)
    decodedImg = tf.image.per_image_standardization(decodedImg)
    #decodedImg = tf.sub(decodedImg,tf.reduce_mean(decodedImg))
    return decodedImg

def read(csvPath,batchSize,inSize,num_epochs,shuffle,augment=1,feats=4):
    csv = tf.train.string_input_producer([csvPath],num_epochs=num_epochs,shuffle=shuffle)
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csv)
    defaults = [tf.constant([],shape=[1],dtype = tf.int32),
                tf.constant([],dtype = tf.string), 
                tf.constant([],dtype = tf.string) ]
    label,xPath,maskPath = tf.decode_csv(v,record_defaults = defaults)
    label = oneHot(idx=label)
    xPathRe = tf.reshape(xPath,[1])
    x = getImg(xPath,inSize)
    mask = getImg(maskPath,inSize)

    if feats == 4:
        x = tf.concat(2,[x,mask])
        inSize += [4]
    else:
        # normal rgb img
        inSize += [3] 

    if augment == 1:
        x = aug(x,inSize)

    Q = tf.FIFOQueue(128,[tf.float32,tf.float32,tf.string],shapes=[inSize,[10],[1]])
    enQ = Q.enqueue([x,label,xPathRe])
    QR = tf.train.QueueRunner(
            Q,
            [enQ]*24,
            Q.close(),
            Q.close(cancel_pending_enqueues=True)
            )
    tf.train.add_queue_runner(QR) 
    dQ = Q.dequeue()
    X,Y,path = tf.train.batch(dQ,batchSize,16,allow_smaller_final_batch=True)

    return X, Y, path

if __name__ == "__main__":
    inSize = [200,200]
    #makeCsv()
    augment()
    train = pd.read_csv("train.csv")[:1]
    train.to_csv("trainEg.csv",index=0)
    XC, Y, path = read(csvPath="trainEg.csv",batchSize=1,inSize=inSize,num_epochs=100,shuffle=True,augment=1,feats=4)
    image = tf.placeholder(tf.float32)

    def eg():
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            tf.initialize_local_variables().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            count = 0
            try:
                while True:
                    p, xc, y = sess.run([path,XC,Y])
                    print(p)
                    im = xc[0,:,:,:3]
                    for i in xrange(1):
                        print(im.shape)
                        show(im)
                    if coord.should_stop():
                        break
            except Exception,e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
