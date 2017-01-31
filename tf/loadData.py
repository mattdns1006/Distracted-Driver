import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import cv2, glob
import pdb

def aug(img,inSize):
    cropSize = int(0.9*inSize[0])
    img = tf.random_crop(img,[cropSize,cropSize])
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

def show(img):
    plt.imshow(img,cmap=cm.gray)
    plt.show()

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
    return decodedImg

def read(csvPath,batchSize,inSize,num_epochs,shuffle,feats=4):
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
    maskSize = list(inSize)
    maskSize += [1]
    inSize += [3]

    Q = tf.FIFOQueue(128,[tf.float32,tf.float32,tf.float32,tf.string],shapes=[inSize,maskSize,[10],[1]])
    enQ = Q.enqueue([x,mask,label,xPathRe])
    QR = tf.train.QueueRunner(
            Q,
            [enQ]*32,
            Q.close(),
            Q.close(cancel_pending_enqueues=True)
            )
    tf.train.add_queue_runner(QR) 
    dQ = Q.dequeue()
    X,XM,Y,path = tf.train.batch(dQ,batchSize,16,allow_smaller_final_batch=True)
    if feats == 4:
        # rgb + mask
        XC = tf.concat(3,[X,XM])
    else:
        # normal rgb img
        XC = X

    return XC, Y, path

if __name__ == "__main__":
    inSize = [200,200]
    makeCsv()
    XC, Y, path = read(csvPath="train.csv",batchSize=4,inSize=inSize,num_epochs=10,shuffle=True)
    image = tf.placeholder(tf.float32)
    augImg = aug(image,inSize)

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
                for i in xrange(10):
                    augIm = augImg.eval(feed_dict={image:im})
                    show(augIm)
                    pdb.set_trace()
                if coord.should_stop():
                    break
        except Exception,e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
