import cv2,os,sys, glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import loadData
from model import model0
import matplotlib.cm as cm
sys.path.append("/Users/matt/misc/tfFunctions/")
import paramCount
from dice import dice

class Decode():
    def __init__(self):
        self.names = {0: "safe driving",
                        1: "texting - right",
                        2: "talking on the phone - right",
                        3: "texting - left",
                        4: "talking on the phone - left",
                        5: "operating the radio",
                        6: "drinking",
                        7: "reaching behind",
                        8: "hair and makeup",
                        9: "talking to passenger"}

    def get(self,label):
        return self.names[label]


def showBatch(batchX,y,yPred,wp,figsize=(15,15)):
    outSize = 200 # width
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w,c)
    plt.subplot(121)
    plt.imshow(batchX[:,:,:3])
    plt.subplot(122)
    plt.imshow(batchX[:,:,3],cmap=cm.gray)
    plt.title("yPred = {0}, y = {1}".format(yPred,y))
    plt.savefig(wp)
    plt.close()

def varSummary(var,name):
    with tf.name_scope('summary'):
        tf.summary.scalar(name, var)
        tf.summary.histogram(name, var)

def imgSummary(name,img):
    tf.summary.image(name,img)

def lossFn(y,yPred):
    with tf.variable_scope("loss"):
        y = tf.argmax(y,1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=yPred))
        varSummary(loss,"loss")
    with tf.variable_scope("accuracy"):
        correct = tf.equal(tf.argmax(yPred,1),y)
        acc = tf.reduce_mean(tf.cast(correct,tf.float32))
        varSummary(acc,"accuracy")
    return loss

def trainer(lossFn, learningRate):
    return tf.train.AdamOptimizer(learningRate).minimize(lossFn)

def nodes(batchSize,inSize,trainOrTest,initFeats,incFeats,nDown,num_epochs,augment):
    if trainOrTest == "train":
        csvPath = "trainCV.csv"
        print("Training on subset.")
        shuffle = True
    elif trainOrTest == "trainAll":
        csvPath = "train.csv"
        print("Training on all.")
        shuffle = True
    elif trainOrTest == "test":
        csvPath = "testCV.csv"
        print("Testing on validation set")
        shuffle = True
        num_epochs = 1
    elif trainOrTest == "fit":
        csvPath = "test.csv"
        num_epochs = 1
        shuffle = False
    X,Y,xPath = loadData.read(csvPath=csvPath,
            batchSize=batchSize,
            inSize=inSize,
            shuffle=shuffle,
            num_epochs = num_epochs,
            augment = augment
            ) #nodes
    is_training = tf.placeholder(tf.bool)
    drop = tf.placeholder(tf.float32)
    YPred = model0(X,is_training=is_training,nDown=nDown,initFeats=initFeats,featsInc=incFeats,dropout=drop)
    loss = lossFn(Y,YPred)
    learningRate = tf.placeholder(tf.float32)
    trainOp = trainer(loss,learningRate)
    saver = tf.train.Saver()

    if trainOrTest == "fit":
        YPred = tf.nn.softmax(YPred)

    return saver,xPath,X,Y,YPred,loss,is_training,trainOp,learningRate, drop

if __name__ == "__main__":
    import pdb
    nEpochs = 3 
    flags = tf.app.flags
    FLAGS = flags.FLAGS 
    flags.DEFINE_float("lr",0.001,"Initial learning rate.")
    flags.DEFINE_float("feats",4,"Use the 4th feature i.e. mask?.")
    flags.DEFINE_float("lrD",1.00,"Learning rate division rate applied every epoch. (DEFAULT - nothing happens)")
    flags.DEFINE_integer("inSize",256,"Size of input image")
    flags.DEFINE_integer("initFeats",16,"Initial number of features.")
    flags.DEFINE_integer("incFeats",32,"Number of features growing.")
    flags.DEFINE_float("drop",0.8,"Keep prob for dropout.")
    flags.DEFINE_integer("aug",1,"Augment.")
    flags.DEFINE_integer("nDown",7,"Number of blocks going down.")
    flags.DEFINE_integer("bS",10,"Batch size.")
    flags.DEFINE_integer("load",0,"Load saved model.")
    flags.DEFINE_integer("trainAll",0,"Train on all data.")
    flags.DEFINE_integer("fit",0,"Fit training data.")
    flags.DEFINE_integer("show",0,"Show for sanity.")
    flags.DEFINE_integer("nEpochs",20,"Number of epochs to train for.")
    flags.DEFINE_integer("test",0,"Just test.")
    batchSize = FLAGS.bS
    load = FLAGS.load
    if FLAGS.fit == 1 or FLAGS.test == 1:
        load = 1
    specification = "{0}_{1:.6f}_{2}_{3}_{4}_{5}_{6:.3f}_{7}".format(FLAGS.bS,FLAGS.lr,FLAGS.inSize,FLAGS.initFeats,FLAGS.incFeats,FLAGS.nDown,FLAGS.drop,FLAGS.aug)
    print("Specification = {0}".format(specification))
    modelDir = "models/" + specification + "/"
    imgPath = modelDir + "imgs/"
    if not FLAGS.fit:
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)
            os.mkdir(imgPath)
    savePath = modelDir + "model.tf"
    trCount = teCount = 0
    trTe = "train"
    assert FLAGS.test + FLAGS.trainAll + FLAGS.fit in [0,1], "Only one of trainAll, test or fit == 1"
    assert FLAGS.feats in [3,4], "feats must be either 3 (RGB) or 4(RBG+mask)"
    what = ["train"]
    aug = FLAGS.aug
    if FLAGS.test == 1:
        what = ["test"]
        load = 1
        aug = 0
        FLAGS.nEpochs = 1
    if FLAGS.fit == 1:
        what = ["fit"]
        print("Initializing dataframe to save into.")
        df = []
        FLAGS.nEpochs = 1
        load = 1
        aug = 0
        trTe = "fit"
    if FLAGS.trainAll == 1:
        what = ["train"]
    decode = Decode()

    for trTe in what:
        tf.reset_default_graph()
        saver,XPath,X,Y,YPred,loss,is_training,trainOp,learningRate,drop = nodes(
            batchSize=FLAGS.bS,
            trainOrTest=trTe,
            inSize = [FLAGS.inSize,FLAGS.inSize],
            initFeats=FLAGS.initFeats,
            incFeats=FLAGS.incFeats,
            nDown=FLAGS.nDown,
            num_epochs=FLAGS.nEpochs,
            augment = aug
            )

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

        merged = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if load == 1:
                print("Restoring {0}.".format(specification))
                saver.restore(sess,savePath)
            else:
                tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            trWriter = tf.summary.FileWriter("summary/{0}/train/".format(specification),sess.graph)
            teWriter = tf.summary.FileWriter("summary/{0}/test/".format(specification),sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            count = 0
            try:
                while True:
                    if trTe in ["train","trainAll"]:
                        _, summary,x,y,yPred,xPath = sess.run([trainOp,merged,X,Y,YPred,XPath],feed_dict={is_training:True,
                                                                                                            drop:FLAGS.drop,
                                                                                                            learningRate:FLAGS.lr})

                        trCount += batchSize
                        count += batchSize
                        trWriter.add_summary(summary,trCount)
                        if count % 100 == 0:
                            print("Seen {0} examples".format(count))
                            x = x[[0],:]
                            y = y[[0],:].argmax()
                            yPred = yPred[[0],:].argmax()
                            showBatch(x,y,yPred,wp="{0}/train.jpg".format(imgPath))
                            if FLAGS.show == 1:
                                pass

                        if count % 10000 == 0:
                            print("Saving")
                            saver.save(sess,savePath)
                        if count > 30000:
                            print("Finished training cba")
                            break
                    elif trTe == "test":
                        summary,x,y,yPred,xPath = sess.run([merged,X,Y,YPred,XPath],feed_dict={is_training:False,drop:FLAGS.drop})
                        teCount += batchSize
                        teWriter.add_summary(summary)
                        if teCount % 100 == 0:
                            print("Seen {0} examples".format(teCount))
                            x = x[[0],:]
                            y = y[[0],:].argmax()
                            yPred = yPred[[0],:].argmax()
                            showBatch(x,y,yPred,wp="{0}/test.jpg".format(imgPath))

                    elif trTe == "fit":
                        x, yPred,fp = sess.run([X,YPred,XPath],feed_dict={is_training:False,drop:FLAGS.drop})
                        count += x.shape[0]
                        for i in xrange(x.shape[0]):
                            row = fp[i].tolist() + yPred[i].tolist()
                            df.append(row) 
                        if count % 600 == 0:
                            print(count)
                            xeg = x[[0],:]
                            yeg = "NA" 
                            yPredeg = decode.get(yPred[[0],:].argmax())
                            showBatch(xeg,yeg,yPredeg,wp="{0}/fit.jpg".format(imgPath))

                    else:
                        break

                    if coord.should_stop():
                        break
            except Exception,e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            print("Finished! Seen {0} examples".format(count))

            if trTe == "train":
                lrC = FLAGS.lr
                FLAGS.lr /= FLAGS.lrD
                print("Dropped learning rate from {0} to {1}".format(lrC,FLAGS.lr))
                print("Saving in {0}".format(savePath))
                saver.save(sess,savePath)
            elif trTe == "fit":
                Df = pd.DataFrame(df)
                Df.columns = ["img","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"]
                Df["img"] = Df.img.apply(lambda x: x.split("/")[-1])
                Df.to_csv("submission.csv",index=0)
                print("Written submission file.")

            sess.close()

                    
