import cv2,os,sys, glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import loadData
from model import model0
import matplotlib.cm as cm
sys.path.append("/Users/matt/misc/tfFunctions/")
import paramCount
from dice import dice

def showBatch(batchX,y,yPred,wp,figsize=(15,15)):
    outSize = 200 # width
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w,c)
    plt.subplot(121)
    plt.imshow(batchX[:,:,:3])
    plt.subplot(122)
    plt.imshow(batchX[:,:,3],cmap=cm.gray)
    plt.title("yPred = {0}, y = {1}".format(yPred,y))
    pdb.set_trace()
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

def nodes(batchSize,inSize,trainOrTest,initFeats,incFeats,nDown,num_epochs):
    if trainOrTest == "train":
        csvPath = "train.csv"
        print("Training")
        shuffle = True
    elif trainOrTest == "fit":
        csvPath = "test.csv"
        num_epochs = 1
        batchSize = 1
        shuffle = False
    X,Y,xPath = loadData.read(csvPath=csvPath,
            batchSize=batchSize,
            inSize=inSize,
            shuffle=shuffle,
            num_epochs = num_epochs
            ) #nodes
    #XTestPath,XTest = loadData.testRead([inSize[0],inSize[1]])
    is_training = tf.placeholder(tf.bool)
    YPred = model0(X,is_training=is_training,nDown=nDown,initFeats=initFeats,featsInc=incFeats)
    loss = lossFn(Y,YPred)
    learningRate = tf.placeholder(tf.float32)
    trainOp = trainer(loss,learningRate)
    saver = tf.train.Saver()

    if trainOrTest == "fit":
        YPred = tf.nn.softmax(Y)

    return saver,xPath,X,Y,YPred,loss,is_training,trainOp,learningRate#, XTestPath, XTest

if __name__ == "__main__":
    import pdb
    nEpochs = 20
    flags = tf.app.flags
    FLAGS = flags.FLAGS 
    flags.DEFINE_float("lr",0.001,"Initial learning rate.")
    flags.DEFINE_float("lrD",1.00,"Learning rate division rate applied every epoch. (DEFAULT - nothing happens)")
    flags.DEFINE_integer("inSize",256,"Size of input image")
    flags.DEFINE_integer("initFeats",32,"Initial number of features.")
    flags.DEFINE_integer("incFeats",0,"Number of features growing.")
    flags.DEFINE_integer("nDown",5,"Number of blocks going down.")
    flags.DEFINE_integer("bS",10,"Batch size.")
    flags.DEFINE_integer("load",0,"Load saved model.")
    flags.DEFINE_integer("trainAll",0,"Train on all data.")
    flags.DEFINE_integer("fit",0,"Fit training data.")
    flags.DEFINE_integer("fitTest",0,"Fit actual test data.")
    flags.DEFINE_integer("show",0,"Show for sanity.")
    flags.DEFINE_integer("nEpochs",200,"Number of epochs to train for.")
    batchSize = FLAGS.bS
    load = FLAGS.load
    if FLAGS.fit == 1 or FLAGS.fitTest == 1:
        load = 1
    specification = "{0}_{1:.6f}_{2}_{3}_{4}_{5}".format(FLAGS.bS,FLAGS.lr,FLAGS.inSize,FLAGS.initFeats,FLAGS.incFeats,FLAGS.nDown)
    print("Specification = {0}".format(specification))
    modelDir = "models/" + specification + "/"
    imgPath = modelDir + "imgs/"
    if not FLAGS.fit:
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)
            os.mkdir(imgPath)
    savePath = modelDir + "model.tf"
    trCount = teCount = 0
    tr = "train"
    what = ["train","test"]

    if FLAGS.fit == 1:
        what = ["fit"]
    for trTe in what:
        if trTe in ["fit","test"]:
            load = 1
            FLAGS.nEpochs = 1
            tf.reset_default_graph()
        saver,XPath,X,Y,YPred,loss,is_training,trainOp,learningRate = nodes(
                batchSize=FLAGS.bS,
                trainOrTest=trTe,
                inSize = [FLAGS.inSize,FLAGS.inSize],
                initFeats=FLAGS.initFeats,
                incFeats=FLAGS.incFeats,
                nDown=FLAGS.nDown,
                num_epochs=FLAGS.nEpochs
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
                        _, summary,x,y,yPred,xPath = sess.run([trainOp,merged,X,Y,YPred,XPath],feed_dict={is_training:True,learningRate:FLAGS.lr})

                        trCount += batchSize
                        count += batchSize
                        trWriter.add_summary(summary,trCount)
                        if count % 100 == 0:
                            print("Seen {0} examples".format(count))
                            x = x[[0],:]
                            y = y[[0],:]
                            yPred = yPred[[0],:]
                            showBatch(x,y,yPred,wp="train.jpg")
                            if FLAGS.show == 1:
                                pass
                        #        x = x[[0],:]
                        #        y = y[[0],:]
                        #        yPred = yPred[[0],:]
                        #        showBatch(x,y,yPred,"{0}_{1}_.jpg".format(imgPath,trTe))
                        #        # Random test image
                        #        xTest,xTestPath = sess.run([XTest,XTestPath])
                        #        yPred = YPred.eval(feed_dict={X:xTest,is_training:False})


                        if count % 10000 == 0:
                            print("Saving")
                            saver.save(sess,savePath)

                    elif trTe == "fit":
                        x, yPred,fp = sess.run([X,YPred,xPath],feed_dict={is_training:False})
                        showBatch(x,yPred,fp)
                        pdb.set_trace()
                        for i in range(x.shape[0]):

                            wp = fp[i][0].replace("head_","m4_")
                            im = (yPred[i][:,:,::-1]*255.0).astype(np.uint8)
                            im = cv2.resize(im,(500,500))
                            count += 1
                            
                            cv2.imwrite(wp,im)
                            if np.random.uniform() < 0.03:
                                print(count)
                                showBatch(x,yPred,yPred,"{0}_{1}.jpg".format(imgPath,trTe))
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
            print("Saving in {0}".format(savePath))
            lrC = FLAGS.lr
            FLAGS.lr /= FLAGS.lrD
            print("Dropped learning rate from {0} to {1}".format(lrC,FLAGS.lr))
            if trTe == "train":
                print("Saving")
                saver.save(sess,savePath)
            sess.close()
                    
