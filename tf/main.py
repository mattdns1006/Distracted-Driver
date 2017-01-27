import tensorflow as tf
import matplotlib.pyplot as plt
from loadData import *
import sys, os
from model import model0
sys.path.append("/Users/matt/misc/tfFunctions/")

def varSum(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

def lossFn(y,yPred):
    return tf.reduce_mean(tf.square(tf.sub(y,yPred)))

def trainer(lossFn, learningRate):
    return tf.train.AdamOptimizer(learningRate).minimize(lossFn)

def nodes(batchSize,inSize,num_epochs,trainOrTest):
    if trainOrTest == "train":
        csvPath = "train.csv"
        print("Training")
        is_training = True
        shuffle = True
    elif trainOrTest == "test":
        print("Testing")
        csvPath = "testCV.csv"
        is_training = False
        shuffle = False
    elif trainOrTest == "fit":
        print("Fitting")
        csvPath = "test.csv"
        is_training = False
        shuffle = True 
    X,Y,path = read(csvPath=csvPath,
            batchSize=batchSize,
            inSize=inSize,
            num_epochs=num_epochs,
            shuffle=shuffle) #nodes
    return X,Y,path

if __name__ == "__main__":
    import pdb
    flags = tf.app.flags
    FLAGS = flags.FLAGS 
    flags.DEFINE_float("lr",0.001,"Initial learning rate.")
    flags.DEFINE_float("lrD",1.00,"Learning rate division rate applied every epoch. (DEFAULT - nothing happens)")
    flags.DEFINE_integer("sf",256,"Size of input image")
    flags.DEFINE_integer("initFeats",64,"Initial number of features.")
    flags.DEFINE_integer("incFeats",0,"Number of features growing.")
    flags.DEFINE_integer("nDown",6,"Number of blocks going down.")
    flags.DEFINE_integer("nDense",2,"Number of dense layers.")
    flags.DEFINE_integer("denseFeats",128,"Number of units in dense layers.")
    flags.DEFINE_integer("bS",10,"Batch size.")
    flags.DEFINE_integer("load",0,"Load saved model.")
    flags.DEFINE_integer("trainAll",0,"Train on all data.")
    flags.DEFINE_integer("fit",0,"Fit test data.")
    flags.DEFINE_integer("show",0,"Show for sanity.")
    flags.DEFINE_integer("nEpochs",2,"Number of epochs to train for.")
    batchSize = FLAGS.bS
    load = FLAGS.load
    specification = "{0}_{1}_{2}_{3}_{4}_{5:.6f}_{6}".format(FLAGS.sf,FLAGS.initFeats,FLAGS.incFeats,FLAGS.nDown,FLAGS.nDense,FLAGS.lr,FLAGS.bS)
    print("Specification = {0}".format(specification))
    modelDir = "models/" + specification + "/"
    if not FLAGS.fit:
        if not os.path.exists(modelDir):
            os.mkdir(modelDir)
    savePath = modelDir + "model0.tf"

    load = FLAGS.load 
    what = "train"
    nEpochs = FLAGS.nEpochs
    if FLAGS.fit == 1:
        what = "fit"
        load = 1
        nEpochs = 1

    inSize = [FLAGS.sf,FLAGS.sf]
    batchSize = FLAGS.bS
    is_training = tf.placeholder(tf.bool)
    X, Y, Xpath = nodes(batchSize=batchSize,inSize=inSize,num_epochs=1,trainOrTest=what)
    YPred = model0(X,is_training=is_training)
    mse = lossFn(Y,YPred)
    varSum(mse)
    trainOp = trainer(mse,FLAGS.lr)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    count = 0
    with tf.Session() as sess:
        if load == 1:
            saver.restore(sess,savePath)
        else:
            tf.initialize_all_variables().run()
        tf.local_variables_initializer().run()
        train_writer = tf.summary.FileWriter("summary/{0}/train/".format(specification),sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            while True:
                if what == "train":
                    _, summary = sess.run([trainOp,merged],feed_dict={is_training:True})
                    count += batchSize
                    train_writer.add_summary(summary,count)
                elif what == "fit":
                    x,y,xPath = sess.run([X,Y,XPath],feed_dict={is_training:False})
                    pdb.set_trace()
                if coord.should_stop():
                    break
        except Exception,e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
        saver.save(sess,savePath)
        sess.close()
