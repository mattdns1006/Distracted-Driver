import tensorflow as tf
import numpy as np
import sys
sys.path.append("/home/msmith/misc/tfFunctions")
from batchNorm2 import bn

def imSum(name,img,max_outputs=10):
    tf.summary.image(name,img,max_outputs)

def W(shape,weightInit,scale=0.05):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        fw, fh, nIn, nOut = shape
        fan_in = fw*fh*nIn # Number of weights for one output neuron in nOut
        fan_out = nOut #  
    else:
        raise ValueError, "Not a valid shape"

    if weightInit == "uniform":
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "normal":
        init = tf.random_normal(shape,mean=0,stddev=scale)

    elif weightInit == "lecun_uniform":
        scale = np.sqrt(3.0/(fan_in))
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "glorot_normal":
        scale = np.sqrt(2.0/(fan_in+fan_out))
        init = tf.random_normal(shape,mean=0,stddev=scale)

    elif weightInit == "glorot_uniform":
        scale = np.sqrt(6.0/(fan_in+fan_out))
        init = tf.random_uniform(shape,minval=-scale,maxval=scale)

    elif weightInit == "zeros":
        init = tf.zeros(shape)
    else:
        raise ValueError, "{0} not a valid weight intializer.".format(weightInit)

    return tf.Variable(init)

def B(shape):
    return tf.Variable(tf.constant(0.0,shape=[shape]))

def convolution2d(inTensor,inFeats,outFeats,filterSize,stride=1):
    with tf.name_scope("conv2d"):
        with tf.name_scope("w"):
            weight = W([filterSize,filterSize,inFeats,outFeats],"lecun_uniform")
        with tf.name_scope("b"):
            bias = B(outFeats)
        with tf.name_scope("conv"):
            out = tf.nn.conv2d(inTensor,weight,strides=[1,stride,stride,1],padding='SAME') + bias
    return out

def dilated_convolution2d(inTensor,inFeats,outFeats,filterSize,dilation):
    with tf.name_scope("dilation"):
        with tf.name_scope("w"):
            weight = W([filterSize,filterSize,inFeats,outFeats],"lecun_uniform")
        with tf.name_scope("b"):
            bias = B(outFeats)
        with tf.name_scope("conv"):
            out = tf.nn.atrous_conv2d(value=inTensor,filters=weight,rate=dilation,padding='SAME') + bias
    return out 


def model0(x,is_training,initFeats=16,featsInc=0,nDown=6,filterSize=3,decay=0.95,dropout=1.0):
    imSum("x",x)
    af = tf.nn.relu
    print(x.get_shape())
    dilation = 2
    with tf.variable_scope("convIn"):
        x1 = af(bn(convolution2d(x,4,initFeats,3,stride=2),is_training=is_training,name="bn_0",decay=decay))
        #for i in xrange(initFeats):
        #    f = tf.split(3,initFeats,x1)[i]
        #    imSum(f,"f{0}".format(i))


    for block in range(nDown):
        if block == 0:
            inFeats = initFeats 
            outFeats = initFeats + featsInc
        else:
            inFeats = outFeats 
            outFeats = outFeats + featsInc
        with tf.variable_scope("block_down_{0}".format(block)):
	    x2 = af(bn(convolution2d(x1,inFeats,outFeats,1,stride=1),is_training=is_training,name="bn_{0}_0".format(nDown),decay=decay))
	    x3 = af(bn(convolution2d(x1,inFeats,outFeats,2,stride=1),is_training=is_training,name="bn_{0}_1".format(nDown),decay=decay))
	    x4 = af(bn(dilated_convolution2d(x1,inFeats,outFeats,2,dilation=dilation),is_training=is_training,name="bn_{0}_2".format(nDown)))
	    x5 = af(bn(dilated_convolution2d(x1,inFeats,outFeats,2,dilation=dilation-1),is_training=is_training,name="bn_{0}_3".format(nDown)))
            x6 = bn(x5 + x4 + x3 + x2,is_training=is_training,name="bn_{0}_4".format(nDown))
	    #x1 = tf.nn.max_pool(x6,[1,3,3,1],[1,2,2,1],"SAME")
    	    print(x6.get_shape())
            x1,_,_ = tf.nn.fractional_max_pool(x6,[1.0,1.5,1.5,1.0],True,True)
            dilation += 2

    with tf.variable_scope("reshape"):

        sizeBeforeReshape = x1.get_shape().as_list()
    	print(x1.get_shape())
        nFeats = sizeBeforeReshape[1]*sizeBeforeReshape[2]*sizeBeforeReshape[3]
        flatten = tf.reshape(x1, [-1, nFeats])
        flatten = tf.nn.dropout(flatten,dropout)
    	print(flatten.get_shape())

    with tf.variable_scope("lin"):
        nLin1 = 256 
        wLin1 = W([nFeats,nLin1],"lecun_uniform")
        bLin1 = B(nLin1)
        linear = af(bn(tf.matmul(flatten,wLin1) + bLin1,name="bn7",is_training=is_training))
        linear = tf.nn.dropout(linear,dropout)
    	print(linear.get_shape())

    with tf.variable_scope("out"):
        nLin2 = 10 
        wLin2 = W([nLin1,nLin2],"lecun_uniform")
        bLin2 = B(nLin2)
        yPred = tf.matmul(linear,wLin2) + bLin2
        print(yPred.get_shape())
    return yPred

def resNet(x,is_training,initFeats=16,featsInc=0,nDown=6,filterSize=3,decay=0.95,dropout=1.0):
    af = tf.nn.elu
    print(x.get_shape())
    with tf.variable_scope("convIn"):
        x1 = af(bn(convolution2d(x,4,initFeats,3,stride=1),is_training=is_training,name="bn_0",decay=decay))
    inFeats = outFeats = initFeats 
    for block in range(nDown):

        with tf.variable_scope("res_{0}".format(block)):
            with tf.variable_scope("big_shortcut_{0}".format(block)):
                bigShortcut = af(bn(convolution2d(x1,outFeats,outFeats,1,stride=1),is_training=is_training,name="bn",decay=decay))
            for miniBlock in xrange(3):
                with tf.variable_scope("res_{0}_{1}".format(block,miniBlock)):
                    shortcut = af(bn(convolution2d(x1,outFeats,outFeats,1,stride=1),is_training=is_training,name="bn_0".format(block),decay=decay))
                    forward = af(bn(convolution2d(x1,outFeats,outFeats/2,3,stride=1),is_training=is_training,name="bn_1".format(block),decay=decay))
                    forward = af(bn(convolution2d(forward,outFeats/2,outFeats,3,stride=1),is_training=is_training,name="bn_2".format(block),decay=decay))
                    out = af(bn(shortcut + forward,is_training=is_training,name="bn_3".format(block)))
                    x1 = out
                    print(x1.get_shape())
            print(x1.get_shape())
            with tf.variable_scope("transfer_{0}".format(block)):
                out = af(bn(out + bigShortcut,is_training=is_training,name="bn_0".format(block)))
                inFeats = outFeats 
                outFeats += featsInc
                x1 = af(bn(convolution2d(out,inFeats,outFeats,3,stride=2),is_training=is_training,name="bn_1".format(block),decay=decay))

    with tf.variable_scope("reshape"):

        sizeBeforeReshape = x1.get_shape().as_list()
    	print(x1.get_shape())
        nFeats = sizeBeforeReshape[1]*sizeBeforeReshape[2]*sizeBeforeReshape[3]
        flatten = tf.reshape(x1, [-1, nFeats])
        flatten = tf.nn.dropout(flatten,dropout)
    	print(flatten.get_shape())

    with tf.variable_scope("lin"):
        nLin1 = 256 
        wLin1 = W([nFeats,nLin1],"lecun_uniform")
        bLin1 = B(nLin1)
        linear = af(bn(tf.matmul(flatten,wLin1) + bLin1,name="bn7",is_training=is_training))
        linear = tf.nn.dropout(linear,dropout)
    	print(linear.get_shape())

    with tf.variable_scope("out"):
        nLin2 = 10 
        wLin2 = W([nLin1,nLin2],"lecun_uniform")
        bLin2 = B(nLin2)
        yPred = tf.matmul(linear,wLin2) + bLin2
        print(yPred.get_shape())
    return yPred


if __name__ == "__main__":
    import pdb
    import numpy as np
    X = tf.placeholder(tf.float32,shape=[None,256,256,4])
    is_training = tf.placeholder(tf.bool,name="is_training")
    Y = model0(X,is_training=is_training,initFeats=16,featsInc=32,nDown=6)
    weights = [w for w in tf.trainable_variables() if "/w/" in w.name]
    pdb.set_trace()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in xrange(10):
            x = np.random.rand(1,256,256,4)
            y_ = sess.run([Y],feed_dict={X:x,is_training.name:True})[0]
            print(y_.shape)
            if i == 9:
                pdb.set_trace()


