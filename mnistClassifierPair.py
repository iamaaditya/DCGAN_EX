import tensorflow as tf
import numpy as np
from scipy import misc

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def convLayer(input,in_features,out_features,pool_dim):
    filters = tf.Variable(tf.truncated_normal([5,5,in_features,out_features],stddev=0.1))
    biases = tf.Variable(tf.truncated_normal([out_features],stddev=0.1))
    convolved = tf.nn.relu(tf.nn.conv2d(input,filters,strides=[1,1,1,1],padding='SAME'))
    pooled = tf.nn.max_pool(convolved,ksize=[1,pool_dim,pool_dim,1],strides=[1,pool_dim,pool_dim,1],padding='SAME')
    return pooled

def fcLayer(input,in_dim,out_dim,relu=True):
    weight = tf.Variable(tf.truncated_normal([in_dim,out_dim],stddev=0.1))
    bias = tf.Variable(tf.truncated_normal([out_dim],stddev=0.1))
    if(relu):
        return tf.nn.relu(tf.matmul(input,weight)+bias)
    else:
        return tf.matmul(input,weight)+bias

x = tf.placeholder(tf.float32,shape=[None,28*28])
x_image = tf.reshape(x,[-1,28,28,1])
y = tf.placeholder(tf.float32,shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

h0 = convLayer(x_image,1,8,2)
h1 = convLayer(h0,8,64,2)
h1_flat = tf.reshape(h1,[-1,7*7*64])
h2 = fcLayer(h1_flat,7*7*64,1024)
h2_drop = tf.nn.dropout(h2,keep_prob)
out = fcLayer(h2_drop,1024,10,relu=False)



with tf.Session() as sess:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    predictions = tf.argmax(out,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())


    for i in range(20000):
      batch = mnist.train.next_batch(64)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

    for ep in range(0,25):
        im = misc.imread('samples/final_'+str(ep)+'_1.png')
        im_gray = im[:,:,0:1]

        counts = np.zeros(10)
        im_reshaped = np.zeros((4096,28,28,1))
        for xi in range(0,64):
            for yi in range(0,64):
                im_reshaped[64*xi + yi,:,:,:] = im_gray[28*xi:28*(xi+1),28*yi:28*(yi+1)]

        preds = predictions.eval(feed_dict={x_image:im_reshaped,keep_prob:1.0})
        for p in preds:
            counts[p] = counts[p]+1

        print(counts)
    print("=================")
    for ep in range(0,25):
        im = misc.imread('samples/final_'+str(ep)+'_2.png')
        im_gray = im[:,:,0:1]

        counts = np.zeros(10)
        im_reshaped = np.zeros((4096,28,28,1))
        for xi in range(0,64):
            for yi in range(0,64):
                im_reshaped[64*xi + yi,:,:,:] = im_gray[28*xi:28*(xi+1),28*yi:28*(yi+1)]

        preds = predictions.eval(feed_dict={x_image:im_reshaped,keep_prob:1.0})
        for p in preds:
            counts[p] = counts[p]+1

        print(counts)
