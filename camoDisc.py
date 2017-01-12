import tensorflow as tf
from ops import *

class Discriminator(object):
    def __init__(self,name,batch_size=16):
        self.name = name
        self.batch_size = batch_size
        self.has_built = False

    def build(self,images):
        with tf.variable_scope("Discriminator_" + self.name,reuse=self.has_built):
            # h0 = lrelu(batch_norm(conv2d(images, 8,name='d_h0_conv'),scope='d_h0_conv')) #128x128x8
            # h1 = lrelu(batch_norm(conv2d(h0,16,name='d_h1_conv'),scope='d_h1_conv')) #64x64x16
            # h2 = lrelu(batch_norm(conv2d(h1,32,name='d_h2_conv'),scope='d_h2_conv')) #32x32x32
            # h3 = lrelu(batch_norm(conv2d(h1,64,name='d_h3_conv'),scope='d_h3_conv')) #16x16x64
            # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            #smaller disc
            h0 = lrelu(batch_norm(conv2d(images, 4,name='d_h0_conv'),scope='d_h0_conv')) #128x128x8
            h1 = lrelu(batch_norm(conv2d(h0,8,name='d_h1_conv'),scope='d_h1_conv')) #64x64x16
            h2 = lrelu(batch_norm(conv2d(h1,8,name='d_h2_conv'),scope='d_h2_conv')) #32x32x32
            h3 = lrelu(batch_norm(conv2d(h1,16,name='d_h3_conv'),scope='d_h3_conv')) #16x16x64
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            #minimal disc
            #h4 = linear(tf.reshape(images,[self.batch_size,-1]),1,'d_h4_lin')

            self.has_built=True
            return h4,tf.nn.sigmoid(h4)
