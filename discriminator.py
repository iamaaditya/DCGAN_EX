import tensorflow as tf
from ops import *

class Discriminator(object):
    def __init__(self,model_name,y_dim=None,df_dim=64,dfc_dim=1024):
        self.model_name = model_name
        self.y_dim = y_dim
        self.df_dim = df_dim
        self.dfc_dim = dfc_dim

        self.has_built = False

    def build(self, image, y=None):
        with tf.variable_scope("Discriminator_" + self.model_name,reuse=self.has_built):
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),scope="d_h1_conv"))
                h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, name='d_h2_conv'),scope="d_h2_conv"))
                h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, name='d_h3_conv'),scope="d_h3_conv"))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(batch_norm(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv'),scope="d_h1_conv"))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = tf.concat(1, [h1, y])

                h2 = lrelu(batch_norm(linear(h1, self.dfc_dim, 'd_h2_lin'),scope="d_h2_lin"))
                h2 = tf.concat(1, [h2, y])

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3
