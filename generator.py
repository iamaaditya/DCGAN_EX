import tensorflow as tf
from ops import *

class Generator(object):

    def __init__(self,model_name,batch_size=64,y_dim=None,output_size=64,gf_dim=64,gfc_dim=1024,c_dim=3):
        self.model_name = model_name
        self.batch_size=batch_size
        self.y_dim = y_dim
        self.output_size = output_size
        self.gf_dim=gf_dim
        self.gfc_dim=gfc_dim
        self.c_dim=c_dim


        self.has_built=False

    def build(self,z,y=None):
        with tf.variable_scope("Generator_" + self.model_name,reuse=self.has_built):
            if not self.y_dim:
                s = self.output_size
                s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
                h0 = tf.nn.relu(batch_norm(self.h0,scope="g_h0_lin"))

                self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                    [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(batch_norm(self.h1,scope="g_h1"))

                h2, self.h2_w, self.h2_b = deconv2d(h1,
                    [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(batch_norm(h2,scope="g_h2"))

                h3, self.h3_w, self.h3_b = deconv2d(h2,
                    [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(batch_norm(h3,scope="g_h3"))

                h4, self.h4_w, self.h4_b = deconv2d(h3,
                    [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)
                self.has_built=True
                return tf.nn.tanh(h4)
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(batch_norm(linear(z, self.gfc_dim, 'g_h0_lin'),scope="g_h0_lin"))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(batch_norm(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'),scope="g_h1_lin"))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(batch_norm(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'),scope="g_h2"))
                h2 = conv_cond_concat(h2, yb)
                self.has_built=True
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def build_sampler(self,z,y=None):
        with tf.variable_scope("Generator_" + self.model_name,reuse=self.has_built):
            if not self.y_dim:

                s = self.output_size
                s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

                # project `z` and reshape
                h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin'),
                                [-1, s16, s16, self.gf_dim * 8])
                h0 = tf.nn.relu(batch_norm(h0, is_training=False,scope="g_h0_lin"))

                h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=False,scope="g_h1"))

                h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=False,scope="g_h2"))

                h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=False,scope="g_h3"))

                h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(batch_norm(linear(z, self.gfc_dim, 'g_h0_lin'),is_training=False,scope="g_h0_lin"))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(batch_norm(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), is_training=False,scope="g_h1_lin"))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(batch_norm(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), is_training=False,scope="g_h2"))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))
