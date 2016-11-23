import os
import scipy.misc
import numpy as np


from ops import *
from utils import *

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver


import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")

flags.DEFINE_string("db_name", "DCGAN_EX", "Name of the DB for mongo")
FLAGS = flags.FLAGS


ex = Experiment(FLAGS.db_name)
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(MongoObserver.create(db_name=FLAGS.db_name))




class Discriminator(object):
    def __init__(self, sess, ex, m_name, batch_size=64,df_dim=64,dfc_dim=1024,y_dim=None,c_dim=3):
        self.sess = sess
        self.m_name = m_name
        self.batch_size = batch_size
        self.df_dim = df_dim
        self.dfc_dim = dfc_dim
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.has_built = False

        with tf.variable_scope('discriminator_' + self.m_name):
            self.d_bn1 = batch_norm(name='d_bn1')
            self.d_bn2 = batch_norm(name='d_bn2')
            if not self.y_dim:
                self.d_bn3 = batch_norm(name='d_bn3')

    def build(self,image,y=None):
        with tf.variable_scope('discriminator_' + self.m_name, reuse=self.has_built) as scope:

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, name='d_h3_lin')
                self.has_built=True
                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = tf.concat(1, [h1, y])

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat(1, [h2, y])

                h3 = linear(h2, 1, 'd_h3_lin')
                self.has_built=True
                return tf.nn.sigmoid(h3), h3


class Generator(object):
    def __init__(self, sess, ex, m_name, batch_size=64,gf_dim=64,gfc_dim=1024,y_dim=None,c_dim=3,output_size=64,z_dim=100):
        self.sess = sess
        self.m_name = m_name
        self.batch_size = batch_size
        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.output_size = output_size
        self.z_dim=z_dim
        self.has_built = False


        with tf.variable_scope('generator_' + self.m_name):
            self.g_bn0 = batch_norm(name='g_bn0')
            self.g_bn1 = batch_norm(name='g_bn1')
            self.g_bn2 = batch_norm(name='g_bn2')

            if not self.y_dim:
                self.g_bn3 = batch_norm(name='g_bn3')

    def build(self,z,y=None):
        with tf.variable_scope('generator_' + self.m_name,reuse=self.has_built) as scope:
            if not self.y_dim:
                s = self.output_size
                s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)
                self.has_built=True
                return tf.nn.tanh(h4)
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                print "*"*10, self.m_name, "*"*10
                k = linear(z, self.gfc_dim, 'g_h0_lin')
                kk = self.g_bn0(k)
                kkk = tf.nn.relu(kk)
                h0  = kkk
                # h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'+'_'+self.m_name)))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4,'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)
                self.has_built=True
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def build_sampler(self, z, y=None):
        with tf.variable_scope('generator_' + self.m_name,reuse=self.has_built) as scope:


            if not self.y_dim:

                s = self.output_size
                s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

                # project `z` and reshape
                h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16,'g_h0_lin'), [-1, s16, s16, self.gf_dim*8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')
                self.has_built=True
                return tf.nn.tanh(h4)
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim,'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)
                self.has_built=True
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

class AdversarialNetwork(object):
    def __init__(self,discriminator,generator):
        self.discriminator=discriminator
        self.generator=generator

    def build_model(self):
        combined_m_name = '_'+self.generator.m_name+'_'+self.discriminator.m_name
        self.images = tf.placeholder(tf.float32, [self.discriminator.batch_size] + [self.generator.output_size, self.generator.output_size, self.generator.c_dim], name='real_images'+combined_m_name)
        self.z = tf.placeholder(tf.float32, [None, self.generator.z_dim], name='z'+combined_m_name)

        if(self.discriminator.y_dim):
            self.y= tf.placeholder(tf.float32, [self.discriminator.batch_size, self.discriminator.y_dim], name='y'+combined_m_name)
            self.G = self.generator.build(self.z, self.y)
            self.D, self.D_logits  = self.discriminator.build(self.images, self.y)

            self.sampler = self.generator.build_sampler(self.z, self.y)
            self.D_, self.D_logits_ = self.discriminator.build(self.G, self.y)
        else:
            self.G = self.generator.build(self.z)
            self.D, self.D_logits = self.discriminator.build(self.images)

            self.sampler = self.generator.build_sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G)

        self.d_sum = tf.histogram_summary("d"+'_'+self.discriminator.m_name, self.D)
        self.d__sum = tf.histogram_summary("d_"+'_'+self.discriminator.m_name, self.D_)
        self.G_sum = tf.image_summary("G"+'_'+self.generator.m_name, self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real"+'_'+self.discriminator.m_name, self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake"+'_'+self.discriminator.m_name, self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss"+'_'+self.generator.m_name, self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss"+'_'+self.discriminator.m_name, self.d_loss)

        t_vars = tf.trainable_variables()

        # self.d_vars = [var for var in t_vars if 'd_' in var.name ]
        # self.g_vars = [var for var in t_vars if 'g_' in var.name ]
        self.d_vars = [var for var in t_vars if 'd_' in var.name and self.discriminator.m_name in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name and self.generator.m_name in var.name]

        self.saver = tf.train.Saver()

    def build_optimizers(self,config):
        self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        return d_optim,g_optim

def train(sess,config,ex):

    d1 = Discriminator(sess, ex, "Disc1",batch_size=config.batch_size,y_dim=10,c_dim=1)
    d2 = Discriminator(sess, ex, "Disc2",batch_size=config.batch_size,y_dim=10,c_dim=1)

    g1 = Generator(sess,ex,"Gen1",batch_size=config.batch_size,y_dim=10,c_dim=1,output_size=28)
    g2 = Generator(sess,ex,"Gen2",batch_size=config.batch_size,y_dim=10,c_dim=1,output_size=28)

    adv11 = AdversarialNetwork(d1,g1)
    adv11.build_model()
    adv12 = AdversarialNetwork(d1,g2)
    adv12.build_model()
    adv21 = AdversarialNetwork(d2,g1)
    adv21.build_model()
    adv22 = AdversarialNetwork(d2,g2)
    adv22.build_model()



with tf.Session() as sess:
        train(sess,FLAGS,ex)
