import tensorflow as tf
import numpy as np

class Adversarial_Pair(object):
    def __init__(self,generator,discriminator):
        self.generator=generator
        self.discriminator=discriminator

    def build(self,config,sample_size=64,z_dim=100):

        if self.generator.y_dim:
            self.y= tf.placeholder(tf.float32, [config.batch_size, config.y_dim], name='y')
        else:
            self.y = None

        self.images = tf.placeholder(tf.float32, [config.batch_size] + [config.output_size, config.output_size, config.c_dim],name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [sample_size] + [config.output_size, config.output_size, config.c_dim],name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, z_dim],name='z')

        self.G = self.generator.build(self.z,self.y)
        self.D,self.D_logits = self.discriminator.build(self.images,self.y)
        self.D_,self.D_logits_ = self.discriminator.build(self.G,self.y)
        self.sampler = self.generator.build_sampler(self.z,self.y)

    def build_loss(self):
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

    def build_train_ops(self,config):

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'Discriminator_'+self.discriminator.model_name in var.name]
        self.g_vars = [var for var in t_vars if 'Generator_'+self.generator.model_name in var.name]

        self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
