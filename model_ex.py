from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from time import time

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, ex, m_name, image_size=108, is_crop=True,
                 batch_size=64, sample_size = 64, output_size=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.m_name = m_name
        
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1'+'_'+self.m_name)
        self.d_bn2 = batch_norm(name='d_bn2'+'_'+self.m_name)

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3'+'_'+self.m_name)

        self.g_bn0 = batch_norm(name='g_bn0'+'_'+self.m_name)
        self.g_bn1 = batch_norm(name='g_bn1'+'_'+self.m_name)
        self.g_bn2 = batch_norm(name='g_bn2'+'_'+self.m_name)

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3'+'_'+self.m_name)

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y'+'_'+self.m_name)

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim], name='real_images'+'_'+self.m_name)
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim], name='sample_images'+'_'+self.m_name)
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z'+'_'+self.m_name)

        self.z_sum = tf.histogram_summary("z"+'_'+self.m_name, self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits  = self.discriminator(self.images, self.y, reuse=False)

            self.sampler = self.sampler(self.z, self.y)
            self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(self.images)

            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        

        self.d_sum = tf.histogram_summary("d"+'_'+self.m_name, self.D)
        self.d__sum = tf.histogram_summary("d_"+'_'+self.m_name, self.D_)
        self.G_sum = tf.image_summary("G"+'_'+self.m_name, self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real"+'_'+self.m_name, self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake"+'_'+self.m_name, self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss"+'_'+self.m_name, self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss"+'_'+self.m_name, self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name and self.m_name in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name and self.m_name in var.name]

        self.saver = tf.train.Saver()


    def discriminator(self, image, y=None, reuse=True):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

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
            
            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        if not self.y_dim:
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin'+'_'+self.m_name, with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1'+'_'+self.m_name, with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2'+'_'+self.m_name, with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3'+'_'+self.m_name, with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4'+'_'+self.m_name, with_w=True)

            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4) 

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'+'_'+self.m_name)))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4,'g_h1_lin'+'_'+self.m_name)))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'+'_'+self.m_name)))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'+'_'+self.m_name))

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16,'g_h0_lin'+'_'+self.m_name), [-1, s16, s16, self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1'+'_'+self.m_name)
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2'+'_'+self.m_name)
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3'+'_'+self.m_name)
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4'+'_'+self.m_name)

            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4)

            # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim,'g_h0_lin'+'_'+self.m_name)))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'+'_'+self.m_name), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'+'_'+self.m_name), train=False))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'+'_'+self.m_name))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec
            
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False



def train(sess, config, ex):
    """Train DCGAN"""

    if config.dataset == 'mnist':
        dcgan_1 = DCGAN(sess, ex, "1", image_size=config.image_size, batch_size=config.batch_size, y_dim=10, output_size=28, c_dim=1,
                dataset_name=config.dataset, is_crop=config.is_crop, checkpoint_dir=config.checkpoint_dir, sample_dir=config.sample_dir)
        dcgan_2 = DCGAN(sess, ex, "2", image_size=config.image_size, batch_size=config.batch_size, y_dim=10, output_size=28, c_dim=1,
                dataset_name=config.dataset, is_crop=config.is_crop, checkpoint_dir=config.checkpoint_dir, sample_dir=config.sample_dir)
    else:
        dcgan_1 = DCGAN(sess, ex, "1", image_size=config.image_size, batch_size=config.batch_size, output_size=config.output_size, c_dim=config.c_dim,
                dataset_name=config.dataset, is_crop=config.is_crop, checkpoint_dir=config.checkpoint_dir, sample_dir=config.sample_dir,gf_dim=32,df_dim=16)
        dcgan_2 = DCGAN(sess, ex, "2", image_size=config.image_size, batch_size=config.batch_size, output_size=config.output_size, c_dim=config.c_dim,
                dataset_name=config.dataset, is_crop=config.is_crop, checkpoint_dir=config.checkpoint_dir, sample_dir=config.sample_dir,gf_dim=32,df_dim=16)
    if config.dataset == 'mnist':
        data_X, data_y = dcgan_1.load_mnist()
    else:
        data = glob(os.path.join("./data", config.dataset, "*.png"))
        #np.random.shuffle(data)
    ex.info['errD_1'] = []
    ex.info['errD_2'] = []
    ex.info['errG_1'] = []
    ex.info['errG_2'] = []

    directory = "./tensorboards/" + str(time())
    writer = tf.train.SummaryWriter(directory, dcgan_1.sess.graph)

    d_optim_1 = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(dcgan_1.d_loss, var_list=dcgan_1.d_vars)
    d_optim_2 = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(dcgan_2.d_loss, var_list=dcgan_2.d_vars)
    g_optim_1 = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(dcgan_1.g_loss, var_list=dcgan_1.g_vars)
    g_optim_2 = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(dcgan_2.g_loss, var_list=dcgan_2.g_vars)
    tf.initialize_all_variables().run()

    dcgan_1.g_sum = tf.merge_summary([dcgan_1.z_sum, dcgan_1.d__sum, dcgan_1.G_sum, dcgan_1.d_loss_fake_sum, dcgan_1.g_loss_sum])
    dcgan_2.g_sum = tf.merge_summary([dcgan_2.z_sum, dcgan_2.d__sum, dcgan_2.G_sum, dcgan_2.d_loss_fake_sum, dcgan_2.g_loss_sum])

    dcgan_1.d_sum = tf.merge_summary([dcgan_1.z_sum, dcgan_1.d_sum, dcgan_1.d_loss_real_sum, dcgan_1.d_loss_sum])
    dcgan_2.d_sum = tf.merge_summary([dcgan_2.z_sum, dcgan_2.d_sum, dcgan_2.d_loss_real_sum, dcgan_2.d_loss_sum])


    sample_z_1 = np.random.uniform(-1, 1, size=(dcgan_1.sample_size , dcgan_1.z_dim))
    sample_z_2 = np.random.uniform(-1, 1, size=(dcgan_2.sample_size , dcgan_2.z_dim))
    
    if config.dataset == 'mnist':
        sample_images = data_X[0:dcgan_1.sample_size]
        sample_labels = data_y[0:dcgan_1.sample_size]
    else:
        sample_files = data[0:dcgan_1.sample_size]
        sample = [get_image(sample_file, dcgan_1.image_size, is_crop=dcgan_1.is_crop, resize_w=dcgan_1.output_size, is_grayscale = dcgan_1.is_grayscale) for sample_file in sample_files]
        if (dcgan_1.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        
    counter = 1
    start_time = time()

    # TODO fix the loading of pretrained model
    # if dcgan_1.load(dcgan_1.checkpoint_dir):
    #     print(" [*] Load SUCCESS")
    # else:
    #     print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
        if config.dataset == 'mnist':
            batch_idxs = min(len(data_X), config.train_size) // config.batch_size
        else:            
            data = glob(os.path.join("./data", config.dataset, "*.png"))
            batch_idxs = min(len(data), config.train_size) // config.batch_size

        for idx in xrange(0, batch_idxs):
            if config.dataset == 'mnist':
                batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
            else:
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, dcgan_1.image_size, is_crop=dcgan_1.is_crop, resize_w=dcgan_1.output_size, is_grayscale = dcgan_1.is_grayscale) for batch_file in batch_files]
                if (dcgan_1.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

            batch_z_1 = np.random.uniform(-1, 1, [config.batch_size, dcgan_1.z_dim]).astype(np.float32)
            batch_z_2 = np.random.uniform(-1, 1, [config.batch_size, dcgan_2.z_dim]).astype(np.float32)

            if config.dataset == 'mnist':
                # Update D network
                # 1
                _, summary_str = sess.run([d_optim_1, dcgan_1.d_sum], feed_dict={dcgan_1.images: batch_images, dcgan_1.z: batch_z_1, dcgan_1.y:batch_labels})
                writer.add_summary(summary_str, counter)
                # 2
                _, summary_str = sess.run([d_optim_2, dcgan_2.d_sum], feed_dict={dcgan_2.images: batch_images, dcgan_2.z: batch_z_2, dcgan_2.y:batch_labels})
                writer.add_summary(summary_str, counter)

                # Update G network
                # 1
                _, summary_str = sess.run([g_optim_1, dcgan_1.g_sum], feed_dict={dcgan_1.z: batch_z_1, dcgan_1.y:batch_labels})
                writer.add_summary(summary_str, counter)
                # 2
                _, summary_str = sess.run([g_optim_2, dcgan_2.g_sum], feed_dict={dcgan_2.z: batch_z_2, dcgan_2.y:batch_labels})
                writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # 1
                _, summary_str = sess.run([g_optim_1, dcgan_1.g_sum],feed_dict={dcgan_1.z: batch_z_1, dcgan_1.y:batch_labels})
                writer.add_summary(summary_str, counter)
                # 2
                _, summary_str = sess.run([g_optim_2, dcgan_2.g_sum],feed_dict={dcgan_2.z: batch_z_2, dcgan_2.y:batch_labels})
                writer.add_summary(summary_str, counter)
                
                # 1
                errD_fake_1 = dcgan_1.d_loss_fake.eval({dcgan_1.z: batch_z_1, dcgan_1.y:batch_labels})
                errD_real_1 = dcgan_1.d_loss_real.eval({dcgan_1.images: batch_images, dcgan_1.y:batch_labels})
                errG_1 = dcgan_1.g_loss.eval({dcgan_1.z: batch_z_1, dcgan_1.y:batch_labels})
                # 2
                errD_fake_2 = dcgan_2.d_loss_fake.eval({dcgan_2.z: batch_z_2, dcgan_2.y:batch_labels})
                errD_real_2 = dcgan_2.d_loss_real.eval({dcgan_2.images: batch_images, dcgan_2.y:batch_labels})
                errG_2 = dcgan_2.g_loss.eval({dcgan_2.z: batch_z_2, dcgan_2.y:batch_labels})
            else:
                # Update D network
                # 1
                _, summary_str = sess.run([d_optim_1, dcgan_1.d_sum], feed_dict={dcgan_1.images: batch_images, dcgan_1.z: batch_z_1})
                writer.add_summary(summary_str, counter)
                # 2
                _, summary_str = sess.run([d_optim_2, dcgan_2.d_sum], feed_dict={dcgan_2.images: batch_images, dcgan_2.z: batch_z_2})
                writer.add_summary(summary_str, counter)

                # Update G network
                # 1 
                _, summary_str = sess.run([g_optim_1, dcgan_1.g_sum], feed_dict={dcgan_1.z: batch_z_1})
                writer.add_summary(summary_str, counter)
                # 2 
                _, summary_str = sess.run([g_optim_2, dcgan_2.g_sum], feed_dict={dcgan_2.z: batch_z_2})
                writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # 1 
                _, summary_str = sess.run([g_optim_1, dcgan_1.g_sum], feed_dict={dcgan_1.z: batch_z_1})
                writer.add_summary(summary_str, counter)
                # 2 
                _, summary_str = sess.run([g_optim_2, dcgan_2.g_sum], feed_dict={dcgan_2.z: batch_z_2})
                writer.add_summary(summary_str, counter)
                
                # if epoch % 5 == 0:
                #     pass
                # else:
                # 1 
                errD_fake_1 = dcgan_1.d_loss_fake.eval({dcgan_1.z: batch_z_1})
                errD_real_1 = dcgan_1.d_loss_real.eval({dcgan_1.images: batch_images})
                errG_1      = dcgan_1.g_loss.eval({dcgan_1.z: batch_z_1})
                # 2 
                errD_fake_2 = dcgan_2.d_loss_fake.eval({dcgan_2.z: batch_z_2})
                errD_real_2 = dcgan_2.d_loss_real.eval({dcgan_2.images: batch_images})
                errG_2      = dcgan_2.g_loss.eval({dcgan_2.z: batch_z_2})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_1: %.8f, g_loss_1: %.8f, d_loss_2: %.8f, g_loss_2: %.8f" \
                % (epoch, idx, batch_idxs, time()-start_time, errD_fake_1+errD_real_1, errG_1, errD_fake_2+errD_real_2, errG_2))

            if np.mod(counter, 100) == 1:
                if config.dataset == 'mnist':
                    # 1
                    samples_1, d_loss_1, g_loss_1 = sess.run([dcgan_1.sampler, dcgan_1.d_loss, dcgan_1.g_loss],
                        feed_dict={dcgan_1.z: sample_z_1, dcgan_1.images: sample_images, dcgan_1.y:batch_labels})
                    # 2
                    samples_2, d_loss_2, g_loss_2 = sess.run([dcgan_2.sampler, dcgan_2.d_loss, dcgan_2.g_loss],
                        feed_dict={dcgan_2.z: sample_z_2, dcgan_2.images: sample_images, dcgan_2.y:batch_labels})
                else:
                    # 1
                    samples_1, d_loss_1, g_loss_1 = sess.run([dcgan_1.sampler, dcgan_1.d_loss, dcgan_1.g_loss],
                        feed_dict={dcgan_1.z: sample_z_1, dcgan_1.images: sample_images})
                    # 2
                    samples_2, d_loss_2, g_loss_2 = sess.run([dcgan_2.sampler, dcgan_2.d_loss, dcgan_2.g_loss],
                        feed_dict={dcgan_2.z: sample_z_2, dcgan_2.images: sample_images})

                save_images(samples_1, [8, 8], './{}/train_{:02d}_{:04d}_1.png'.format(config.sample_dir, epoch, idx))
                save_images(samples_2, [8, 8], './{}/train_{:02d}_{:04d}_2.png'.format(config.sample_dir, epoch, idx))
                print("[Sample] d_loss_1: %.8f, g_loss_1: %.8f, d_loss_2: %.8f, g_loss_2: %.8f" % (d_loss_1, g_loss_1, d_loss_2, g_loss_2))

            # TODO save both the model
            # if np.mod(counter, 500) == 2: \
            #         dcgan_1.save(config.checkpoint_dir, counter)

        # 1 
        ex.info['errD_1'].append(errD_fake_1+errD_real_1)
        ex.info['errG_1'].append(errG_1)
        # 2 
        ex.info['errD_2'].append(errD_fake_2+errD_real_2)
        ex.info['errG_2'].append(errG_2)

