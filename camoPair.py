import tensorflow as tf
import os
from ops import *
from glob import glob
from camoGen import Generator
from camoDisc import Discriminator
import time

class Pair(object):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.batch_size = generator.batch_size
        self.discriminator = discriminator

    def build(self):
        self.real_images = tf.placeholder(tf.float32,[self.batch_size,256,256,3],name='real_images')
        self.patch_masks = tf.placeholder(tf.float32,[self.batch_size,256,256,3],name='patch_masks')
        self.patch_offsets = tf.placeholder(tf.int32,[self.batch_size,2],name='patch_offsets')

        self.G = self.generator.build(self.real_images, self.patch_masks,self.patch_offsets)
        self.D, self.D_logits = self.discriminator.build(self.real_images)
        self.D_,self.D_logits_ = self.discriminator.build(self.G)

    def build_loss(self):
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,tf.zeros_like(self.D_logits_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,tf.ones_like(self.D_logits_)))

        self.d_loss = self.d_loss_real+self.d_loss_fake

    def build_train_ops(self):
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'Discriminator_'+self.discriminator.name in var.name]
        self.g_vars = [var for var in t_vars if 'Generator_'+self.generator.name in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

    def load_data(self):
        self.data = glob(os.path.join("./data", "texture", "*.jpg"))
        self.all_images = [get_image(image, 256, is_crop=True, resize_w=256, is_grayscale = False) for image in self.data]
        print("FINISHED LOADING DATA")

    def get_batch(self,index):
        batch = self.all_images[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = np.array(batch).astype(np.float32)

        patch_masks = np.ones([self.batch_size,256,256,3])
        patch_offsets = np.zeros([self.batch_size,2])
        for i in range(0,self.batch_size):
            randX = np.random.randint(256-32)
            randY = np.random.randint(256-32)
            patch_masks[i,randX:randX+32,randY:randY+32,:] = 0
            patch_offsets[i][0] = randX
            patch_offsets[i][1] = randY

        return batch_images,patch_masks,patch_offsets

    def train(self,epochs,sess):
        tf.initialize_all_variables().run()
        sample_images,sample_masks,sample_offsets = self.get_batch(0)
        counter = 0
        for epoch in xrange(epochs):
            print(epoch)
            max_index = len(self.data)//self.batch_size
            for index in xrange(max_index):
                start_time = time.time()

                batch_images,batch_masks,batch_offsets = self.get_batch(index)

                _,errD_fake,errD_real = sess.run([self.d_optim,self.d_loss_fake,self.d_loss_real],feed_dict={self.real_images: batch_images,self.patch_masks: batch_masks, self.patch_offsets: batch_offsets})

                _,errG = sess.run([self.g_optim,self.g_loss],feed_dict={self.real_images: batch_images,self.patch_masks: batch_masks, self.patch_offsets: batch_offsets})



                counter = counter+1
                end_time = time.time()
                print('{:4d} - {:4d}:  Time: {:.4f}, D_loss_fake: {:.4f}, D_loss_real: {:.4f}, G_loss: {:.4f}'.format(epoch,index,end_time-start_time,errD_fake,errD_real,errG))
                if(counter%100==0):
                    result = self.G.eval({self.real_images: sample_images,self.patch_masks: sample_masks, self.patch_offsets: sample_offsets})
                    save_images(result,[4,4],'./camo_samples/train_{:02d}_{:04d}_1.png'.format(epoch,index))
                    d_real = self.D.eval({self.real_images: sample_images,self.patch_masks: sample_masks, self.patch_offsets: sample_offsets})
                    d_fake = self.D_.eval({self.real_images: sample_images,self.patch_masks: sample_masks, self.patch_offsets: sample_offsets})
                    print(d_real)
                    print(d_fake)


    def print_model_params(verbose=True):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            if verbose: print("name: " + str(variable.name) + " - shape:" + str(shape))
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            if verbose: print("variable parameters: " , variable_parametes)
            total_parameters += variable_parametes
        if verbose: print("total params: ", total_parameters)
        return total_parameters

with tf.Session() as sess:
    g = Generator("gen")
    d = Discriminator("disc")
    p = Pair(g,d)
    p.build()
    p.build_loss()
    p.build_train_ops()
    p.print_model_params()
    p.load_data()
    p.train(100,sess)
