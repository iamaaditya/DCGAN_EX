import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from adversarial_pair import Adversarial_Pair
from utils import *

class Trainer(object):
    def __init__(self,config,sess,rotate_samples=False):
        self.config = config
        self.sess = sess
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)
        self.rotate_samples = rotate_samples

    def load_data(self):
        if self.config.dataset == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.sample_images = self.data_X[0:self.config.sample_size]
            self.sample_labels = self.data_y[0:self.config.sample_size]
        else:
            self.data = glob(os.path.join("./data", self.config.dataset, "*.jpg"))
            sample_files = self.data[0:self.config.sample_size]
            sample = [get_image(sample_file, self.config.image_size, is_crop=self.config.is_crop, resize_w=self.config.output_size, is_grayscale = (self.config.c_dim==1)) for sample_file in sample_files]
            if (self.config.c_dim==1):
                self.sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                self.sample_images = np.array(sample).astype(np.float32)

    def set_data(self,data_X,data_y=None):
        self.data_X = data_X
        self.data_y = data_y
        self.sample_images = self.data_X[0:self.config.sample_size]
        self.sample_labels = self.data_y[0:self.config.sample_size]

    def clear_y(self):
        self.data_y = np.zeros_like(self.data_y)

    def train_single(self,adv):

        d_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv.d_loss, var_list=adv.d_vars)
        g_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv.g_loss, var_list=adv.g_vars)
        tf.initialize_all_variables().run()
        sample_z = np.random.uniform(-1, 1, size=(adv.sample_size, adv.z_dim))
        _,_,sample_y = self.get_batch(0,adv)




        counter = 1
        start_time = time.time()

        ####TODO: ADD LOADING OF CHECKPOINTS
        #if self.load(self.checkpoint_dir):
        #    print(" [*] Load SUCCESS")
        #else:
        #    print(" [!] Load failed...")

        for epoch in xrange(self.config.epoch):
            if self.config.dataset == 'mnist':
                batch_idxs = min(len(self.data_X), self.config.train_size) // self.config.batch_size
            else:
                batch_idxs = min(len(self.data), self.config.train_size) // self.config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_z,batch_images,batch_labels = self.get_batch(idx,adv)
                if self.config.dataset == 'mnist':
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, adv.d_sum],
                        feed_dict={ adv.images: batch_images, adv.z: batch_z, adv.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, adv.G_sum],
                        feed_dict={ adv.z: batch_z, adv.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #_, summary_str = self.sess.run([g_optim, adv.G_sum],
                    #    feed_dict={ adv.z: batch_z, adv.y:batch_labels })
                    #self.writer.add_summary(summary_str, counter)

                    errD_fake = adv.d_loss_fake.eval({adv.z: batch_z, adv.y:batch_labels})
                    errD_real = adv.d_loss_real.eval({adv.images: batch_images, adv.y:batch_labels})
                    errG = adv.g_loss.eval({adv.z: batch_z, adv.y:batch_labels})
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, adv.d_sum],
                        feed_dict={ adv.images: batch_images, adv.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, adv.G_sum],
                        feed_dict={ adv.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #_, summary_str = self.sess.run([g_optim, adv.G_sum],
                    #    feed_dict={ adv.z: batch_z })
                    #self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({adv.z: batch_z})
                    errD_real = self.d_loss_real.eval({adv.images: batch_images})
                    errG = self.g_loss.eval({adv.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if self.config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [adv.sampler, adv.d_loss, adv.g_loss],
                            feed_dict={adv.z: sample_z, adv.images: self.sample_images, adv.y:sample_y}
                        )
                    else:
                        samples, d_loss, g_loss = self.sess.run(
                            [adv.sampler, adv.d_loss, adv.g_loss],
                            feed_dict={adv.z: sample_z, adv.images: self.sample_images}
                        )

                    save_images(samples, [8, 8],
                                './{}/train_{:02d}_{:04d}.png'.format(self.config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

            final_samples = ()
            for i in range(0,8):
                for j in range(0,8):
                    final_sample_z = np.random.uniform(-1, 1, size=(adv.sample_size, adv.z_dim))
                    samples = self.sess.run(
                            [adv.sampler],
                            feed_dict={adv.z: final_sample_z, adv.y:sample_y}
                        )
                    final_samples = final_samples + (samples,)

            final_samples = np.hstack(final_samples)[0]
            save_images(final_samples,[64,64],'./{}/final_{}.png'.format(self.config.sample_dir,epoch))

    def train_two_generators(self,gen1,gen2,disc):
        adv1 = Adversarial_Pair(gen1,disc)
        adv2 = Adversarial_Pair(gen2,disc)
        adv1.build(self.config)
        adv2.build(self.config)
        adv1.build_loss()
        adv2.build_loss()
        adv1.build_train_ops(self.config)
        adv2.build_train_ops(self.config)

        d_optim_1 = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv1.d_loss, var_list=adv1.d_vars)
        g_optim_1 = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv1.g_loss, var_list=adv1.g_vars)
        d_optim_2 = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv2.d_loss, var_list=adv2.d_vars)
        g_optim_2 = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv2.g_loss, var_list=adv2.g_vars)

        tf.initialize_all_variables().run()
        sample_z = np.random.uniform(-1, 1, size=(adv1.sample_size, adv1.z_dim))
        _,_,sample_y = self.get_batch(0,adv1)

        counter = 1
        start_time = time.time()

        for epoch in xrange(self.config.epoch):
            if self.config.dataset == 'mnist':
                batch_idxs = min(len(self.data_X), self.config.train_size) // self.config.batch_size
            else:
                batch_idxs = min(len(self.data), self.config.train_size) // self.config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_z,batch_images,batch_labels = self.get_batch(idx,adv1)
                if self.config.dataset == 'mnist':
                    if((idx//10)%2==0):
                        # Update D network
                        _, summary_str = self.sess.run([d_optim_1, adv1.d_sum],
                            feed_dict={ adv1.images: batch_images, adv1.z: batch_z, adv1.y:batch_labels })
                        self.writer.add_summary(summary_str, counter)

                        # Update G network
                        _, summary_str = self.sess.run([g_optim_1, adv1.G_sum],
                            feed_dict={ adv1.z: batch_z, adv1.y:batch_labels })
                        self.writer.add_summary(summary_str, counter)
                    else:
                        # Update D network
                        _, summary_str = self.sess.run([d_optim_2, adv2.d_sum],
                            feed_dict={ adv2.images: batch_images, adv2.z: batch_z, adv2.y:batch_labels })
                        self.writer.add_summary(summary_str, counter)

                        # Update G network
                        _, summary_str = self.sess.run([g_optim_2, adv2.G_sum],
                            feed_dict={ adv2.z: batch_z, adv2.y:batch_labels })
                        self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #_, summary_str = self.sess.run([g_optim, adv.G_sum],
                    #    feed_dict={ adv.z: batch_z, adv.y:batch_labels })
                    #self.writer.add_summary(summary_str, counter)

                    errD_fake_1 = adv1.d_loss_fake.eval({adv1.z: batch_z, adv1.y:batch_labels})
                    errD_fake_2 = adv2.d_loss_fake.eval({adv2.z: batch_z, adv2.y:batch_labels})
                    errD_real = adv1.d_loss_real.eval({adv1.images: batch_images, adv1.y:batch_labels})

                    errG_1 = adv1.g_loss.eval({adv1.z: batch_z, adv1.y:batch_labels})
                    errG_2 = adv2.g_loss.eval({adv2.z: batch_z, adv2.y:batch_labels})
                else:

                    print("NOT YET IMPLEMENTED")
                    exit()
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, adv.d_sum],
                        feed_dict={ adv.images: batch_images, adv.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, adv.G_sum],
                        feed_dict={ adv.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    #_, summary_str = self.sess.run([g_optim, adv.G_sum],
                    #    feed_dict={ adv.z: batch_z })
                    #self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({adv.z: batch_z})
                    errD_real = self.d_loss_real.eval({adv.images: batch_images})
                    errG = self.g_loss.eval({adv.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_1: %.4f, g_loss_1: %.4f, d_loss_2: %.4f, g_loss_2: %.4f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake_1+errD_real, errG_1,errD_fake_2+errD_real,errG_2))

                if np.mod(counter, 100) == 1:
                    if self.config.dataset == 'mnist':
                        samples_1, d_loss_1, g_loss_1 = self.sess.run(
                            [adv1.sampler, adv1.d_loss, adv1.g_loss],
                            feed_dict={adv1.z: sample_z, adv1.images: self.sample_images, adv1.y:sample_y}
                        )
                        samples_2, d_loss_2, g_loss_2 = self.sess.run(
                            [adv2.sampler, adv2.d_loss, adv2.g_loss],
                            feed_dict={adv2.z: sample_z, adv2.images: self.sample_images, adv2.y:sample_y}
                        )
                    else:
                        print("NOT YET IMPLEMENTED")
                        exit()
                        samples, d_loss, g_loss = self.sess.run(
                            [adv.sampler, adv.d_loss, adv.g_loss],
                            feed_dict={adv.z: sample_z, adv.images: self.sample_images}
                        )

                    save_images(samples_1, [8, 8],
                                './{}/train_{:02d}_{:04d}_1.png'.format(self.config.sample_dir, epoch, idx))
                    save_images(samples_2, [8, 8],
                                './{}/train_{:02d}_{:04d}_2.png'.format(self.config.sample_dir, epoch, idx))
                    print("[Sample] d_loss_1: %.8f, g_loss_1: %.8f" % (d_loss_1, g_loss_1))
                    print("[Sample] d_loss_2: %.8f, g_loss_2: %.8f" % (d_loss_2, g_loss_2))
            final_samples = ()
            for i in range(0,8):
                for j in range(0,8):
                    final_sample_z = np.random.uniform(-1, 1, size=(adv1.sample_size, adv1.z_dim))
                    samples = self.sess.run(
                            [adv1.sampler],
                            feed_dict={adv1.z: final_sample_z, adv1.y:sample_y}
                        )
                    final_samples = final_samples + (samples,)

            final_samples = np.hstack(final_samples)[0]
            save_images(final_samples,[64,64],'./{}/final_{}_1.png'.format(self.config.sample_dir,epoch))
            final_samples = ()
            for i in range(0,8):
                for j in range(0,8):
                    final_sample_z = np.random.uniform(-1, 1, size=(adv2.sample_size, adv2.z_dim))
                    samples = self.sess.run(
                            [adv2.sampler],
                            feed_dict={adv2.z: final_sample_z, adv2.y:sample_y}
                        )
                    final_samples = final_samples + (samples,)

            final_samples = np.hstack(final_samples)[0]
            save_images(final_samples,[64,64],'./{}/final_{}_2.png'.format(self.config.sample_dir,epoch))

    def train_pair(self,gen1,gen2,disc1,disc2):
        adv11 = Adversarial_Pair(gen1,disc1)
        adv12 = Adversarial_Pair(gen1,disc2)
        adv21 = Adversarial_Pair(gen2,disc1)
        adv22 = Adversarial_Pair(gen2,disc2)

        adv11.build(self.config)
        adv12.build(self.config)
        adv21.build(self.config)
        adv22.build(self.config)

        adv11.build_loss()
        adv12.build_loss()
        adv21.build_loss()
        adv22.build_loss()

        adv11.build_train_ops(self.config)
        adv12.build_train_ops(self.config)
        adv21.build_train_ops(self.config)
        adv22.build_train_ops(self.config)




    def get_batch(self,idx,adv):
        if self.config.dataset == 'mnist':
            batch_images = self.data_X[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
            batch_labels = self.data_y[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
        else:
            batch_files = self.data[idx*self.config.batch_size:(idx+1)*self.config.batch_size]
            batch = [get_image(batch_file, self.config.image_size, is_crop=self.config.is_crop, resize_w=adv.generator.output_size, is_grayscale = adv.is_grayscale) for batch_file in batch_files]
            if (self.is_grayscale):
                batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [self.config.batch_size, adv.z_dim]).astype(np.float32)

        if(self.rotate_samples):
            for i in range(0,self.config.batch_size):
                rotations = np.random.randint(4)
                batch_images[i] = np.rot90(batch_images[i],k=rotations)

        return batch_z,batch_images,batch_labels

    def load_mnist(self):
        data_dir = os.path.join("./data", self.config.dataset)

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

        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0

        return X/255.,y_vec
