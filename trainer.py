import os
import time
from glob import glob
import tensorflow as tf
import numpy as np

class Trainer(object):
    def __init__(self,config,sess,adv):
        self.config = config
        self.sess = sess
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)
        self.adv = adv

    def load_data(self):
        if self.config.dataset == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
        else:
            self.data = glob(os.path.join("./data", self.config.dataset, "*.jpg"))

    def train_single(self):
        adv = self.adv

        d_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv.d_loss, var_list=adv.d_vars)
        g_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv.g_loss, var_list=adv.g_vars)
        tf.initialize_all_variables().run()
        sample_z = np.random.uniform(-1, 1, size=(adv.sample_size, adv.z_dim))

        if self.config.dataset == 'mnist':
            sample_images = self.data_X[0:adv.sample_size]
            sample_labels = self.data_y[0:adv.sample_size]
        else:
            sample_files = self.data[0:adv.sample_size]
            sample = [get_image(sample_file, self.config.image_size, is_crop=self.config.is_crop, resize_w=adv.generator.output_size, is_grayscale = adv.is_grayscale) for sample_file in sample_files]
            if (adv.is_grayscale):
                sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_images = np.array(sample).astype(np.float32)

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
                batch_z,batch_images,batch_labels = self.get_batch(idx)
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
                    _, summary_str = self.sess.run([g_optim, adv.G_sum],
                        feed_dict={ adv.z: batch_z, adv.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

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
                    _, summary_str = self.sess.run([g_optim, adv.G_sum],
                        feed_dict={ adv.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({adv.z: batch_z})
                    errD_real = self.d_loss_real.eval({adv.images: batch_images})
                    errG = self.g_loss.eval({adv.z: batch_z})

    def get_batch(self,idx):
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

        batch_z = np.random.uniform(-1, 1, [self.config.batch_size, self.adv.z_dim]).astype(np.float32)

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
