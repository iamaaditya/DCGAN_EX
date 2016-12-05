import os
import time
from glob import glob
import tensorflow as tf
import numpy as np

class Trainer(object):
    def __init__(self,config):
        self.config = config

    def load_data(self):
        if self.config.dataset == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
        else:
            self.data = glob(os.path.join("./data", self.config.dataset, "*.jpg"))

    def train_single(self,adv):



        d_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv.d_loss, var_list=adv.d_vars)
        g_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1) \
                          .minimize(adv.g_loss, var_list=adv.g_vars)
        tf.initialize_all_variables().run()
        sample_z = np.random.uniform(-1, 1, size=(adv.sample_size, adv.z_dim))

        if self.config.dataset == 'mnist':
            sample_images = data_X[0:adv.sample_size]
            sample_labels = data_y[0:adv.sample_size]
        else:
            sample_files = data[0:adv.sample_size]
            sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for sample_file in sample_files]
            if (self.is_grayscale):
                sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()


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
