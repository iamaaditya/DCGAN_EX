import os
import numpy as np
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from adversarial_pair import Adversarial_Pair
from trainer import Trainer


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("sample_size", 64, "The size of sample images [64]")
FLAGS = flags.FLAGS

FLAGS.c_dim=1
FLAGS.output_size=28
FLAGS.dataset = "mnist"

data_dir = os.path.join("./data", "mnist")

fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
loaded = np.fromfile(file=fd,dtype=np.uint8)
trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
loaded = np.fromfile(file=fd,dtype=np.uint8)
trY = loaded[8:].reshape((60000)).astype(np.float)


zeros = trX[trY==0]
ones = trX[trY==1]
#data = np.concatenate((zeros,ones),axis=0)
data = zeros
y_vals = np.zeros_like(trY)

print(zeros.shape)
print(ones.shape)
print(data.shape)

with tf.Session() as sess:
    gen = Generator("gen",output_size=FLAGS.output_size,c_dim=FLAGS.c_dim,y_dim=10)
    disc = Discriminator("disc",c_dim=FLAGS.c_dim,y_dim=10)

    adv = Adversarial_Pair(gen,disc,sample_size=FLAGS.sample_size)
    adv.build(FLAGS)
    adv.build_loss()
    adv.build_train_ops(FLAGS)

    trainer = Trainer(FLAGS,sess,rotate_samples=True)
    trainer.load_data()
    trainer.clear_y()
    trainer.train_single(adv)
