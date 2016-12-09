import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from adversarial_pair import Adversarial_Pair
from trainer import Trainer
import numpy as np


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
FLAGS = flags.FLAGS


with tf.Session() as sess:
    gen = Generator("gen",y_dim=10,output_size=FLAGS.output_size,c_dim=FLAGS.c_dim)
    disc = Discriminator("disc",y_dim=10,c_dim=FLAGS.c_dim)

    adv = Adversarial_Pair(gen,disc)
    adv.build(FLAGS)
    adv.build_loss()
    adv.build_train_ops(FLAGS)

    trainer = Trainer(FLAGS,sess,adv)
    trainer.load_data()
    trainer.train_single()
