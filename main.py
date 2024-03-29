import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver


import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 2, "Epoch to train [25]")
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

@ex.config
def command_line_params():
    config = flags.FLAGS.__flags
    print(config)


@ex.automain
def start_sacred():
    main(FLAGS)
    return 0

def main(FLAGS):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(sess, ex=ex, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, y_dim=10, output_size=28, c_dim=1,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir)
        else:
            dcgan = DCGAN(sess, ex=ex, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, output_size=FLAGS.output_size, c_dim=FLAGS.c_dim,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir,gf_dim=32,df_dim=16)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)

        if FLAGS.visualize:
            # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
            #                               [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
            #                               [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
            #                               [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
            #                               [dcgan.h4_w, dcgan.h4_b, None])

            # Below is codes for visualization
            # OPTION = 2
            # visualize(sess, dcgan, FLAGS, OPTION)
            pass

if __name__ == '__main__':
    # tf.app.run()
    # main(FLAGS)
    start_sacred()

