import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from adversarial_pair import Adversarial_Pair

gen = Generator("gen_1")
disc = Discriminator("dis_1")

z = tf.placeholder(tf.float32, [64, 100],name='z')

images = tf.placeholder(tf.float32, [64] + [64, 64, 3],name='real_images')

adv = Adversarial_Pair(gen,disc)
adv.build(z,images)
adv.build_loss()
adv.build_train_ops()


vars = tf.all_variables()
for var in vars:
    print(var.name)
