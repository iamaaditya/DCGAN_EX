import tensorflow as tf
from generator import Generator

gen = Generator("gen_1")

z = tf.placeholder(tf.float32, [64, 100],name='z')

generated = gen.build(z)
gen.build_sampler(z)

images = tf.placeholder(tf.float32, [64] + [64, 64, 3],name='real_images')

disc = Discriminator("dis_1")

disc.build(images)
disc.build(generated)

vars = tf.all_variables()
for var in vars:
    print(var.name)
