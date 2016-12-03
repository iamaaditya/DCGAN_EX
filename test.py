import tensorflow as tf
from generator import Generator

gen = Generator("gen_1")

z = tf.placeholder(tf.float32, [64, 100],name='z')

gen.build(z)
gen.build_sampler(z)
