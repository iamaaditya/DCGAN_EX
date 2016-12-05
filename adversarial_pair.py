import tensorflow as tf
class Adversarial_Pair(object):
    def __init__(self,generator,discriminator):
        self.generator=generator
        self.discriminator=discriminator

    def build(self,z,images,y=None):
        self.G = self.generator.build(z,y)
        self.D,self.D_logits = self.discriminator.build(images,y)
        self.D_,self.D_logits_ = self.discriminator.build(self.G,y)
        self.sampler = self.generator.build_sampler(z,y)

    def build_loss(self):
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

    def build_train_ops(self,config):

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'Discriminator_'+discriminator.model_name in var.name]
        self.g_vars = [var for var in t_vars if 'Generator_'+generator.model_name in var.name]

        self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
