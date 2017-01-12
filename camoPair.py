class Pair(object):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def build(self):
        self.real_images = tf.placeholder(tf.float32,[self.generator.batch_size,256,256,3],name='real_images')
        self.patch_masks = tf.placeholder(tf.float32,[self.generator.batch_size,256,256,3],name='patch_masks')

        self.G = self.generator.build(self.real_images, self.patch_masks)
        self.D, self.D_logits = self.discriminator.buil(self.real_images)
        self.D_,self.D_logits_ = self.discriminator.build(self.G)

    def build_loss(self):
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,tf.zeros_like(self.D_logits_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,tf.ones_like(self.D_logits_)))

        self.d_loss = self.d_loss_real+self.d_loss_fake

    def build_train_ops(self):
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'Discriminator_'+self.discriminator.model_name in var.name]
        self.g_vars = [var for var in t_vars if 'Generator_'+self.generator.model_name in var.name]

        self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
