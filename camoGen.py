import tensorflow as tf
import os
from ops import *
from glob import glob
from utils import *
import numpy as np
class Generator(object):
    def __init__(self,name,batch_size=16,output_size=32):
        self.name = name
        self.batch_size = batch_size
        self.output_size = output_size
        self.has_built = False

    def build(self,input_images,patch_masks,patch_offsets):
        with tf.variable_scope("Generator_" + self.name,reuse=self.has_built):
            self.im = input_images
            self.pm = patch_masks
            self.po = patch_offsets
            # h0 = lrelu(conv2d(input_images, 8, d_h=1,d_w=1,name='g_h0_conv'))
            # h1 = lrelu(batch_norm(conv2d(h0, 32, d_h=4,d_w=4,name='g_h1_conv'),scope="g_h1_conv"))
            # h2 = lrelu(batch_norm(conv2d(h1,128,d_h=4,d_w=4,name='g_h2_conv'),scope="g_h2_conv"))
            # h3 = lrelu(batch_norm(conv2d(h2,512,d_h=4,d_w=4,name='g_h3_conv'),scope="g_h3_conv"))
            # h4,_,_ = deconv2d(h3,[self.batch_size,8,8,128],name='g_h4_deconv', with_w=True)
            # h4 = lrelu(batch_norm(h4,scope='g_h4_deconv'))
            # h5,_,_ = deconv2d(h4,[self.batch_size,16,16,16],name='g_h5_deconv',with_w=True)
            # h5 = lrelu(batch_norm(h5,scope='g_h5_deconv'))
            # h6,_,_ = deconv2d(h5,[self.batch_size,32,32,3],name='g_h6_deconv',with_w=True)


            #smaller gen
            h0 = lrelu(conv2d(input_images, 4, d_h=2,d_w=2,name='g_h0_conv')) #128x128x4
            h1 = lrelu(batch_norm(conv2d(h0, 16, d_h=4,d_w=4,name='g_h1_conv'),scope="g_h1_conv")) #32x32x16
            h2 = lrelu(batch_norm(conv2d(h1,64,d_h=4,d_w=4,name='g_h2_conv'),scope="g_h2_conv")) #8x8x64
            h3,_,_ = deconv2d(h2,[self.batch_size,16,16,32],name='g_h3_deconv', with_w=True) #16x16x32
            h3 = lrelu(batch_norm(h3,scope='g_h3_deconv'))
            h4,_,_ = deconv2d(h3,[self.batch_size,32,32,3],name='g_h4_deconv',with_w=True) #32x32x3


            #minimal gen
            # h4 = lrelu(batch_norm(conv2d(input_images,3,d_h=8,d_w=8,name='g_h0_conv'),scope='g_h0_conv'))
            #output of 32x32x3 patches
            patches = tf.nn.tanh(h4)

            #create hole of 0s in input images
            images_with_hole = input_images*patch_masks

            #split output into a list of 64 patches
            unstacked_patches = tf.unstack(patches)
            unstacked_offsets = tf.unstack(patch_offsets)
            padded_patches = []
            for i in range(self.batch_size):
                patch = unstacked_patches[i]
                xOffset = unstacked_offsets[i][0]
                yOffset = unstacked_offsets[i][1]
                paddings = tf.stack([[xOffset,256-(32+xOffset)],[yOffset,256-(32+yOffset)],[0,0]])
                padded_patch = tf.pad(patch,paddings)
                padded_patches.append(padded_patch)

            patched_images = images_with_hole + padded_patches
            self.has_built=True
            return patched_images


# with tf.Session() as sess:
#     g = Generator("gen")
#     input = tf.placeholder(tf.float32, [64,256,256,3])
#     patch_masks_placeholder = tf.placeholder(tf.float32,[64,256,256,3])
#     patch_offsets_placeholder = tf.placeholder(tf.int32,[64,2])
#     output = g.build(input,patch_masks_placeholder,patch_offsets_placeholder)
#
#     data = glob(os.path.join("./data", "texture", "*.jpg"))
#     sample_files = data[0:64]
#     sample = [get_image(sample_file, 256, is_crop=True, resize_w=256, is_grayscale = False) for sample_file in sample_files]
#     patch_masks = np.ones([64,256,256,3])
#     patch_offsets = np.zeros([64,2])
#     for i in range(0,64):
#         randX = np.random.randint(256-32)
#         randY = np.random.randint(256-32)
#         patch_masks[i,randX:randX+32,randY:randY+32,:] = 0
#         patch_offsets[i][0] = randX
#         patch_offsets[i][1] = randY
#     tf.initialize_all_variables().run()
#     patched = output.eval({g.im: sample,g.pm: patch_masks, g.po: patch_offsets})
#     save_images(patched,[8,8],"patch.png")
