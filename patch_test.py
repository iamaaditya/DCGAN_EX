import os
from glob import glob
from utils import *
import numpy as np

data = glob(os.path.join("./data", "texture", "*.jpg"))
sample_files = data[0:64]
sample = [get_image(sample_file, 256, is_crop=True, resize_w=256, is_grayscale = False) for sample_file in sample_files]

patch_masks = np.ones([64,256,256,3])
for i in range(0,64):
    randX = np.random.randint(256-32)
    randY = np.random.randint(256-32)
    patch_masks[i,randX:randX+32,randY:randY+32,:] = 0

sample = sample*patch_masks

patches = np.ones(64,32,32,3)
for i in range(0,64):
    padded = 

save_images(sample,[8,8],"patch.png")
