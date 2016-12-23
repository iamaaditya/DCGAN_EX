import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import numpy as np

data_dir = "./data/mnist"

fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
loaded = np.fromfile(file=fd,dtype=np.uint8)
trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

print(trX[0].shape)

plt.imshow(trX[0,:,:,0],cmap="gray")
plt.savefig("orig.png")

trX[0] = np.rot90(trX[0])

plt.imshow(trX[0,:,:,0],cmap="gray")
plt.savefig("rotated.png")
