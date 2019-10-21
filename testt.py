from skimage import io
from skimage import color
import numpy as np

rgb = io.imread('C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\ILSVRC2012_train_256\\n01855032_5783.JPEG')
print(rgb.shape, rgb.dtype)
noise = np.random.normal(0, 10, (rgb.shape[0], rgb.shape[1], rgb.shape[2]))
rgb = rgb + noise
rgb = rgb.astype(np.uint8)
print(noise.shape)
io.imshow(rgb)
io.show()
