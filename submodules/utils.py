import numpy as np
from PIL import Image


"""
numpy to Image et vice versa
"""
def convert_from_np_to_image(img):
    # return Image.fromarray(img)
    return Image.fromarray(img.astype(np.uint8))



"""
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
"""
def normalize(img):
    arr = np.array(img)
    arr = arr.astype('float')

    # Do not touch the alpha channel
    print(img)
    minval = arr.min()
    maxval = arr.max()

    arr -= minval
    arr *= (255.0/(maxval-minval))
    print(arr.min())
    print(arr.max())
    return Image.fromarray(arr.astype('uint8'),'L')
