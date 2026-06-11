import fastbook
fastbook.setup_book()

from fstai.visin.all import *
from fastbook import *

matplotlib.rc('image',cmap='Greys')

path = untar_data(URLs.MNIST_SAMPLE)
path.BASE_PATH = path

#path.ls()

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)

dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs,dist_3_sqr

#out []: (tensor(0.1114), tensor(0,2021))
data= [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)
