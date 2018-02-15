import numpy as np
import cv2
# from dilation_net import DilationNet
# from datasets import CONFIG
from utils import interp_map
# import matplotlib.pyplot as plt
import theano.tensor as T
from keras import backend as K
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Input, AtrousConvolution2D
from keras.layers import Dropout, UpSampling2D, ZeroPadding2D,Multiply,Dot
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.utils import plot_model
from utils import softmax
import pickle
import pydot
from keras.layers import Permute, Reshape, Activation
from keras.callbacks import CSVLogger,ModelCheckpoint
from keras.optimizers import SGD
if __name__ == '__main__':


    input_shape = (3,600,600)

    # get the model
    model = get_dilation_model_voc(input_shape = input_shape , classes=21)
    model.load_weights("/home/zoro/Desktop/keras/dilation/mydataset/weigths/trail1/trial1_50epcoches.h5")
    # model = load_model('/home/zoro/Desktop/keras/dilation/mydataset/weigths/model.json')
    # model.compile(optimizer='sgd', loss='categorical_crossentropy')
    # model.summary()	
    model = Reshape(target_shape=(66, 66, 21))(x)
    model = Permute(dims=(3, 1, 2))(model)
    # model = 