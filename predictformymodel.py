import numpy as np
import cv2
from trainmymodel import get_dilation_model_voc
# from dilation_net import DilationNet
# from datasets import CONFIG
from utils import interp_map
import matplotlib.pyplot as plt
from scipy.misc import imsave
from keras.utils import plot_model
# from keras.models /import load_model
from datasets import CONFIG
from keras.layers import Permute, Reshape, Activation
from keras.models import load_model
# from keras.models import model_from_json
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
def predict(image,model):
    # exit()
    image,image_size,input_dims = changesize(image = image)
    image = image.astype(np.float32) - (102.93, 111.36, 116.52)
    image = image.transpose([2,0,1])
    # image = np.reshape (image , (1,3,600,600))
    model_in = np.zeros(input_dims, dtype=np.float32)

    model_in[0] = image
    prob = model.predict(model_in)[0]
    print prob.shape
    prob = np.reshape(prob,(66,66,21))
    prob = np.reshape(prob,(21,66,66))
    # print prob.shape
    # exit()
    # return type(im_size0)
    return makeimage(prob,image_size)

def makeimage(prob,image_size):
    print prob.shape
    # exit()
    if CONFIG[ds]['zoom'] > 1:
        prob = interp_map(prob, CONFIG[ds]['zoom'], image_size[1], image_size[0])

    prediction = np.argmax(prob, axis=0)
    print prediction.shape
    # exit()
    color_image = CONFIG[ds]['palette'][prediction.ravel()].reshape(image_size)
    return color_image

ds = 'mydataset'
def changesize(image):
    conv_margin = CONFIG[ds]['conv_margin']
    print 'conv_margin = ' , conv_margin    
    input_dims = (1,) + CONFIG[ds]['input_shape']
    print 'input dims ' , input_dims
    batch_size, num_channels, input_height, input_width = input_dims
    model_in = np.zeros(input_dims, dtype=np.float32)
    image_size = image.shape
    output_height = input_height - 2 * conv_margin
    output_width = input_width - 2 * conv_margin
    print output_width , output_height
    image = cv2.copyMakeBorder(image, conv_margin, conv_margin,
                               conv_margin, conv_margin,
                               cv2.BORDER_REFLECT_101)
    # imsave('/home/zoro/Desktop/check0.jpg',image)
    print 'img_size = ',image_size
    tile = image[0:0 + input_height,0:0 + input_width, :]
    margin = [0, input_height - tile.shape[0],0, input_width - tile.shape[1]]
    tile = cv2.copyMakeBorder(tile, margin[0], margin[1],margin[2], margin[3],cv2.BORDER_REFLECT_101)
    print 'final size = ' , tile.shape
    return tile,image_size,input_dims
# im_size = []
# def interp_map(prob, zoom, width, height):
#     zoom_prob = np.zeros((prob.shape[0], height, width), dtype=np.float32)
#     for c in range(prob.shape[0]):
#         for h in range(height):
#             for w in range(width):
#                 r0 = h // zoom
#                 r1 = r0 + 1
#                 c0 = w // zoom
#                 c1 = c0 + 1
#                 rt = float(h) / zoom - r0
#                 ct = float(w) / zoom - c0
#                 v0 = rt * prob[c, r1, c0] + (1 - rt) * prob[c, r0, c0]
#                 v1 = rt * prob[c, r1, c1] + (1 - rt) * prob[c, r0, c1]
#                 zoom_prob[c, h, w] = (1 - ct) * v0 + ct * v1
#     return zoom_prob
if __name__ == '__main__':


    input_shape = (3,600,600)

    # get the model
    model = get_dilation_model_voc(input_shape = input_shape , classes=21)
    model.load_weights("/home/zoro/Desktop/keras/dilation/mydataset/weights.h5")
    # model.add(Reshape(target_shape=(66,66,21)))
    # model.add(Permute(dims=(3, 1, 2)))
    # model = load_model('/home/zoro/Desktop/keras/dilation/mydataset/weigths/model.json')
    # model.compile(optimizer='sgd', loss='categorical_crossentropy')
    # model.summary()
    # new_model = Reshape(target_shape=(66,66,21))(model.output)
    # new_model = Permute(dims=(3, 1, 2))(new_model)
    # new_model_final = Model(input = model.input, output = new_model)
    # new_model_final.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # new_model_final.summary()
    # read and predict a image
    im = cv2.imread('/home/zoro/Desktop/keras/dilation/imgs_test/voc.jpg')
    # im_size.append()
    # newmodel = 
    # model = Reshape(target_shape=(66, 66, 21))(model)
    # model = Permute(dims=(3, 1, 2))(model)
    y_img = predict(im, model)

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    a.set_title('Image')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(y_img)
    a.set_title('Semantic segmentation')
    cv2.imwrite('/home/zoro/Desktop/keras/dilation/imgs_test/voc1.png',y_img)
    plt.show(fig)
