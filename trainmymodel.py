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
from keras.models import load_model

from keras.layers import BatchNormalization
# class_weights = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

nb_classes = 21
#######################################################################################################

ds = 'voc12'
CONFIG = {
	'voc12': {
		'classes': 21,
		'weights_file': 'dilation_voc12.h5',
		'weights_url': 'http://imagelab.ing.unimore.it/files/dilation_keras/voc12.h5',
		'input_shape': (3, 600, 600),
		'test_image': 'imgs_test/voc.jpg',
		'mean_pixel': (102.93, 111.36, 116.52),
		'palette': np.array([[0, 0, 0],
							[128, 0, 0],
							[0, 128, 0],
							[128, 128, 0],
							[0, 0, 128],
							[128, 0, 128],
							[0, 128, 128],
							[128, 128, 128],
							[64, 0, 0],
							[192, 0, 0],
							[64, 128, 0],
							[192, 128, 0],
							[64, 0, 128],
							[192, 0, 128],
							[64, 128, 128],
							[192, 128, 128],
							[0, 64, 0],
							[128, 64, 0],
							[0, 192, 0],
							[128, 192, 0],
							[0, 64, 128]], dtype='uint8'),
		'zoom': 8,
		'conv_margin': 36
	}
	}
#######################################################################################################

input_shape = (3,600,600)

def get_dilation_model_voc(input_shape,  classes):

	# if input_tensor is None:
	#     model_in = Input(shape=input_shape)
	# else:
		# if not K.is_keras_tensor(input_tensor):
		#     model_in = Input(tensor=input_tensor, shape=input_shape)
		# else:
		#     model_in = input_tensor
	model_in = Input(shape=input_shape)    

	h = Convolution2D(32, 3, 3, activation='relu', name='conv1_1')(model_in)
	h = Convolution2D(32, 3, 3, activation='relu', name='conv1_2')(h)
	h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)

	h = Convolution2D(64, 3, 3, activation='relu', name='conv2_1')(h)
	h = Convolution2D(64, 3, 3, activation='relu', name='conv2_2')(h)
	h = Convolution2D(128, 3, 3, activation='relu', name='conv2_3')(h)
	h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
	
	h = AtrousConvolution2D(128, 5, 5, atrous_rate=(2, 2), activation='relu', name='dilation1_1')(h)
	h = AtrousConvolution2D(128, 7, 7, atrous_rate=(4, 4), activation='relu', name='dilation1_2')(h)
	
	h = Dropout(0.5, name='drop1')(h)
	h = Convolution2D(512, 1, 1, activation='relu', name='fc7')(h)
	h = Dropout(0.5, name='drop2')(h)   
	# h = BatchNormalization()(h)
	h = AtrousConvolution2D(4 * classes, 5, 5, atrous_rate=(5, 5), activation='tanh', name='dilation2_1')(h)
	h = AtrousConvolution2D(4 * classes, 5, 5, atrous_rate=(5, 5), activation='tanh', name='dilation2_2')(h)
	h = AtrousConvolution2D(4 * classes, 3, 3, atrous_rate=(4, 4), activation='tanh', name='dilation2_3')(h)

	logits = Convolution2D(classes, 1, 1, name='logits')(h)
	logits = Permute(dims=(2, 3, 1))(logits)
	# logits = Reshape(target_shape=(66*66 ,21))(logits)
	model_out = Activation('softmax')(logits)
	# model_out = Reshape(target_shape=(66*66,21))(model_out)
	# model_out = softmax(logits)
	# model_out.load_weights("/home/zoro/Desktop/keras/dilation/mydataset/weigths/weights0.h5")
	model = Model(input=model_in, output=model_out, name='dilation_voc12')
	# model.load_weights("/home/zoro/Desktop/keras/dilation/mydataset/weigths/weights0.h5")
	return model
# class WeightsSaver(Callback):
#     def __init__(self, model, N):
#         self.model = model
#         self.N = N
#         self.batch = 0

#     def on_batch_end(self, batch, logs={}):
#         if self.batch % self.N == 0:
#             name = 'weights%08d.h5' % self.batch
#             self.model.save_weights('/home/zoro/Desktop/keras/dilation/mydataset/weights/' + name)
#         self.batch += 1
def loadimagedataset():
	file = open('/home/zoro/Desktop/keras/dilation/mydataset/pictureDataset.pkl','r')
	imagedataset = pickle.load(file)
	imagedataset = np.asarray(imagedataset)
	print imagedataset.shape , 'done'
	return imagedataset
def loadprobabilitydataset():
	file = open('/home/zoro/Desktop/keras/dilation/mydataset/probabilitydata.pkl','r')
	probabilityValues = pickle.load(file)
	probabilityValues = np.asarray(probabilityValues)
	print probabilityValues.shape
	return probabilityValues

if __name__ == '__main__':
	sgd = SGD(lr=1, momentum=0.9, decay=0.03 )
	model = get_dilation_model_voc(input_shape = input_shape,classes=21)
	model.compile(optimizer='Adadelta', loss='mean_squared_error',metrics=['accuracy'])
	model.summary()
	plot_model(model,show_shapes=True,show_layer_names=True ,to_file='/home/zoro/Desktop/model.png')
	# exit()
	imagedataset = loadimagedataset()
	for i in range(len(imagedataset)):
		imagedataset[i] = imagedataset[i].astype(np.float32) - CONFIG[ds]['mean_pixel']

	print 'reached'

	imagedataset = imagedataset.transpose([0,3,1,2])/255
	probabilityValues = loadprobabilitydataset()
	print probabilityValues.shape
	# exit()
	probabilityValues = np.reshape(probabilityValues,(291,21,66,66))
	probabilityValues = probabilityValues.transpose([0,2,3,1])
	# probabilityValues = np.reshape(probabilityValues,(291,66*66,21))
	print 'reached'
	
	
	
	
	# model.load_weights("/home/zoro/Desktop/keras/dilation/mydataset/weigths/weights0.h5")
	print 'weight loaded'
	csv_logger = CSVLogger('/home/zoro/Desktop/keras/dilation/mydataset/weigths/log.csv', append=True, separator=';')
	# filepath = '/home/zoro/Desktop/keras/dilation/mydataset/weigths/' 
	# modelcheckpoint = ModelCheckpoint(filepath = filepath, monitor='val_loss',save_best_only=False ,save_weights_only=True,period = 10) 
	callbacks_list = [csv_logger]
	model.fit(imagedataset,probabilityValues,batch_size = 2 , nb_epoch=40,validation_split = 0.2, callbacks=callbacks_list)
	print 'fit ended'
	# model_json = model.to_json()#to save the model 
	# with open("/home/zoro/Desktop/keras/dilation/mydataset/model.json", "w") as json_file:
	# 	json_file.write(model_json)
		# serialize weights to HDF5
	model.save_weights("/home/zoro/Desktop/keras/dilation/mydataset/weights.h5")
	print("Saved model to disk")


