import os,glob
import numpy as np
import cv2
import pickle
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

def createmyarray(path):
	os.chdir('/home/zoro/Desktop/keras/dilation/mydataset/newupdatedimage/')
	output = open('/home/zoro/Desktop/keras/dilation/mydataset/outputoforiginalclassDataset.pkl', 'wb')
	path = sorted(path)
	print path
	myarr = [] # this will save all the data
	for i in range(len(path)):
		im = cv2.imread(path[i])
		print im.shape
		im = im.astype(np.float32) - CONFIG[ds]['mean_pixel']
		myarr.append(im)
	# return myarr	
	pickle.dump(myarr,output)


# if __name__ == '__main__':

os.chdir('/home/zoro/Desktop/keras/dilation/mydataset/images/')
path = sorted(glob.glob('*'))
print path
createmyarray(path)

