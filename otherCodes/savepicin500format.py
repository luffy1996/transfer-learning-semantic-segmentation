import numpy as np
import cv2,os,glob
from scipy.misc import imsave
# from datasets import CONFIG
# conv_margin = 36
# imput_shape = (600,600)

ds = 'voc12'

def saveUpdateImage(path):
	os.chdir('/home/zoro/Desktop/keras/dilation/mydataset/images')
	image = cv2.imread(path)
	# print im.shape
	# image = image.astype(np.float32) - CONFIG[ds]['mean_pixel']
	conv_margin = CONFIG[ds]['conv_margin']
	# print 'conv_margin = ' , conv_margin	
	input_dims = (1,) + CONFIG[ds]['input_shape']
	# print 'input dims ' , input_dims
	batch_size, num_channels, input_height, input_width = input_dims
	model_in = np.zeros(input_dims, dtype=np.float32)
	image_size = image.shape
	output_height = input_height - 2 * conv_margin
	output_width = input_width - 2 * conv_margin
	# print output_width , output_height
	image = cv2.copyMakeBorder(image, conv_margin, conv_margin,
							   conv_margin, conv_margin,
							   cv2.BORDER_REFLECT_101)
	num_tiles_h = image_size[0] // output_height + (1 if image_size[0] % output_height else 0)
	num_tiles_w = image_size[1] // output_width + (1 if image_size[1] % output_width else 0)
	# print num_tiles_h, num_tiles_w
	row_prediction = []
	count = 0 
	for h in range(num_tiles_h):
		col_prediction = []
		for w in range(num_tiles_w):
			offset = [output_height * h,
					  output_width * w]
			# print ('offset ',h,offset,len(offset))
			tile = image[offset[0]:offset[0] + input_height,
						 offset[1]:offset[1] + input_width, :]
			# print ('tile ',h,tile,len(tile))	             
			margin = [0, input_height - tile.shape[0],
					  0, input_width - tile.shape[1]]
			print ('margin ',h,margin,len(margin))          
			tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
									  margin[2], margin[3],
									  cv2.BORDER_REFLECT_101)
			image = tile           
	# imsave('/home/zoro/Desktop/check0.jpg',image)
	# print 'img_size = ',image_size
	imsave('/home/zoro/Desktop/keras/dilation/mydataset/newupdatedimage/'\
			+ my_made_dataset[i],image)
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
if __name__ == '__main__':
	os.chdir('/home/zoro/Desktop/keras/dilation/mydataset/images')
	my_made_dataset = sorted(glob.glob('*'))
	for i in range(len(my_made_dataset)):
		saveUpdateImage(my_made_dataset[i])
		# break