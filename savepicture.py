import cv2,os
import numpy as np
model = cv2.imread('/home/zoro/Desktop/model.png')
realmodel = cv2.imread('/home/zoro/Desktop/realmodel.png')
# model = np.array(model)
# realmodel = np.array(model)
# print type(model)
os.chdir('/home/zoro/Desktop/')
# model_arr =[]
# realmodel_arr = []
print model.shape,realmodel.shape
length = model.shape[0]/500
for i in range(length+1):
	if (i<=length-1):
		cv2.imwrite('model' + str(i) + '.png',model[i*500:(i+1)*500,:,:])
	else :
		cv2.imwrite('model' + str(i) + '.png',model[i*500:,:,:])	

length = realmodel.shape[0]/500

for i in range(length+1):
	if (i<=length-1):
		cv2.imwrite('realmodel' + str(i) + '.png',realmodel[i*500:(i+1)*500,:,:])
	else :
		cv2.imwrite('realmodel' + str(i) + '.png',realmodel[i*500:,:,:])			