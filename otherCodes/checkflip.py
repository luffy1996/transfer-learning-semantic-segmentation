import pickle
import numpy as np
file = open('/home/zoro/Desktop/keras/dilation/mydataset/probabilitydata.pkl','r')
data = pickle.load(file)
data = np.array(data)
print data.shape