import os
import glob
from sklearn.utils import shuffle
os.chdir('/home/zoro/Desktop/keras/dilation')
os.system('mkdir -p mydataset')
os.chdir('mydataset')
os.system('mkdir -p images')
os.system('mkdir -p classes')
os.chdir('/home/zoro/Desktop/dhall/VOC2012/SegmentationClass')
TotalDataAvailable = shuffle(sorted(glob.glob('*')))
print len(TotalDataAvailable)
newdata = TotalDataAvailable[0:len(TotalDataAvailable)//10]
for i in range(len(newdata)):
	file = newdata[i]
	os.system ('cp /home/zoro/Desktop/dhall/VOC2012/SegmentationClass/' + file\
		+ ' /home/zoro/Desktop/keras/dilation/mydataset/classes/')
	os.system ('cp /home/zoro/Desktop/dhall/VOC2012/JPEGImages/' + file[:-3] + 'jpg'\
		+ ' /home/zoro/Desktop/keras/dilation/mydataset/images/')
print len(newdata)	