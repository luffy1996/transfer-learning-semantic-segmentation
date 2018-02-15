import numpy as np
import cv2
from dilation_net import DilationNet
from datasets import CONFIG
from utils import interp_map
import matplotlib.pyplot as plt
from scipy.misc import imsave
import glob,os
import pickle

# predict function, mostly reported as it was in the original repo
def predict(image, model, ds):

    image = image.astype(np.float32) - CONFIG[ds]['mean_pixel']
    conv_margin = CONFIG[ds]['conv_margin']
    # print 'conv_margin = ' , conv_margin	
    input_dims = (1,) + CONFIG[ds]['input_shape']
    # print 'input dims ' , input_dims
    batch_size, num_channels, input_height, input_width = input_dims
    model_in = np.zeros(input_dims, dtype=np.float32)
    # print '############################################'
    # print '############################################'
    # print '############################################'
    # print model_in.shape
    # print input_height , input_width
    # print '############################################'
    # print '############################################'
    image_size = image.shape
    output_height = input_height - 2 * conv_margin
    output_width = input_width - 2 * conv_margin
    # print output_width , output_height
    image = cv2.copyMakeBorder(image, conv_margin, conv_margin,
                               conv_margin, conv_margin,
                               cv2.BORDER_REFLECT_101)
    # imsave('/home/zoro/Desktop/check0.jpg',image)
    # print 'img_size = ',image_size
    num_tiles_h = image_size[0] // output_height + (1 if image_size[0] % output_height else 0)
    num_tiles_w = image_size[1] // output_width + (1 if image_size[1] % output_width else 0)
    # # print num_tiles_h, num_tiles_w
    row_prediction = []
    count = 0 
    for h in range(num_tiles_h):
        col_prediction = []
        for w in range(num_tiles_w):
            offset = [output_height * h,
                      output_width * w]
            # # print ('offset ',h,offset,len(offset))
            tile = image[offset[0]:offset[0] + input_height,
                         offset[1]:offset[1] + input_width, :]
            # # print ('tile ',h,tile,len(tile))	             
            margin = [0, input_height - tile.shape[0],
                      0, input_width - tile.shape[1]]
            # print ('margin ',h,margin,len(margin))          
            tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                      margin[2], margin[3],
                                      cv2.BORDER_REFLECT_101)
            model_in[0] = tile.transpose([2, 0, 1])
            # # print type(model_in) , model_in.shape
            # predicted_model = model.predict(model_in)
            # # print (np.asarray(predicted_model)).shape
            prob = model.predict(model_in)[0]
            # print len(prob),len(prob[0]) , len(prob[0][0])
            col_prediction.append(prob)
            # # print h,col_prediction
            # print '##############################################'
            count = count + 1
            # print h , w , count
        col_prediction = np.concatenate(col_prediction, axis=2)
        row_prediction.append(col_prediction)
    prob = np.concatenate(row_prediction, axis=1)
    # print 'check', prob.shape 
    # if CONFIG[ds]['zoom'] > 1:
    #     prob = interp_map(prob, CONFIG[ds]['zoom'], image_size[1], image_size[0])

    # prediction = np.argmax(prob, axis=0)
    # # print prediction.shape
    # # # print 'dikha bhai' , prediction.shape
    # # # print prediction[0:100,0:100]
    # # color_image = CONFIG[ds]['palette'][prediction.ravel()].reshape(image_size)
    # # # print 'size dekh lo bhai' , color_image.shape
    prob = np.reshape(prob,(21*66*66))
    return prob.astype(dtype=np.float16)

if __name__ == '__main__':

    ds = 'voc12'  # choose between cityscapes, kitti, camvid, voc12
    finaldata = []

    # get the model
    model = DilationNet(dataset=ds)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    # model.summary()

    # read and predict a image
    os.chdir('/home/zoro/Desktop/keras/dilation/mydataset/images')
    my_made_dataset = glob.glob('*')
    # file = open('/home/zoro/Desktop/keras/dilation/mydataset/classtextfile.txt','w')
    output = open('/home/zoro/Desktop/keras/dilation/mydataset/probabilitydata.pkl', 'wb')
    for i in range(len(my_made_dataset)):
        im = cv2.imread(my_made_dataset[i])
        # # print im.shape
        # im = cv2.read('/home/zoro/Desktop/keras/dilation/imgs_test/'+ds+'.png')
        probability = predict(im, model, ds)
        # img_shape = probability.shapee
        # file.write(str(img_shape[0]) + ' ' + str(img_shape[1]))
        
        finaldata.append(probability)
        # prob = prob.tolist()
        # # prob = map(str,prob)
        # for i in range(len(prob)):
        #     file.write (str(prob[i]) + ' ')
        # file.write ('\n')
        # imsave('/home/zoro/Desktop/keras/dilation/mydataset/predictionFromOriginalWeights/'\
            # + my_made_dataset[i],y_img)
        # if (i == 5):
        #     break        
        # del y_img,im
        # break
        # # print type(y_img)
        # # print y_img.shape
        # plot results
        # fig = plt.figure()
        # a = fig.add_subplot(1, 2, 1)
        # imgplot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # a.set_title('Image')
        # a = fig.add_subplot(1, 2, 2)
        # imgplot = plt.imshow(y_img)
        # a.set_title('Semantic segmentation')
        # plt.show(fig)
    # file.close    
    pickle.dump(finaldata,output)