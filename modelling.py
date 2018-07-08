'''
MODEL CREATION
Author: Zain Ul Mustafa
Code Ownership: Sciengit (Copyrighted 2018)
Dataset Ownership: NUST Airworks (Copyrighted 2018)
'''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, adadelta
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import cv2
from PIL import Image
from numpy import *
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from tensorflow.contrib.estimator.python.estimator.extenders import add_metrics

def main():
    img_rows, img_cols = 100, 100
    img_data_list=[]

    batch_size = 32
    nb_epoch = 35
    img_channels = 1    #grayscale
    nb_filters = 32
    nb_pool = 2
    nb_conv = 5
    
    path1 = "/TOTAL/"
    path2 = "/TOTAL_res/"
    ##################################################################################
    '''READING ORIGINAL IMAGES'''
    listing = os.listdir(path1)
    num_samples = size(listing)
    print(num_samples)
    ##################################################################################
    '''READING AND RESIZING'''
    for file in listing:
        im = cv2.imread(path1 + file)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_resize = cv2.resize(im_gray, (img_rows,img_cols))
        cv2.imwrite(path2 + file, im_resize)
        img_data_list.append(im_resize)
        print(path2+file)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #endif
    #endfor

    imlist = os.listdir(path2)
    num_samples = size(imlist)
    print(num_samples)
    ##################################################################################
    '''ADJUSTING THE LIST'''
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    print('ATL')
    print(img_data.shape)    #(7200,100,100)
    ##################################################################################
    '''CHANGING THE SHAPE'''
    img_data = np.expand_dims(img_data, axis=4)
    print('CTS')
    print(img_data.shape)    #(7200,100,100,1)
    ###################################################################################
    '''FLATTENING THE LIST'''
    '''def image_to_feature_vector(image, size=(img_rows, img_cols)):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return cv2.resize(image, size).flatten()
    #enddef
    img_data_list=[]
    for file in listing:
        inp = cv2.imread(path1 + file)
        inp_gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
        inp_img_flatten=image_to_feature_vector(inp_gray,(img_rows,img_cols))
        img_data_list.append(inp_img_flatten)
    #endfor
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    print('FTL1')
    print(img_data.shape)
    img_data_scaled = preprocessing.scale(img_data)
    print('FTL2')
    print(img_data_scaled.shape)    #(900,40000)'''
    ###################################################################################
    '''RESHAPING'''
    '''img_data_scaled = img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,img_channels)
    print('R')
    print(img_data_scaled.shape)'''
    ###################################################################################
    '''ADDING LABELS TO IMAGES'''
    nb_classes = 36
    num_samples = img_data.shape[0]
    label = np.ones((num_samples,), dtype = int)
    label[0:201] = 0#0
    label[201:402] = 1 #1
    label[402:603] = 2#2
    label[603:804] = 3#3
    label[804:1005] = 4#4
    label[1005:1206] = 5#5
    label[1206:1407] = 6#6
    label[1407:1608] = 7#7
    label[1608:1809] = 8#8
    label[1809:2010] = 9#9
    label[2010:2211] = 10#A
    label[2211:2412] = 11#B
    label[2412:2613] = 12#C
    label[2613:2814] = 13#D
    label[2814:3015] = 14#E
    label[3015:3216] = 15#F
    label[3216:3417] = 16#G
    label[3417:3618] = 17#H
    label[3618:3819] = 18#I
    label[3818:4020] = 19#J
    label[4020:4221] = 20#K
    label[4221:4422] = 21#L
    label[4422:4623] = 22#M
    label[4623:4824] = 23#N
    label[4824:5025] = 24#O
    label[5025:5226] = 25#P
    label[5226:5427] = 26#Q
    label[5427:5628] = 27#R
    label[5628:5829] = 28#S
    label[5829:6030] = 29#T
    label[6030:6231] = 30#U
    label[6231:6432] = 31#V
    label[6432:6633] = 32#W
    label[6633:6834] = 33#X
    label[6834:7035] = 32#Y
    label[7035:7236] = 35#Z
    names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    # one-hot encoding
    Y = np_utils.to_categorical(label, nb_classes)
    ###################################################################################
    '''SHUFFLING AND SPLITTING'''
    x, y = shuffle(img_data, Y, random_state=4)
    print(y[:10])
    X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    ####################################################################################
    '''CNN MODELLING'''
    input_shape = img_data[0].shape
    print('CM')
    print(input_shape)

    model = Sequential()
    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same' , input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.6))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # summary of the model
    model.summary()
    model.get_config()
    model.layers[0].get_config()
    model.layers[0].input_shape            
    model.layers[0].output_shape            
    model.layers[0].get_weights()
    np.shape(model.layers[0].get_weights()[0])
    model.layers[0].trainable
    
    # training starts here
    hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_test, y_test))
    hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,  verbose=1, validation_split=0.2)
    
    # calculating the score
    score = model.evaluate(x_test, y_test, verbose=0)
    
    #####################################################################################
    '''SAVING THE MODEL'''
    # serialize model to JSON
    model_json = model.to_json()
    with open("/model2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/model2.h5")
    print("Saved model to disk")
    
    #####################################################################################
    '''DISPLAY THE SCORE'''
    # visualizing losses and accuracy
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)
    
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.show()
    
    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.show()
    ####################################################################################
    '''DISPLAY SCORE'''
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])
    print(model.predict(x_test[1:10]))
    print(model.predict_classes(x_test[1:10]))
    print(y_test[1:10])
    #####################################################################################
    '''TESTING A NEW IMAGE'''
    test_image = cv2.imread('/sample_data/thresh/pic1.jpg')
    test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image=cv2.resize(test_image,(img_rows,img_cols))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    print(test_image.shape)

    test_image = np.expand_dims(test_image, axis = 3)
    test_image = np.expand_dims(test_image, axis = 0)
    print(test_image.shape)
    
    print((model.predict(test_image)))
    print(model.predict_classes(test_image))
    
    return
#enddef

if __name__ == '__main__':
    main()