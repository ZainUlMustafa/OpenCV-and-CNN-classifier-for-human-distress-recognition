'''
LOAD MODEL
Author: Zain Ul Mustafa
Code Ownership: Sciengit (Copyrighted 2018)
Dataset Ownership: NUST Airworks (Copyrighted 2018)
'''
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import cv2
from time import sleep
from numpy import *
import serial
from sklearn.cross_validation import train_test_split

##########################################################################################
###LOADING MODEL###
with open("model2.json", "r") as json_file:
    model = model_from_json(json_file.read())

# Load weights into the new model
model.load_weights("model2.h5")
print("Model loaded from disk fine")

##########################################################################################
###DETECTION###
names = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
img_rows, img_cols = 100, 100
min_area = 39**2
character_detect = ''
a=b=c=k=''
cap = cv2.VideoCapture(0)

# connect to serial for GPS
try:
    data = serial.Serial('COM3', 56700)
    k = data.readline()
    k = data.readline()
    sleep(1000)
    k = ""
except:
    print("Error in port name")

# start live camera
while(True):
    sleep(100000)
    ret, im = cap.read()
    #im = cv2.resize(im, (0,0), fx=1.2, fy=1.2)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    #lower_red = np.array([0,140,59])
    #upper_red = np.array([179,255,255])
    lower_red = np.array([0,24,119])
    upper_red = np.array([4,255,255])          
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(im, im, mask=mask)
    res = cv2.GaussianBlur(res, (1,1), 0)
    cv2.imshow('mask', res)      
    
    cropped_character = cv2.resize(res, (img_rows,img_cols))
    cx = 100 
    cy = 100
    prediction = -1
    # detection of contours
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(res_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
    for contour in contours:
        area = cv2.contourArea(contour)
        if area>min_area:
            print('AREA: ', area)
            # calculating the centroid
            moment = cv2.moments(contour)
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            # make a rectangle bounding the contour
            [x, y, w, h] = cv2.boundingRect(contour)
            # draw a rectangle surrounding the contour on the trainer image
            cv2.rectangle(im, (x, y), (w+x, h+y), (0,255,0), 2)
            cropped_character = im[y:y+h, x:x+w]
            #print("cropped")
            cropped_character = cv2.cvtColor(cropped_character, cv2.COLOR_BGR2GRAY)
            cropped_character = cv2.resize(cropped_character, (0,0), fx=5.5, fy=5.5)
            # binary
            _, ct = cv2.threshold(cropped_character, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            _, ct = cv2.threshold(ct, 0, 255, cv2.THRESH_BINARY_INV)
            
            kernel = np.ones((3,3), np.uint8)
            ct = cv2.dilate(ct, kernel, iterations=1)
            ct = cv2.dilate(ct, kernel, iterations=1)
            ct = cv2.dilate(ct, kernel, iterations=1)
            
            #cv2.imshow('ct', im)
            #cv2.waitKey(0)
            #ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)
            cv2.imshow('ct', ct)
            ct = cv2.resize(ct, (img_rows,img_cols))
            ct = np.array(ct)
            ct = ct.astype('float32')
            ct /= 255
            
            ct = np.expand_dims(ct, axis = 3)
            ct = np.expand_dims(ct, axis = 0)
            prediction = model.predict_classes(ct)
            #character_detect = ""
        else:
            prediction = -1
        #endif
    #endfor
    
    if prediction == -1:
        print(-1)
    elif prediction != -1:
        try:
            k = data.readline()
            a = k[20:29]
            b = k[32:42]
            c = a+'N, '+b+'E'
        except:
            c = 'no data'
        character_detect = names[int(prediction)]
        with open("coor.txt", "w") as text_file:
            text_file.write(character_detect +', '+ c)
            text_file.write('\n')
        #endwith
    #endif
    
    #cv2.circle(im, (cx,cy), 5, (255,0,0), 2)
    cv2.putText(im, str(character_detect)+', '+str(c), (cx-50,cy+150), 1, 2, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('Detect', im)
    #cv2.imshow('thresh', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #endif
#endwhile

cap.release()
cv2.destroyAllWindows()
###DETECTION END###
##########################################################################################