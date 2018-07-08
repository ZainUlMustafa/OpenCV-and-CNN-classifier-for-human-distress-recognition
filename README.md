# OpenCV-and-CNN-classifier-for-human-distress-recognition
To recognize the human distress, OpenCV is used to process the image first into a way that it gets easier and less faulty for the classifer to classify 36 total distress characters. It is developed for drone research for IMecheUASChallenge by Sciengit and NUST Airworks in 2018

<hr>
<br>
<b>THIS PROGRAM IS WRITTEN BY ZAIN UL MUSTAFA IN A RESEARCH BASED PROJECT
OWNED BY SCIENGIT. DATASET IS EXPLICITLY PROVIDED AND COLLECTED UNDER
NUST AIRWORKS WHICH AS PER REGULATION IS COPYRIGHTED. DATASET AS A WHOLE
IS NOT PROVIDED WITH THIS PROJECT OF SCIENGIT</b>

<hr>
Sciengit 2018
Author: Zain Ul Mustafa
<hr>

This code is created for the recognition of signs of distress. Every alphabet and the numerics which 
accounts for a total of 36 characters means something which can be assigned by you with anything
e.g A means NEED WHEAT, B means NEED RICE, E means MEDICAL EMERGENCY, and etc 

Convolutional Neural Network (CNN) is used as a classifier.

A microcontroller Arduino with a GPS is connected to the the computer which throws the GPS RMC data
to the serial port which is read by this software. So whenever a character is detected, not only
the character gets detected but also its GPS location is known.

We have trained two different models in which we changed few parameters to just see 
which turns out to be better. An average of 85% accuracy is achieved using the current
parameters.

If you want, change number of epoch, number of neurons, number of hidden layers, 
filter window size etc to see what turns out to be the result that suits you the best.

Few images that Sciengit owns are contained in sample_data folder
so you can see what we have trained using this technique and modify them and test them.

Results of this model is provided in the repository.

Download the TOTAL.zipper from this link: <a href="http://bit.ly/NAWData2018">bit.ly/NAWData2018</a>
and rename it to TOTAL.zip

Before running this code, you must:
1) Unzip the TOTAL.zip
2) Make a folder TOTAL_res in the same directory

<hr>
Your project folder should be in this hierarchy:
<br>
Main folder
<br>
''''''''TOTAL
<br>
'''''''''''''''''''''0 (1).jpg
<br>
''''''''TOTAL_res
<br>
''''''''modelling.py
<br>
''''''''load_model.py
<br>
''''''''model.h5
<br>
''''''''model.json
