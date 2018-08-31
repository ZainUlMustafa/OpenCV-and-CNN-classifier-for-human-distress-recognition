# OpenCV-and-CNN-classifier-for-human-distress-recognition
To recognize the human distress, OpenCV is used to process the image first into a way that it gets easier and less faulty for the classifer to classify 36 total distress characters. It is developed for drone research for IMecheUASChallenge by Sciengit and NUST Airworks in 2018
<hr>

<img src="https://raw.githubusercontent.com/ZainUlMustafa/OpenCV-and-CNN-classifier-for-human-distress-recognition/master/predictions%20(1).PNG" width="70%" height="70%">

<img src="https://raw.githubusercontent.com/ZainUlMustafa/OpenCV-and-CNN-classifier-for-human-distress-recognition/master/predictions%20(2).PNG" width="70%" height="70%">

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

Request to dowload our dataset: 
<p>>>> <a href="mailto:sciengit@gmail.com">sciengit@gmail.com</a></p>
<p>>>> <a href="mailto:nust.airworks@gmail.com">nust.airworks@gmail.com</a></p>

Before running this code, you must:
1) Unzip the dataset file you get from us (TOTAL.zip)
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
