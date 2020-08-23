# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 19:51:06 2020

@author: diggee
"""

#%% library imports

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from mtcnn.mtcnn import MTCNN
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #to force the tf model to load into CPU and not GPU    

#%% read image
    
def read_image(testing_folder, image_filename, w_pixels, h_pixels):
    image = cv2.imread(testing_folder + image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w_pixels, h_pixels))
    return image   
            
#%% detecting face using MTCNN
    
def detect_face(image, face_detection_threshold):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_rects = []
    for i in range(len(faces)):
        if faces[i]['confidence'] > face_detection_threshold:
            face_rects.append(faces[i]['box'])
    return face_rects

#%% preprocessing detected face image
    
def preprocess_face_image(face_image, n_pixels):
    face_image = face_image/255
    face_image = cv2.resize(face_image, dsize = (n_pixels, n_pixels), interpolation = cv2.INTER_CUBIC)
    face_image = np.reshape(face_image, (-1, n_pixels, n_pixels, 3))
    return face_image

#%% mask detection model import

def mask_model():    
    my_model = tf.keras.models.load_model('my_model')
    my_model.load_weights('model.h5')
    return my_model

#%% mask detection
    
def detect_mask(face_image, my_model):
    result = my_model.predict_classes(face_image)
    return result

#%% drawing result
    
def draw_on_image(result, image, x, y, w, h, font_scale, rectangle_thickness, font_thickness):
    if result == 0:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), rectangle_thickness) 
        cv2.putText(image, 'mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), font_thickness)
    else:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), rectangle_thickness) 
        cv2.putText(image, 'no mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
    return image

#%% read video from disk
    
def read_video(testing_folder, video_filename, w_pixels, h_pixels):
    cap = cv2.VideoCapture(testing_folder + video_filename)   
    writer = cv2.VideoWriter('myvideo7.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30,  (w_pixels, h_pixels))
    while cap.isOpened():
        ret, image = cap.read()
        if ret == True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (w_pixels, h_pixels))
            face_rects = detect_face(image, face_detection_threshold)
            my_model = mask_model()
            for (x,y,w,h) in face_rects: 
                face_image = preprocess_face_image(image[y:y+h, x:x+w], n_pixels)  
                result = detect_mask(face_image, my_model)
                image = draw_on_image(result, image, x, y, w, h, font_scale, rectangle_thickness, font_thickness)
            cv2.imshow('window', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()  
    writer.release()
    cv2.destroyAllWindows()   

#%% read video from webcam
    
def webcam_video(w_pixels, h_pixels):
    cap = cv2.VideoCapture(0)    
    while cap.isOpened():
        ret, image = cap.read()
        if ret == True:
            face_rects = detect_face(image, face_detection_threshold)
            my_model = mask_model()
            for (x,y,w,h) in face_rects: 
                face_image = preprocess_face_image(image[y:y+h, x:x+w], n_pixels) 
                result = detect_mask(face_image, my_model)
                image = draw_on_image(result, image, x, y, w, h, font_scale, rectangle_thickness, font_thickness)
            cv2.imshow('window', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()  
    cv2.destroyAllWindows() 
    
#%% main()
    
if __name__ == '__main__':
    font_scale = 1
    font_thickness = 2
    rectangle_thickness = 3
    face_detection_threshold = 0.9
    n_pixels = 100      # (pixels x pixels) of face image required to feed into CNN model
    w_pixels = 1920      # horizontal pixels (width) that input image/ video frame will be resized to
    h_pixels = 1080       # vertical pixels (height) that input image/ video frame will be resized to
    
    while True:
        try:
            choice = int(input('What would you like to run the face mask detector on?\n1. Image\n2. Video file on disk\n3. Video from webcam\n'))
        except ValueError:
            print('please enter a number')
            continue    
        if choice not in [1,2,3]:
            print('Please enter 1, 2 or 3')
            continue
        else:
            print('selected choice is ' + str(choice))  
            break
           
    if choice == 1:
        testing_folder = 'final_testing/'
        image_filename = 'vindhya.jpg'
        image = read_image(testing_folder, image_filename, w_pixels, h_pixels)
        face_rects = detect_face(image, face_detection_threshold)
        my_model = mask_model()
        for (x,y,w,h) in face_rects: 
            face_image = preprocess_face_image(image[y:y+h, x:x+w], n_pixels)  
            result = detect_mask(face_image, my_model)
            image = draw_on_image(result, image, x, y, w, h, font_scale, rectangle_thickness, font_thickness)
        cv2.imshow('window', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(-1)
        
    elif choice == 2:
        testing_folder = 'final_testing/'
        video_filename = 'video7.mp4'
        read_video(testing_folder, video_filename, w_pixels, h_pixels)
        
    else:
        webcam_video(w_pixels, h_pixels)    
    
