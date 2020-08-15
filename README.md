face mask classifier that works on photos, videos and live video through webcam.

faces are first detected through MTCNN and then the detected face is passed to the mask classifier model. 
the mask classifier model is CNN based, and has a weighted accuracy of ~96% in classifying if a facial mask is present or not.

mask_classifier.py has the CNN model and its details.

face_mask_detection.py can be used to load the model with its optimal weights and run on an image/ video for classification purpose. 

myvideo1.mp4 and myvideo2.mp4 are examples of the mask classifier in action. 
