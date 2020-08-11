import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os
from shutil import copyfile
import random
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

#%% transforming images

def split_and_copy(split_fraction, train_dir, valid_dir, source_dir):
    files = []
    count = 1
    for filename in os.listdir(source_dir):             
        file = source_dir + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
            count += 1
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * split_fraction)
    testing_length = int(len(files) - training_length)
    shuffle(files)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = source_dir + filename
        destination = train_dir + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = source_dir + filename
        destination = valid_dir + filename
        copyfile(this_file, destination)

#%% copying images
        
def image_copy():    
    
    if os.path.exists('image_data'):
        pass
    else:
        os.mkdir('image_data')
        os.mkdir('image_data/training/')
        os.mkdir('image_data/training/mask/')
        os.mkdir('image_data/training/no_mask/')
        os.mkdir('image_data/validation/')
        os.mkdir('image_data/validation/mask/')
        os.mkdir('image_data/validation/no_mask/')
    
    mask_images_source = 'images/with_mask/'
    no_mask_images_source = 'images/without_mask/'
    train_mask_images = 'image_data/training/mask/'
    train_no_mask_images = 'image_data/training/no_mask/'
    valid_mask_images = 'image_data/validation/mask/'
    valid_no_mask_images = 'image_data/validation/no_mask/'

    # a = len(os.listdir(no_mask_images_source))
    # b = len(os.listdir(mask_images_source))
    # n_files = a if a<b else b
    split_fraction = 0.8
    
    if os.path.exists('image_data/'):
        split_and_copy(split_fraction, train_mask_images, valid_mask_images, mask_images_source)
        split_and_copy(split_fraction, train_no_mask_images, valid_no_mask_images, no_mask_images_source)

#%% neural network
    
def neural_network(training_dir, validation_dir, n_filters, n_pixels):
    model = Sequential()
    model.add(Conv2D(filters = n_filters, kernel_size = 3, activation = 'relu', 
                      input_shape = (n_pixels, n_pixels, 3), data_format = 'channels_last', padding = 'same'))
    model.add(MaxPooling2D(2, padding = 'same'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = n_filters*2, kernel_size = 3, activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2, padding = 'same'))
    # model.add(BatchNormalization())  
    # model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = n_filters*4, kernel_size = 3, activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2, padding = 'same'))
    # model.add(BatchNormalization())  
    # model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = n_filters*8, kernel_size = 3, activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2, padding = 'same'))
    # model.add(BatchNormalization())  
    # model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = n_filters*16, kernel_size = 3, activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(2, padding = 'same'))
    # model.add(BatchNormalization())  
    # model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(256, 'relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(128, 'relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(64, 'relu'))
    # model.add(BatchNormalization())
    model.add(Dense(1, 'sigmoid'))

    model.summary()
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 30, min_lr = 0.00001, verbose = 1)
    checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True)
    stop_early = EarlyStopping(monitor = 'val_loss', patience = 50)

    train_datagen = ImageDataGenerator(
      rotation_range=20,
      rescale = 1/255,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      fill_mode='nearest',
      horizontal_flip = True
      )
    
    validation_datagen = ImageDataGenerator(
      rotation_range=20,
      rescale = 1/255,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      fill_mode='nearest',
      horizontal_flip = True
      )

    train_generator = train_datagen.flow_from_directory(training_dir, batch_size = 128, 
                                                        class_mode = 'binary', target_size = (n_pixels,n_pixels))
    valid_generator = validation_datagen.flow_from_directory(validation_dir, batch_size = 128, 
                                                             class_mode = 'binary', target_size = (n_pixels,n_pixels))
    history = model.fit(train_generator, epochs = 500, verbose = 1, validation_data = valid_generator, 
                        callbacks = [checkpoint, reduce_lr, stop_early])

    return model, history

#%% plotting data

def make_plots(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])

#%% prediction

def test_prediction(filename, model, n_pixels):
    image = img_to_array(load_img(testing_folder + filename, target_size = (n_pixels, n_pixels)), dtype = 'uint8')
    image = np.reshape(image, (-1, n_pixels, n_pixels, 3))
    result = model.predict_classes(image)
    return result

#%% evalating the model on the test data and exporting it to csv file

n_filters = 16
n_pixels = 100
# image_copy()
model, history = neural_network('image_data/training/', 'image_data/validation/', n_filters, n_pixels)
make_plots(history)
model.save('my_model')

testing_folder = 'testing_images/'
model.load_weights('model.h5')

filenames = [] 
for file in sorted(os.listdir(testing_folder)):
    filenames.append(file)

labels = [0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,0,0,0,1,1,0]
df = pd.DataFrame()  
df['target'] = labels 
df.index = filenames
df['prediction'] = labels

for file in sorted(os.listdir(testing_folder)):
    df.loc[file, 'prediction'] = test_prediction(file, model, n_pixels)[0][0]
    
print(classification_report(df.target, df.prediction))
print(confusion_matrix(df.target, df.prediction))