import pandas as pa
import numpy as np
import PIL
import category_encoders as ce
import pathlib
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers

from constants import OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH, OBJECT_CLASSIFIER_FOLDER, OBJECT_BATCH_SIZE, OBJECT_EPOCHS, OBJECT_THRESHOLD

class Recognized_object:
    
    def __init__(self, main_category:str, sub_category:str, probability:float) -> None:
        self.main_category = main_category
        self.sub_category = sub_category
        self.probability = probability

class Object_finder:
    
    # Constructor creates a new image classifier and does training of AI of this classifier instance.
    def __init__(self, path:str) -> None:
        
        # Extract the data and name of this image classifier from the path.
        object_finder_name, data_dir = self.__get_dataset(path)
        self.__object_finder_name = object_finder_name
        
        # Divide data to the training and test sets.
        train_ds, val_ds, class_names = self.__divide_dataset(data_dir)
        num_classes = len(class_names)
        self.__sub_categories = class_names
        
        # Scale and shuffle data.
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        normalization_layer = layers.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Create a new neuro network model.
        self.__model = Sequential([
            layers.Rescaling(1./255, input_shape=(OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)])
        
        # Train neuro network model.
        self.__model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.__model.summary()
        history = self.__model.fit(train_ds, validation_data=val_ds, epochs=OBJECT_EPOCHS)

    # Private class method to extract data from some location.
    def __get_dataset(self, path:str) -> tuple:
        if path.find("https://") != -1 or path.find("http://") != -1:
            
            # We have an url address where to get a single file.
            try:
                object_finder_name1 = path.split("/")[-1]
                object_finder_name2 = object_finder_name1.split(".")[0]
            except:
                raise ValueError("Invalid path.")
        
            data_dir = tf.keras.utils.get_file(object_finder_name1, origin=path, extract=True)
            data_dir = pathlib.Path(data_dir).with_suffix('')
            return object_finder_name2, data_dir
        elif os.path.isdir(path):
            
            # We have a directory path where to get several pictures.
            try:
                object_finder_name = os.path.basename(path)
                object_finder_name = object_finder_name.split(".")[0]
            except:
                raise ValueError("Invalid path.")
            
            data_dir = pathlib.Path(path)
            return object_finder_name, data_dir
        elif os.path.isfile(path):
            
            # We have a file path where to get one picture.
            pass
        else:
            return None
    
    # Private class method to divide training data to training and testing parts
    def __divide_dataset(self, data_dir:pathlib.Path) -> tuple:
        
        #try:
        if True:    
            train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            #validation_split=0.2,   # KORJAA Tätä riviä pitää käyttää tuotannossa.
            validation_split=0.5,    # KORJAA Tätä riviä pitää käyttää ehdottoman minimaalisen treenidatan kanssa.
            subset="training",
            seed=123,
            image_size=(OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH),
            batch_size=OBJECT_BATCH_SIZE)
        
            val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            #validation_split=0.2,
            validation_split=0.5,
            subset="validation",
            seed=123,
            image_size=(OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH),
            batch_size=OBJECT_BATCH_SIZE)
        #except Exception as err:
            #print(f"Unexpected error when reading train data from file: {err=}, {type(err)=}")    
           # raise IOError(f"Failed to read train data.") 
        class_names = train_ds.class_names
        return train_ds, val_ds, class_names
       
    # Public class method to train more existing model.
    def add_more_training(self, path:str):
        
        # Extract the data and name of this image classifier from the path.
        object_finder_name, data_dir = self.__get_dataset(path)
        self.__object_finder_name = object_finder_name
        
        # Divide data to the training and test sets.
        train_ds, val_ds, class_names = self.__divide_dataset(data_dir)
        num_classes = len(class_names)
        self.__sub_categories = class_names
        
        # Scale and shuffle data.
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        normalization_layer = layers.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Train neuro network model.
        self.__model.summary()
        history = self.__model.fit(train_ds, validation_data=val_ds, epochs=OBJECT_EPOCHS)

    # Public class method to return the name of this image classifier I.E. the main category.
    def get_name(self) -> str:
        return self.__object_finder_name
    
    # Public class method to return a list of sub categories of the main category.
    def get_sub_categories(self) -> list:
        return self.__sub_categories
    
    # Public class method to classify an image. Use file- or url- string path for image parameter.
    def do_classification_str(self, image:str) -> Recognized_object:
        try:
            img = tf.keras.utils.load_img(image, target_size=(OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH))
        except:
            raise ValueError("Cannot find file.")
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        probabilities = self.__model.predict(img_array)
        score = tf.nn.softmax(probabilities[0])
        if np.max(score) > OBJECT_THRESHOLD:
            sub_category = self.__sub_categories[np.argmax(score)]
            return Recognized_object(self.__object_finder_name, sub_category, np.max(score))
        else:
            return None
        
    # Public class method to classify an image. Use Image type bitmap for image parameter.
    def do_classification_image(self, image:Image) -> Recognized_object:
        newsize = (OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH)
        img = image.resize(newsize);
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        probabilities = self.__model.predict(img_array)
        score = tf.nn.softmax(probabilities[0])
        if np.max(score) > OBJECT_THRESHOLD:
            sub_category = self.__sub_categories[np.argmax(score)]
            return Recognized_object(self.__object_finder_name, sub_category, np.max(score))
        else:
            return None