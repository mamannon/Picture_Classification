from genericpath import isdir
import pandas as pa
import numpy as np
import PIL
import category_encoders as ce
import pathlib
import matplotlib.pyplot as plt
import pickle
import os
import pillow_avif
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
from image_dataset_utils import paths_and_labels_to_dataset
from image_dataset_utils import image_dataset_from_directory

from constants import OBJECT_CLASSIFIER_IMPERTINENT_CLASS_NAME, OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH, OBJECT_CLASSIFIER_FOLDER, OBJECT_BATCH_SIZE, OBJECT_EPOCHS, OBJECT_THRESHOLD

class Recognized_object:
    
    def __init__(self, main_category:str, sub_category:str, probability:float) -> None:
        self.main_category = main_category
        self.sub_category = sub_category
        self.probability = probability

class Object_finder:
    
    # Constructor creates a new image classifier and does training of AI of this classifier instance.
    def __init__(self, path:str, add_impertinent:bool=False) -> None:
        
        # Extract the data and name of this image classifier from the path.
        object_finder_name, data_dir = self.__get_dataset(path)
        self.__object_finder_name = object_finder_name
        
        # Divide data to the training and test sets and get class names.
        train_ds, val_ds, sub_categories = self.__divide_dataset(data_dir)
        num_classes = len(sub_categories)
        self.__sub_categories = sub_categories
                
        # Add "not classified" ie. impertinent training data to the training set. Impertinent data belongs to training data of
        # possible other image classifiers. Current image classifier should not classify those objects.
        if add_impertinent:
            head, not_needed = os.path.split(data_dir)
            head, object_finder = os.path.split(head)
            dirs = os.listdir(head)
            for directory in dirs:
                if (directory != object_finder):
                    directory = os.path.join(head, directory)
                    train_ds, sub_categories = self.__add_impertinent_pictures(directory, train_ds)
                    self.__sub_categories = sub_categories
            num_classes = len(self.__sub_categories)
        
        # Scale and shuffle data.
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        normalization_layer = layers.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Create a new neuro network model.
        '''
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
        '''
        self.__model = Sequential([
            layers.Rescaling(1./255, input_shape=(OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH, 3)),

            layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
    
            layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
    
            layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
    
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_classes)
        ])
        
        # Train neuro network model.
        self.__model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.__model.summary()
        self.__model.fit(train_ds, validation_data=val_ds, epochs=OBJECT_EPOCHS)

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
            
            # We have a file path where to get one file. Not implemented.
            pass
        else:
            return None
    
    # Private class method to divide training data to training and testing parts
    def __divide_dataset(self, data_dir:pathlib.Path) -> tuple:
        
        # Code below defines class names according to the directory names.
        try:  
            train_ds = image_dataset_from_directory(    
            data_dir,
            validation_split=0.2,
            subset="training",
            labels="inferred",
            seed=123,
            color_mode="rgb",
            image_size=(OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH),
            batch_size=OBJECT_BATCH_SIZE)
        
            val_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            labels="inferred",
            seed=123,
            color_mode="rgb",
            image_size=(OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH),
            batch_size=OBJECT_BATCH_SIZE)
        except Exception as err:
            print(f"Unexpected error when reading train data from file: {err=}, {type(err)=}")    
            raise IOError(f"Failed to read train data.") 
        class_names = train_ds.class_names
        return train_ds, val_ds, class_names
    
    # Private class method to increase dataset class label number by one.
    def __increase_label_number(data:np.array, label:int) -> tuple:
        new_label = label + 1
        return data, new_label

    # Private class method to load all files below a root folder recursively into a dataset. Loaded files will be classified as 
    # unsorted. Extends the given dataset picture_list with the new pictures.
    def __add_impertinent_pictures(self, directory:pathlib, picture_list:tf.data.Dataset) -> tuple:
        new_picture_list = None
        class_names_list = None

        try:

            # Create filepath list, label list and class names list.
            string_glob = os.path.join(directory, "*.jpg")           
            filenames_as_tensor = tf.io.matching_files(string_glob)
            file_path_list = filenames_as_tensor.numpy().tolist()
            label_list = [0] * len(file_path_list)
            class_names_list = picture_list.class_names

            # Add impertinent class into picture_list dataset and class_names_list if it is not there.
            try:
                class_names_list.index(OBJECT_CLASSIFIER_IMPERTINENT_CLASS_NAME)
            except:    
                temp = [OBJECT_CLASSIFIER_IMPERTINENT_CLASS_NAME]
                if class_names_list != None:
                    temp.extend(class_names_list)
                class_names_list = temp
                picture_list.class_names = class_names_list
                picture_list = picture_list.map(self.__increase_label_number)

            # Create a new dataset with the new pictures.
            dataset_pictures = paths_and_labels_to_dataset(
                image_paths = file_path_list,
                image_size = [OBJECT_IMG_HEIGHT, OBJECT_IMG_WIDTH],
                num_channels = 3,
                labels = label_list,
                label_mode = "int",
                num_classes = len(class_names_list),
                interpolation = "bilinear",
                data_format = "channels_last",
                crop_to_aspect_ratio=False,
                pad_to_aspect_ratio=False,
                shuffle=False,
                shuffle_buffer_size=None,
                seed=None)           
            dataset_pictures.file_paths = file_path_list  
            dataset_pictures = dataset_pictures.batch(OBJECT_BATCH_SIZE)
            dataset_pictures = dataset_pictures.prefetch(tf.data.AUTOTUNE)
            dataset_pictures.class_names = class_names_list

            # Concatenate the new dataset with the old one.
            new_picture_list = dataset_pictures.concatenate(picture_list)
            new_picture_list.class_names = class_names_list

        except Exception as err:

            # If something failed, return the old dataset.
            if isinstance(picture_list, tf.data.Dataset):
                new_picture_list = picture_list
                new_picture_list.class_names = class_names_list
            else:
                raise ValueError("Invalid picture_list.")

        # Go through all subdirectories and recursively add pictures to the dataset.
        dirs = os.listdir(directory)
        for folder in dirs:
            folder = os.path.join(directory, folder)
            if os.path.isdir(folder):
                new_picture_list, class_names = self.__add_impertinent_pictures(folder, new_picture_list)
                new_picture_list.class_names = class_names

        return new_picture_list, new_picture_list.class_names
        
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
            raise ValueError("Cannot open or find file.")
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        probabilities = self.__model.predict(img_array)
        score = tf.nn.softmax(probabilities[0])
        if np.max(score) > OBJECT_THRESHOLD:
            sub_category = self.__sub_categories[np.argmax(score)]
            return Recognized_object(self.__object_finder_name, sub_category, float(np.max(score)))
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
            return Recognized_object(self.__object_finder_name, sub_category, float(np.max(score)))
        else:
            return None