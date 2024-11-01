import pickle
import os
import numpy as np
import pandas as pa
import category_encoders as ce
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from genericpath import isfile

import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import random_normal
from tensorflow.python.ops.variables import trainable_variables


import constants
#from mybaseoptimizer import My_base_optimizer
#from mysgd import My_SGD
from objectclassifier import Object_classifier
from objectfinder import Recognized_object

class Picture_classifier:
    
    # Constructor creates a new Picture-classifier class instance. Use static deserialize method to 
    # deseralize Picture-classifier class instance from file instead of creating a new Picture_classifier.
    def __init__(self, path:str, destroy_previous=False) -> None:
        self.__encoder = ce.BinaryEncoder(cols="classification")
        self.__optimizer = tf.keras.optimizers.SGD(learning_rate=constants.PICTURE_LEARNING_RATE)
        self.__classes = self.__get_classes(path)
        self.__object_classifier = Object_classifier()
        self.__weights = None
        self.__biases = None
        self.__model = self.__create_picture_classifier_model(path, destroy_previous)  
        self.__features = self.__object_classifier.get_categories()
      
    # Before destructing this Picture_classifier class instance, we need to serialize it.
    def __del__(self) -> None:
        self.__serialize_picture_classifier()
        
    # Private class method which serializes this class instance.
    def __serialize_picture_classifier(self) -> None:
        try:
            if os.path.isfile(constants.PICTURE_CLASSIFIER_NAME):
                os.remove(constants.PICTURE_CLASSIFIER_NAME)
            with open(constants.PICTURE_CLASSIFIER_NAME, "wb+") as out_file:
                pickle.dump(self, out_file)
        except Exception as err:
            print(f"Unexpected error when serializing Picture_classifier: {err=}, {type(err)=}")
    
    # Private class method to find out classes into which to classify pictures.
    # Class names are the top folder names.
    def __get_classes(self, path:str) -> list:
        classes = []
        items = os.listdir(path)
        for item in items:
            if os.path.isdir(os.path.join(path, item)):    
                classes.append(item)
        return classes

    # Private class method to create a trained neural network model to classify pictures. You need to have a
    # path to following directory structure:
    #                   path
    #               folders for picture classification I.E. the way you want to classify pictures: folder names are classes
    #           folders for object classification I.E. Object_finders: folder names are main_categories
    #       folders for further object classification inside an Object_finder: folder names are sub_categories
    #   picture files for model training
    def __create_picture_classifier_model(self, path:str, destroy_previous=False) -> None:       
        
        # First make all object finders or update them. This makes object classifier trained.
        for classpath1 in self.__classes:
            classpath1 = os.path.join(path, classpath1)
            items1 =  os.listdir(classpath1)
            for item1 in items1:
                classpath2 = os.path.join(classpath1, item1)
                if os.path.isdir(classpath2): 
                    self.__object_classifier.make_object_finder(classpath2, destroy_previous)    
                    
        # KORJAA Alla oleva on testausta varten!
        self.__object_classifier.serialize_object_finders()
        # Yllä oleva on testausta varten!
                    
        # Second gather data to train picture classifier: predict a classification for every picture in picture
        # files for model training and make a dataset, which has a column for each feature (sub_category) and 
        # also one column for the class (classification).
        categories = self.__object_classifier.get_categories()
        data = dict(zip(categories, [None]*len(categories)))
        data["classification"] = None
        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                recognized_objects = self.__object_classifier.multi_tile_recognize_objects(filepath)
                row_of_features = self.__parse_recognized_objects_dict(recognized_objects)
                for feature in row_of_features:
                    value = [row_of_features[feature]]
                    old_values = data[feature]
                    if old_values == None:
                        data[feature] = value
                    else:
                        old_values.extend(value)
                temp = root.replace(path, "").split("\\")   
                last_index = len(temp) -1
                value = f"{temp[last_index-2]}"
                old_values = data["classification"]
                if old_values == None:                  
                    data["classification"] = value
                elif type(old_values) == str:
                    data["classification"] = [old_values, value]
                else:
                    old_values.append(value)
        dataset = pa.DataFrame(data)
        
        # KORJAA Alla oleva on testausta varten!
        
        try:
            with open("testidataset.pkl", "wb+") as out_file:
                pickle.dump(dataset, out_file)
        except Exception as err:
            print(f"Unexpected error when serializing testidataset: {err=}, {type(err)=}")      
        
        '''
        dataset = None
        try:
            with open("testidataset.pkl", "rb") as in_file:
                dataset = pickle.load(in_file)
        except Exception as err:
            print(f"Unexpected error when deserializing testidataset: {err=}, {type(err)=}")   
        '''
        #Yllä oleva on testausta varten!
        
        # The dependent variable y is not numerical value in a dataset but a class name. We use binaryencoder to 
        # change it numeric. In order to get back the class name, first need to save the class names in a list
        # and replace class name by index number in a dataset.
        shape = dataset.shape
        ind = dataset.columns.get_loc("classification")
        for i in range(0, shape[0]-1, 1):
            j = self.__classes.index(dataset["classification"][i])
            #dataset.iloc[ind][i] = j   #Don't use this, it may not work here.
            dataset.iloc[i, ind] = j
        dataset = self.__encoder.fit_transform(dataset)
        num_binary_columns = dataset.shape[1] - shape[1] + 1
        
        # Divide dataset into train and test parts and separate feature matrix X from dependent variable y.
        classification = []
        for column in dataset.columns:
            if column.find("classification") != -1:
                classification.append(column)
        X = dataset.drop(classification, axis=1)
        y = dataset[classification]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=125)
        
        # Need to scale all values in all columns in X_train. Let's use standardscaler.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # X_test needs a little different treatment.
        X_test = scaler.transform(X_test)

        # Setup training parameters. 
        num_features = len(self.__object_classifier.get_categories())
        learning_rate = constants.PICTURE_LEARNING_RATE
        training_steps = constants.PICTURE_TRAINING_STEPS
        batch_size = constants.PICTURE_BATCH_SIZE
        display_step = constants.PICTURE_DISPLAY_STEP
        n_hidden = constants.PICTURE_NUMBER_OF_HIDDEN_NEURONS
        
        # Use tf.data API to shuffle and batch data.
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_data = train_data.repeat().shuffle(60000).batch(batch_size).prefetch(1)        
        
        # Store layers weight and bias: use random value generator to initialize weights.
        random_normal = tf.random_normal_initializer()
        self.__weights = {
            "h": tf.Variable(tf.cast(tf.Variable(random_normal([num_features, n_hidden])), dtype="float64"), trainable=True), 
            "out": tf.Variable(tf.cast(tf.Variable(random_normal([n_hidden, num_binary_columns])), dtype="float64"), trainable=True)
        }
        self.__biases = {
            "b": tf.Variable(tf.cast(tf.Variable(tf.zeros([n_hidden])), dtype="float64"), trainable=True),
            "out": tf.Variable(tf.cast(tf.Variable(tf.zeros([num_binary_columns])), dtype="float64"), trainable=True)
        }
        
        # Run training for the given number of steps.
        for index, (batch_x, batch_y) in enumerate(train_data.take(training_steps), start=1):
            self.__optimization(batch_x, batch_y)
            
            # Every DISPLAY_STEP times we print the situation into the terminal.
            if index % display_step == 0:
                pred = self.__neural_net(batch_x)
                loss = self.__cross_entropy(pred, batch_y)
                acc = self.__accuracy(pred, batch_y)
                print(f"Picture classifier training epoch: {index}, loss: {loss}, accuracy: {acc}")
                
        # Finally test model on validation set.
        pred = self.__neural_net(X_test)
        print(f"Test accuracy of Picture Classifier: {self.__accuracy(pred, y_test)}")
                    
    # This private class method is the core of the picture classifier.
    def __neural_net(self, input_data:tf.Tensor) -> tf.Tensor:      
        hidden_layer = tf.add(tf.matmul(input_data, self.__weights["h"]), self.__biases["b"])
        hidden_layer = tf.nn.sigmoid(hidden_layer)
        out_layer = tf.add(tf.matmul(hidden_layer, self.__weights["out"]), self.__biases["out"])
        out_layer = tf.nn.softmax(out_layer)
        return tf.where(out_layer < 0.5, 0.0, tf.where(out_layer >= 0.5, 1.0, out_layer))
    
    # Private class method to compare predicted values to the real ones.
    def __cross_entropy(self, y_pred:tf.Tensor, y_true:tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, dtype="float64")    
        y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-9, 1.0), dtype="float64")    
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), [0, 1]))
    
    # Private class method to adjust the weights and biases between epochs.
    def __optimization(self, X:tf.Tensor, y:tf.Tensor) -> None:
        with tf.GradientTape() as tape:
            tape.watch(self.__weights.values())
            tape.watch(self.__biases.values())
            pred = self.__neural_net(X)
            loss = self.__cross_entropy(pred, y)
            trainable_variables = list(self.__weights.values()) + list(self.__biases.values())
        gradients = tape.gradient(loss, trainable_variables)
        self.__optimizer.apply_gradients(zip(gradients, trainable_variables))
            
    # Private class method to get a tensor of which numbers describe the accuracy of the picture classifier.
    def __accuracy(self, y_pred:tf.Tensor, y_true:tf.Tensor) -> tf.Tensor:
        correct_prediction = tf.equal(tf.cast(y_pred, tf.int64), y_true)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=0)
                    
    # Private class method to parse one row of features as a dictionary.
    def __parse_recognized_objects_dict(self, recognized_objects:list) -> dict:
        row_of_features = dict()
        feature_titles = self.__object_classifier.get_categories()
        for item1 in feature_titles:
            row_of_features[item1] = 0
            for item2 in recognized_objects:
                feature = f"{item2.main_category}:{item2.sub_category}"
                if item1 == feature:
                    probability = row_of_features[item1]
                    row_of_features[item1] = item2.probability + probability
        return row_of_features
    
    # Private class method to parse one row of features as a tensor.
    def __parse_recognized_objects_tensor(self, recognized_objects:list) -> tf.Tensor:
        row_of_features = []
        feature_titles = self.__object_classifier.get_categories()
        for index1 in range(0, len(feature_titles), 1):
            row_of_features.append(0)
            item1 = feature_titles[index1]
            for item2 in recognized_objects:
                feature = f"{item2.main_category}:{item2.sub_category}"
                if item1 == feature:
                    probability = row_of_features[index1]
                    row_of_features[index1] = item2.probability + probability
        feature_tensor = tf.Variable(initial_value=[row_of_features], shape=[1, len(self.__features)])
        return feature_tensor

    # Private class method to decode binary encoded data to int.
    def __binary_decode(self, bit_array:list) -> int:
        integer_value = 0    
        for bit in bit_array:
            bit = int(bit)    
            integer_value = (integer_value << 1) | bit    
        return integer_value
       
    # Public static class method which deserializes this class instance from file.
    @staticmethod
    def deserialize() -> object:
        try:
            with open(constants.PICTURE_CLASSIFIER_NAME, "rb") as in_file:
                return pickle.load(in_file)
        except:
            return None

    # Public class method to move pictures located in input_path folder as classified folders in output_path folder.
    def classify_pictures(self, input_path:str, output_path:str) -> None:
        for picture in os.listdir(input_path):
            picture = os.path.join(input_path, picture)
            if os.path.isfile(picture):
                recognized_objects = self.__object_classifier.multi_tile_recognize_objects(os.path.join(input_path, picture))
                row_of_features = self.__parse_recognized_objects_dict(recognized_objects)
                feature_tensor = tf.cast(self.__parse_recognized_objects_tensor(recognized_objects), dtype="float64")
                pred = self.__neural_net(feature_tensor)
                
                # pred is in a binary-encoded format, need to decode it to get the index of the class name.
                bit_list = pred.numpy()[0].tolist()
                decoded = self.__binary_decode(bit_list)
                if (decoded < len(self.__classes)):
                    class_name = self.__classes[decoded]
                else:
                    
                    # If index is too big, put the picture to the "Unsorted pictures" folder.
                    class_name = "Unsorted pictures"
                
                # Create a folder name <class_name>, if it doesn't exist.
                directory = os.path.join(output_path, class_name)
                if not os.path.isdir(directory):
                    os.mkdir(directory)
                    
                # Move picture from input_path to directory.
                shutil.move(os.path.join(input_path, picture), directory)    
                
    # Public class method to train model.
    def train_picture_classifier(self, path:str, destroy_previous) -> None:
        self.__create_picture_classifier_model(path, destroy_previous)
    

