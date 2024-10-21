# Picture_Classifier

Artificial Intelligence application to classify random pictures according to the objects they show. Uses neural network models recognize objects from a picture and then classifies the picture according to the objects it contains. Picture_Classifier is an application for sorting image files, which sorts images based on image content using neural network artificial intelligence. The application is written in the Python programming language using e.g. tensorflow, keras, numpy, scikit-learn and pandas libraries. The user interface of the application is command line.

## Principle

The basic idea is for the user of the application to create a folder with a mixed set of image files and another folder where the application creates subfolders, the application sorts and moves the image files from the original folder. In order for the application to be able to perform this sorting, it needs to know the sorting categories and learn from the model data what kind of image file is sorted into which category. You have to recognize different objects from the pictures, such as cars, faces, flowers, dogs, etc.

In order to teach the artificial intelligence to sort pictures, it has to be given model data to study and this data to be kept in the right way: we use the folder tree structure of the files to tell which are the categories to be sorted and what kind of objects should be found in this category. We use the following file tree structure from top to bottom:

- Root folder: this folder contains each of the material to be studied. This folder contains only Class Folders folders.

- Class folders: the folders below the root folder represent the categories to be sorted, and the folder names will be the names of the categories that Picture_Classifier uses to create the folders, moving the sorted image files into those folders. Class folders folder immediately contains only Object folders (main category) folders.

- Object folders (main category): these folders each contain their own general topic, for example cars, faces, flowers or whatever, which will be part of the classification criteria, but does not define any classification by itself. Object folder (main category) cannot contain picture files but only folders. Object folder (main category) folder immediately contains only the Object folders (subcategory) folders.

- Object folders (subcategory): these folders each contain their own specific topic, for example cars or family cars, which is the dominant part of the classification criteria, and this does define the classification by itself. Object folders (subcategory) folder immediately contains only image files.

The main category/subcategory division is making it easier to maintain the teaching data, it is not important for the recognition of objects in the Picture_Classifier application. Objects are always identified at the subcategory level.

## Architecture

Picture_Classifier contains several AI models: one for sorting and one for identifying each general object type, or main category. An identifying model classifies all subcategories included by main category.

- The object is identified from the image with the code of the objectfinder.py file. Object_finder finds an object if it fills the image, for example a face in a passport photo, and is trained to recognize faces.

- Identifying the object from the image file to be sorted is done by searching for it at different scales and in different locations. For this, the image is divided into several grids representing different scales, and a recognizable object is searched for in each grid separately. This is implemented with the code in the objectclassifier.py file. In this way, it is possible to obtain several findings in different scales, but only the findings of the scale containing the most findings are taken into account.

- When the objects in the image file have been identified and their numbers have been calculated, it is necessary to decide in which category the image file will be placed based on this information. This is done with the code in the pictureclassifier.py file.

In addition to the files mentioned above, Picture_Classifier contains two other source code files:

- Main.py contains the main() function of the application (even though there is no main function in Python) and a loop where the user of the application can tinker.

- The constants.py file contains all the constants defined in the application. The values ​​of these variables can only be changed inside the Constants.py file!

## Tuning

You can tune the application behaviour with following constants in constants.py file:

OBJECT_CLASSIFIER_FOLDER = the path to the folder where Picture_Classifier stores the serialized Object_finder models.
PICTURE_CLASSIFIER_NAME = filename to save the serialized Picture_classifier model.
OBJECT_BATCH_SIZE = the number of data rows of studying the Object_finder model in one cycle.
PICTURE_BATCH_SIZE = The number of data rows of studying the Picture_classifier model in one cycle.
OBJECT_IMG_HEIGHT = the height of the image in pixels to which the Object_finder model adjusts the recognizable image before recognition. 
OBJECT_IMG_WIDTH = the width of the image in pixels to which the Object_finder model adjusts the recognizable image before recognition.
OBJECT_TILE_OVERLAP_COEFFICIENT = an integer that tells how many overlapping identifications an image is recognized in one scale. 
OBJECT_MULTI_TILE_SIZE_MULTIPLIER = a real number that acts as a coefficient when moving from one scale to another. 
OBJECT_EPOCHS = an integer that tells how many cycles the Object_finder model studies before intermediate storage, on the basis of which the study is continued.  
OBJECT_THRESHOLD = a positive real number between [0, 1], indicating the probability at which the Object_finder model must at least identify an object in order to classify it as identified.
PICTURE_NUMBER_OF_HIDDEN_NEURONS = the number of neurons in the Picture_classifier model.
PICTURE_LEARNING_RATE = a positive (small) real number that adjusts the TensorFlow Keras optimizer's gradient calculation.
PICTURE_TRAINING_STEPS = an integer that tells how many cycles the Picture_classifier model learns the classification in total.
PICTURE_DISPLAY_STEP = an integer after how many cycles the Picture_classifier model reports its learning to the command line.

