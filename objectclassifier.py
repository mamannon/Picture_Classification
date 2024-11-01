from genericpath import isdir
import PIL
import pickle
import os
from itertools import product
from PIL import Image

import constants
from objectfinder import Object_finder, Recognized_object

class Object_classifier:
    
    # Constructor deserializes image classifiers from files.
    def __init__(self) -> None:
        self.__object_finders = []
        self.__object_finder_names = []
        self.__deserialize_object_finders()
     
    # Before destructing Object_Recognizer instance we need to serialize all image classifiers to files.     
    def __del__(self) -> None:
        self.__serialize_object_finders()
        
    # Alla oleva on testausta varten!
    def serialize_object_finders(self) -> None:
        self.__serialize_object_finders()
    # Yllä oleva on testausta varten!
    
    # Private class method which serializes object finders into files.
    def __serialize_object_finders(self) -> None:
        try:
            for index in range(0, len(self.__object_finders), 1):
                filename = os.path.join(constants.OBJECT_CLASSIFIER_FOLDER, f"{self.__object_finder_names[index]}.pkl")
                if os.path.isfile(filename):
                     os.remove(filename)
                with open(filename, "wb+") as out_file:
                    pickle.dump(self.__object_finders[index], out_file)  
        except Exception as err:
            print(f"Unexpected error when serializing Object_finder: {err=}, {type(err)=}")
    
    # Private class method to load object finders from files.
    def __deserialize_object_finders(self) -> None:
        if not os.path.isdir(constants.OBJECT_CLASSIFIER_FOLDER):
            os.mkdir(constants.OBJECT_CLASSIFIER_FOLDER)
        items = os.listdir(constants.OBJECT_CLASSIFIER_FOLDER)
        for item in items:
            if item.endswith(".pkl"):
                object_finder_name = item.split(".")[0]
                item = os.path.join(constants.OBJECT_CLASSIFIER_FOLDER, item)
                try:
                    with open(item, "rb") as in_file:
                        self.__object_finders.append(pickle.load(in_file))
                    self.__object_finder_names.append(object_finder_name)
                except Exception as err:
                    print(f"Unexpected error when deserializing Object_finder: {err=}, {type(err)=}")    
                    raise IOError(f"Failed to deserialize image classifier {object_finder_name}.")               
    
    # Public class method to create a new object finder or teach old.
    def make_object_finder(self, path:str, destroy_previous=False) -> bool:
        try:
            object_finder_name = os.path.basename(path)
            object_finder_name = object_finder_name.split(".")[0]
        except:
            return False
        if not destroy_previous:
            try:
                index = self.__object_finder_names.index(object_finder_name)
                object_finder = self.__object_finders[index]
                object_finder.add_more_training(path)
                return True
            except:
                temp = Object_finder(path)
                self.__object_finder_names.append(temp.get_name())
                self.__object_finders.append(temp)
                return True
        else:
            try:
                index = self.__object_finder_names.index(object_finder_name)
                self.__object_finder_names.pop(index)
                self.__object_finders.pop(index)
            except:
                pass
            temp = Object_finder(path)
            self.__object_finder_names.append(temp.get_name())
            self.__object_finders.append(temp)        
            return True
        
    # Public class method to recognize objects from an image. Use file- or url- string path for image parameter. 
    def recognize_objects_str(self, image:str) -> tuple:
        objects = []
        index = 0
        for classifier in self.__object_finders:
            temp = classifier.do_classification_str(image)
            if temp is not None:
                objects.append(temp)
                index += 1
        return objects, index
    
    # Public class method to recognize objects from an image. Use Image type bitmap for image parameter.
    def recognize_objects_image(self, image:Image) -> tuple:
        objects = []
        index = 0
        for classifier in self.__object_finders:
            temp = classifier.do_classification_image(image)
            if temp is not None:
                objects.append(temp)
                index += 1
        return objects, index
    
    # Public class method to tile image into several images. Use file- or url- string path for image parameter.
    # This method produces OBJECT_TILE_OVERLAP_COEFFICIENT times the same image set overlapping each other.
    def tile_image_str(self, image:str, tile_size:int) -> list:
        images = []
        try:
            img = Image.open(image) 
        except:
            raise ValueError("Cannot open or find file.")
        w, h = img.size
        grid = product(range(0, h-h%tile_size+tile_size, tile_size), range(0, w-w%tile_size+tile_size, tile_size))
        for i, j in grid:
            kk = int(constants.OBJECT_TILE_OVERLAP_COEFFICIENT)
            box = (j, i, j+tile_size, i+tile_size)
            images.append(img.crop(box))
            for k in range(1, kk, 1):
                box = (j+tile_size*k/kk, i+tile_size*k/kk, j+tile_size*(1 + k/kk), i+tile_size*(1 + k/kk))
                images.append(img.crop(box))
        return images
    
    # Public class method to tile image into several images. Use Image type bitmap for image parameter.
    # This method produces OBJECT_TILE_OVERLAP_COEFFICIENT times the same image set overlapping each other.
    def tile_image_image(self, image:Image, tile_size:int) -> list:
        images = []
        w, h = image.size
        grid = product(range(0, h-h%tile_size+tile_size, tile_size), range(0, w-w%tile_size+tile_size, tile_size))
        for i, j in grid:
            kk = int(constants.OBJECT_TILE_OVERLAP_COEFFICIENT)
            box = (j, i, j+tile_size, i+tile_size)
            images.append(image.crop(box))
            for k in range(1, kk, 1):
                box = (j+tile_size*k/kk, i+tile_size*k/kk, j+tile_size*(1 + k/kk), i+tile_size*(1 + k/kk))
                images.append(image.crop(box))
        return images
    
    # Public class method to do multi-tile scan for an image. The size of an object matters: if you have several
    # flowers in a picture, for example, but we are looking for a picture size flower, we need to crop a right size
    # picture from original picture to find the flower.
    def multi_tile_recognize_objects(self, image:PIL) -> list:
        objects = []
        try:
            img = Image.open(image) 
        except:
            raise ValueError("Cannot open or find file.")
        w, h = img.size
        tile_size = 0
        if w > h:
            tile_size = w
        else:
            tile_size = h
            
        # This is the whole image, we will always take all objects form here.
        objects.extend(self.recognize_objects_image(img)[0])
        
        # From tiles we take only those objects, which belong to the most successfull tile set.
        tile_size = int((tile_size - tile_size % constants.OBJECT_MULTI_TILE_SIZE_MULTIPLIER) / constants.OBJECT_MULTI_TILE_SIZE_MULTIPLIER)
        besthits = []
        bestcount = 0
        while tile_size > 100:
            tiles = self.tile_image_image(img, tile_size)
            hits = []
            count = 0
            for tile in tiles:
                h, c = self.recognize_objects_image(tile)
                hits.extend(h)
                count += c
            if bestcount<count:
                bestcount = count
                besthits = hits
            tile_size = int((tile_size - tile_size % constants.OBJECT_MULTI_TILE_SIZE_MULTIPLIER) / constants.OBJECT_MULTI_TILE_SIZE_MULTIPLIER)
        objects.extend(besthits)
        return objects
            
    # Public class method to get all category names of object finders.
    def get_categories(self) -> list:
        categories = []
        for finder in self.__object_finders:
            main_category = finder.get_name()
            sub_categories = finder.get_sub_categories()
            for sub in sub_categories:
                categories.append(f"{main_category}:{sub}")
        return categories
            
            
            
            
            
        
        
        