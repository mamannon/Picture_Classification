from pictureclassifier import Picture_classifier
from objectclassifier import Object_classifier

def main():
    '''
    objects = []
    obj_cla = Object_classifier()
    obj_cla.make_object_finder("https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", False)
    objects = obj_cla.multi_tile_recognize_objects("voikukkia.jpg") 
    print(f"Objekteja löytyi {len(objects)} kappaletta:")
    for recognized_object in objects:
        print(recognized_object.main_category)
        print(recognized_object.sub_category)
        print(recognized_object.probability)
        print("")
    '''
    pic_cla = None
    keep = True
    while keep:
        print("\nGive 'T' if you want to train model, \n'S' if you want to sort pictures or \n'Q' if you want to quit:\n")
        char = input()
        if char == "T":
            print("Write the path of the root directory, where the training material is.\nYou need to have a path to following directory structure:\n                    path\n                folders for picture classification I.E. the way you want to classify pictures: folder names are classes\n            folders for object classification I.E. Object_finders: folder names are main_categories\n         folders for further object classification inside an Object_finder: folder names are sub_categories\n    picture files for model training\n")    
            path = input()
            print("Do you want to save old training? If yes, give 'Y', else 'N' and if you want to quit, give 'Q'.\n")
            char = input()
            if char == "Y":
                if pic_cla == None:
                    pic_cla = Picture_classifier.deserialize()  
                    if pic_cla == None:
                        print("Cannot deserialize model. Creating completely new model.\n")   
                        pic_cla = Picture_classifier(path, True)
                else:
                    pic_cla.train_picture_classifier(path, False)   
            elif char == "N":
                if pic_cla == None:    
                    pic_cla = Picture_classifier(path, True)  
                else:    
                    pic_cla.train_picture_classifier(path, True)                     
            else:
                break    
        elif (char == "S"):
            print("Write the path of the root directory, where are pictures to be sorted.\n")
            input_path = input()
            print("Write the path of the root directory, where sorted pictures will be moved.\n")
            output_path = input() 
            if pic_cla == None:
                pic_cla = Picture_classifier.deserialize()  
            if pic_cla == None:
                print("Cannot deserialize model. You need to train new model.\n") 
                break
            pic_cla.classify_pictures(input_path, output_path)           
        elif (char == "Q"):
            break    
            
if __name__ == "__main__":
    main()