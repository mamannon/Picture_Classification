from pictureclassifier import Picture_classifier
from objectclassifier import Object_classifier

def main():
    pic_cla = None
    keep = True
    while keep:
        print("\nGive 'T' if you want to train model, \n'S' if you want to sort pictures or \n'Q' if you want to quit:\n")
        char = input()
        if char == "T":
            print("Write the path of the root directory, where the training material is.\nYou need to have a path to following directory structure:\n                    path\n                folders for picture classification I.E. the way you want to classify pictures: folder names are classes\n            folders for object classification I.E. Object_finders: folder names are main_categories\n         folders for further object classification inside an Object_finder: folder names are sub_categories\n    picture files for model training\n")    
            path = input()
            print("To make picture classification work better, you may want object detectors to be avare of other object detectors. \nThis way multiple object detectors doesn't classify same objects, however training takes a lot more time. \nDo you want detectors be aware of others? If yes, give 'Y', else 'N'.\n")
            impertinent = input()
            if impertinent == "Y":
                impertinent = True
            else:
                impertinent = False
            print("Do you want to save old trained object detectors and not recreate them? \nIf yes, give 'Y', else 'N' and if you want to quit, give 'Q'.\n")
            char = input()
            if char == "Y":
                if pic_cla == None:
                    pic_cla = Picture_classifier.deserialize()  
                    if pic_cla == None:
                        print("Cannot deserialize model. Creating completely new model.\n")   
                        pic_cla = Picture_classifier(path, impertinent)
                else:
                    pic_cla.train_picture_classifier_model(path, False, impertinent)   
            elif char == "N":
                if pic_cla == None:    
                    pic_cla = Picture_classifier(path, impertinent)  
                else:    
                    pic_cla.train_picture_classifier_model(path, True, impertinent)                     
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