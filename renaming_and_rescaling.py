"""
import os

print(os.getcwd())
print(os.getdirectory())

https://www.guru99.com/python-rename-file.html 

"""

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
"""

# I GIVE UPPPPPPPPPPPP :((((



import os
import shutil
from PIL import Image 


if __name__ == "__main__":
    
    route = os.getcwd() + "/Small_Rock_Collection/Fake_Rock/"
    
    print(route)
    print()
    print()
    # MAKE SURE TO REMOVE ANY HIDDEN FILES LIKE BELOW (.DS_Store) 
    # os.remove(route+".DS_Store")
    
    """
    # This was a test case to see if the code would work on just a single file
    
    file = route + "dwayne_the_three_dimensional_johnson.jpeg"

    if os.path.exists(file):

        # the following changes the file type 

        oldfilename=file.split(".")
        
        if oldfilename[1] == "jpg":
            True
        else:
            img = Image.open(file)
            target_name = oldfilename[0] + ".jpg"
            rgb_image = img.convert('RGB')
            rgb_image.save(target_name)
            print("Converted image saved as " + target_name) 
            
            os.remove(file)       
    else:
        print(file + " not found in given location")
    print()
    print()
    """
    

    i=0
    # iterate over files in
    # that directory
    for filename in os.listdir(route):
        i++
        
        f = os.path.join(route, filename)
        # checking if it is a file
        if os.path.isfile(f):

            # the following changes the file type 
            print(f)
            print(filename)
            print(type(filename))
            
            oldfilename=f.split(".")

            if oldfilename[1] == "jpg":
                True
            else:
                Newfilename=str(i)+".jpg"
                shutil.copyfile(filename, Newfilename)
                

                img = Image.open(f)
                target_name = oldfilename[0] + ".jpg"
                rgb_image = img.convert('RGB')
                rgb_image.save(target_name)
                print("Converted image saved as " + target_name)

                os.remove(f)


            

            # the following renames the images into a succinct naming scheme
      

            


    
    

    """ 
    old method - do not use

    #os.rename('guru99.txt','career.guru99.txt')

    # iterate over files in
    # that directory
    for filename in os.scandir(route):
        if filename.is_file():
            
            print("me and " + str(filename))
            print(filename.path)

            jpg_file = Image.open(filename.path).convert("RGB")
            jpg_file.save()
            os.rename(route + "testname.jpg","Fake_Dwayne_" + str(filename) + ".jpeg")  
    """