"""
Used to generate train.txt and test.txt files for YOLO model training
"""
import glob, os

dataset_path = 'obj'

# Percentage of images to be used for the test set
percentage_test = 20

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')  
file_test = open('test.txt', 'w')

# Populate train.txt and test.txt
counter = 1  
index_test = round(100 / percentage_test)  
for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.png")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test+1:
        counter = 1
        file_test.write('/content/obj' + "/" + title + '.png' + "\n")
    else:
        file_train.write('/content/obj' + "/" + title + '.png' + "\n")
        counter = counter + 1