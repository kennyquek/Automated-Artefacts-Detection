import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os


image_type = "angio"
PATH = "C:/Users/kenny/Desktop/Whole " + image_type
fold_num = "2"

H5_FILE_NAME = "pretrained_resnet50_model_stuctural.h5"

CHECKPOINT_PATH = "C:/Users/kenny/Desktop/checkpoint/whole model/" + H5_FILE_NAME
test_dir = "C:/Users/kenny/Desktop/dataset angio/test"
IMG_HEIGHT = 100
IMG_WIDTH = 100

num_defects_val = 42
num_normal_val = 42

model = tf.keras.models.load_model(CHECKPOINT_PATH)

# {'defects': 0, 'normal': 1}
# Predicting normal images

main_path = "C:/Users/kenny/Desktop/Whole " + image_type + "/validation_" + fold_num  + "/defects/"
files_name_array = os.listdir(main_path)
# predict model with validation set
points = 0
for defects_file_name in files_name_array:
    test_image = tf.keras.preprocessing.image.load_img(
        main_path + defects_file_name
        , target_size=(IMG_HEIGHT, IMG_WIDTH))


    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = tf.keras.applications.resnet50.preprocess_input(test_image)

    result = model.predict(test_image)

    class_index = (result[0][0])
    # print(defects_file_name + " = " + str(class_index))
    print('Predicted:', result)
    if(class_index < 0.5):
        points += 1
    else:
        print(defects_file_name)

print("Defect model scores " + str(points) + " /" + str(num_defects_val) + " " + str(points/num_defects_val * 100) + "%")


main_path = "C:/Users/kenny/Desktop/Whole " + image_type + "/validation_" + fold_num + "/normal/"

files_name_array = os.listdir(main_path)

# print(str(files_name_array))
# normal file index is 1
points = 0
for normal_file_name in files_name_array:
    test_image = tf.keras.preprocessing.image.load_img(
        main_path + normal_file_name
        , target_size=(IMG_HEIGHT, IMG_WIDTH))

    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = tf.keras.applications.resnet50.preprocess_input(test_image)

    result = model.predict(test_image)
    class_index = (result[0][0])

    if (class_index > 0.5):
        points += 1
    else:
        print(normal_file_name)

print("Normal model scores " + str(points) + " /" + str(num_normal_val) + " " + str(points/num_normal_val * 100) + "%")