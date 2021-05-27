import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os

num = "1"
# normal file index is 1
# defect file index is 0
for i in range(2):
    classes = i
    classes_names = ["defects", "normal"]

    checkpoint_path = "C:/Users/kenny/Desktop/checkpoint/whole model/merge_model_" + num + ".h5"
    angio = "C:/Users/kenny/Desktop/Whole angio/validation_" + num +"/" + classes_names[classes] + "/"
    structural = "C:/Users/kenny/Desktop/Whole angio/validation_" + num +" structural/" + classes_names[classes] + "/"
    IMG_HEIGHT = 100
    IMG_WIDTH = 100

    model = tf.keras.models.load_model(checkpoint_path)

    angio_names = os.listdir(angio)
    structural_names = os.listdir(structural)


    # print(angio_names)
    # print(structural_names)


    score = 0

    for i in range(angio_names.__len__()):
        # print(angio_names[i])
        # print(structural_names[i])
        angio_image = tf.keras.preprocessing.image.load_img(
            angio + angio_names[i]
            , target_size=(IMG_HEIGHT, IMG_WIDTH))
        angio_image = tf.keras.preprocessing.image.img_to_array(angio_image)
        angio_image = np.expand_dims(angio_image, axis=0)

        structural_image = tf.keras.preprocessing.image.load_img(
            structural + structural_names[i]
            , target_size=(IMG_HEIGHT, IMG_WIDTH))
        structural_image = tf.keras.preprocessing.image.img_to_array(structural_image)
        structural_image = np.expand_dims(structural_image, axis=0)

        result = model.predict([angio_image,structural_image])
        class_index = int(result[0][0])
        if class_index == classes:
            score += 1
        else:
            print(angio_names[i])
            print(structural_names[i])
        # print( "Prediction = " + str(class_index))


    print("total score " + str(score))