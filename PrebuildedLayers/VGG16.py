import os
os.environ['PYTHONHASHSEED'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
from tfdeterminism import patch
patch()

import os
import numpy as np

import random
import time

# How to get stable results with TensorFlow, setting random seed 1579080527
seed = 0 # int(time.time())
tf.compat.v1.reset_default_graph()
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

from tensorflow.keras import applications, Model, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

import matplotlib.pyplot as plt

IMG_HEIGHT = 100
IMG_WIDTH = 100
batch_size = 8
epochs = 20

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

image_type = "angio"
PATH = "C:/Users/kenny/Desktop/Whole " + image_type
fold_num = "1"
train_dir = os.path.join(PATH, 'train_' + fold_num)
validation_dir = os.path.join(PATH, 'validation_' + fold_num)

train_defects_dir = os.path.join(train_dir, 'defects')
train_normal_dir = os.path.join(train_dir, 'normal')

validation_defects_dir = os.path.join(validation_dir, 'defects')
validation_normal_dir = os.path.join(validation_dir, 'normal')

num_defects_tr = len(os.listdir(train_defects_dir))
num_normal_tr = len(os.listdir(train_normal_dir))

num_defects_val = len(os.listdir(validation_defects_dir))
num_normal_val = len(os.listdir(validation_normal_dir))


total_train = num_defects_tr + num_normal_tr
total_val = num_defects_val + num_normal_val

print("Total number of training sets " + str(total_train))
print("Total number of validation sets " + str(total_val))

train_image_generator = ImageDataGenerator(preprocessing_function=applications.resnet.preprocess_input)  # Generator for our training data
validation_image_generator = ImageDataGenerator(preprocessing_function=applications.resnet.preprocess_input)  # Generator for our validation data

# Did some data argumentation as dataset is abit little

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

# This is a pre-trained model (Also known as transfer learning with weights is "imagenet"?)

base_model = applications.resnet.ResNet101(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
print("Number of layers in the base model: ", len(base_model.layers))
base_model.trainable = True
# #  19 layers in total
# Fine tune from this layer onwards
fine_tune_at = 300
# 250

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

x = base_model.output
# Maxpooling is to reduce dimension so that u can get a more obvious feature extraction
x = layers.MaxPooling2D()(x)
# Prevent over fitting
x = layers.Dropout(0.2, seed = seed)(x)
x = layers.Flatten()(x)  # Becomes a vector
# Might need to add another dense layer (try 4026)
x = layers.Dense(1024, activation='relu')(x) # There is no specific manner of selecting the number of units in a dense layer. You need to try different units and see which gives the best accuracy and the lowest loss value
# To be able to increase the learning rate so as to save more time
x = layers.BatchNormalization()(x)
predictions = layers.Dense(1, activation='sigmoid')(x)  # sigmoid is used for the two-class logistic regression

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# change adam to sgd to increase learning rate due to over fitting SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True),
model.compile(optimizer= SGD(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_data_gen,
          epochs=epochs, steps_per_epoch=total_train//batch_size,
          validation_data=val_data_gen,
          validation_steps= total_val//batch_size)

# save whole model
# model.save('C:/Users/kenny/Desktop/checkpoint/whole model/pretrained_resnet50_model_stuctural.h5')

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
    test_image = tf.keras.applications.resnet.preprocess_input(test_image)
    result = model.predict(test_image)
    class_index = (result[0][0])
    # print(defects_file_name + " = " + str(class_index))
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
    test_image = tf.keras.applications.resnet.preprocess_input(test_image)
    result = model.predict(test_image)
    class_index = (result[0][0])

    # print(normal_file_name + " = " + str(class_index))
    if (class_index > 0.5 ):
        points += 1
    else:
        print(normal_file_name)

print("Normal model scores " + str(points) + " /" + str(num_normal_val) + " " + str(points/num_normal_val * 100) + "%")

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()