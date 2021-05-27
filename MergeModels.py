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

IMG_HEIGHT = 100
IMG_WIDTH = 100
batch_size = 6
epochs = 28
# 30
def generate_generator_multiple(genX1, genX2):
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

PATH = "C:/Users/kenny/Desktop/Whole angio"
fold_num = "1"
train_dir = os.path.join(PATH, 'train_' + fold_num)
validation_dir = os.path.join(PATH, 'validation_' + fold_num)

train_defects_dir = os.path.join(train_dir, 'defects')  # directory with our training cat pictures
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

train_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

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

PATH = "C:/Users/kenny/Desktop/Whole structural"

train_dir = os.path.join(PATH, 'train_' + fold_num)
validation_dir = os.path.join(PATH, 'validation_' + fold_num)

train_defects_dir = os.path.join(train_dir, 'defects')  # directory with our training cat pictures
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

train_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

# Did some data argumentation as dataset is abit little

train_data_gen_2 = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen_2 = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

train_generator=generate_generator_multiple(train_data_gen, train_data_gen_2)

validation_generator=generate_generator_multiple(val_data_gen, val_data_gen_2)



# Angio model
angio_base_model = applications.vgg19.VGG19(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

angio_base_model.trainable = True
#  19 layers in total
# Fine tune from this layer onwards
fine_tune_at = 12

# Freeze all the layers before the `fine_tune_at` layer
for layer in angio_base_model.layers[:fine_tune_at]:
  layer.trainable =  False

x = angio_base_model.output
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.2, seed = seed)(x)
x = layers.Flatten()(x)

structural_base_model = applications.vgg19.VGG19(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

for layer in structural_base_model.layers:
    layer._name = layer._name + str("_2")

structural_base_model.trainable = True
#  19 layers in total
# Fine tune from this layer onwards
fine_tune_at = 12

# Freeze all the layers before the `fine_tune_at` layer
for layer in structural_base_model.layers[:fine_tune_at]:
  layer.trainable =  False

y = structural_base_model.output
y = layers.GlobalAveragePooling2D()(y)
y = layers.Dropout(0.2, seed = seed)(y)
y = layers.Flatten()(y)

concat  = tf.keras.layers.Concatenate()([x, y])

y = layers.Dense(1024, activation='relu')(concat)
y = layers.BatchNormalization()(y)

predictions = layers.Dense(1, activation='sigmoid')(y)

model = Model(inputs=[angio_base_model.input, structural_base_model.input], outputs=predictions)

model.summary()

model.compile(optimizer= Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_generator,
          epochs=epochs, steps_per_epoch=336//batch_size,
          validation_data=validation_generator,
          validation_steps= 84//batch_size)


model.save("C:/Users/kenny/Desktop/checkpoint/whole model/merge_model_" + fold_num.__str__() +".h5")
