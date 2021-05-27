import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


PATH = "C:/Users/kenny/Desktop/dataset angio"

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_defects_dir = os.path.join(train_dir, 'defects')  # directory with our training cat pictures
train_normal_dir = os.path.join(train_dir, 'normal')

validation_defects_dir = os.path.join(validation_dir, 'defects')
validation_normal_dir = os.path.join(validation_dir, 'normal')


num_defects_tr = len(os.listdir(train_defects_dir))
num_normal_tr = len(os.listdir(train_normal_dir))

num_defects_val = len(os.listdir(validation_defects_dir))
num_normal_val = len(os.listdir(validation_normal_dir))


total_train = num_defects_tr
total_val = num_defects_val

print('total training defects images:', num_defects_tr)
print('total training normal images:', num_normal_tr)

print('total validation defects images:', num_defects_val)
print('total validation normal images:', num_normal_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)



batch_size = 8
epochs = 20
IMG_HEIGHT = 100
IMG_WIDTH = 100

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

# plotImages(sample_training_images[:5])

# CNN sequential network
# model = Sequential([
#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
#     MaxPooling2D(),
#     Conv2D(32, 3, padding='same', activation='relu',),
#     MaxPooling2D(),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# checkpoint_path = "C:/Users/kenny/Desktop/checkpoint/training_1/cp.ckpt"
# checkpoint_dir = "C:/Users/kenny/Desktop/checkpoint/training_1/cp.ckpt" #os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# Train the model with the new callback
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps= batch_size
)

# save whole model
model.save('C:/Users/kenny/Desktop/checkpoint/whole model/my_model.h5')

# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# test_dir = "C:/Users/kenny/Desktop/dataset angio/test"
#
# test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
# test_data_gen = test_image_generator.flow_from_directory(
#                                                            directory=test_dir,
#                                                            shuffle=True,
#                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                            class_mode='binary')
#
# test_loss, test_acc = model.evaluate(test_data_gen, verbose=2)
# print(test_acc)

print(train_data_gen.class_indices)

# Predicting normal images
test_image = tf.keras.preprocessing.image.load_img("C:/Users/kenny/Desktop/dataset angio/test/normal/SNASERI0302_SNASERI0302__SNASERI0302_19500601_Male_Angio (12mmx12mm)_20190523143719_OS_20190819170335_Angiography_Superficial.bmp_grid6.bmp"
                                      ,target_size=(IMG_HEIGHT, IMG_WIDTH))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)

# Predicting defect images
test_image = tf.keras.preprocessing.image.load_img("C:/Users/kenny/Desktop/dataset angio/test/defects/SNASERI0272_SNASERI0272__SNASERI0272_19540501_Male_Angio (12mmx12mm)_20190408154135_OS_20190820183112_Angiography_Superficial.bmp_grid22.bmp"
                                      ,target_size=(IMG_HEIGHT, IMG_WIDTH))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)