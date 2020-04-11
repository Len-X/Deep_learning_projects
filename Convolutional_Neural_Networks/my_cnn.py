# Convolutional Neural Network

## PART 1 - Building the CNN

# Importing Packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the CNN

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation="relu")) # also try (128, 128)
# changed from (64, 64, 3)

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer for better accuracy and to reduce overfitting
classifier.add(Conv2D(32, (3, 3), activation="relu")) # from step 1
classifier.add(MaxPooling2D(pool_size=(2, 2))) # from step 2

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation="relu"))  # hidden layer (was originally units=128, changed to 64)
# 128 is a number of hidden nodes (nodes in the hidden layer). We are guestimating that number. No rule of thumb.
classifier.add(Dropout(0.5)) # added dropout to prevent overfitting
classifier.add(Dense(units=1, activation="sigmoid"))  # output layer
# "sigmoid" function because of the binary outcome: cat or dog
# units = 1 - because it'll be only one outcome cat or dog, therefore only 1 output layer


# Compiling the CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

## PART 2 - Fitting the CNN to the images

# Importing from keras library
from keras.preprocessing.image import ImageDataGenerator

# Copied from keras documentation - image preprocessing
# Preprocessing of the images of the train set
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Image preprocessing of the test set
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Apply image argumentation to the images of the training set. And resizing all images to the 64x64 dimensions.
# Creating batches of 32 images each. (creation of the training set)
training_set = train_datagen.flow_from_directory('DL/Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size=(128, 128), #(was originally target_size=(64, 64),
                                                 # changed to 128)
                                                 batch_size=32,
                                                 class_mode='binary')

# Same for test set. (creation of the test set)
test_set = test_datagen.flow_from_directory('DL/Convolutional_Neural_Networks/dataset/test_set',
                                            target_size=(128, 128), #(was originally target_size=(64, 64),
                                            # changed to 128)
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/32, # added/32 (batch size)
                         epochs=50, # changed to 50 (was originally 25)
                         validation_data=test_set,
                         validation_steps=2000/32) # added/32 (batch size)
# 8000 images in training set and 2000 images in test set


# For faster processing use:
#classifier.fit_generator(training_set,
                         #epochs=25,
                         #validation_data=test_set)

## PART 3 - Making new predictions (single prediction)

import numpy as np
from keras.preprocessing import image

test_image = image.load_img("DL/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_2.jpg",
                            target_size=(128, 128)) # was (64, 64)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0) # first dimension will have a first index: 0
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"
print(prediction)