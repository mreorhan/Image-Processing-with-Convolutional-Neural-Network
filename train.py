#Created by @mreorhan
import numpy as np  # Numpy -> linear algebra
import matplotlib.pyplot as plt  # Pyplot -> Data visulation
import glob  # Glob -> For including images
import cv2  # OpenCV -> For filter in images
#from tensorflow import keras  # Tensorflow -> high-level api
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adadelta, Adamax
import os

directory_path = "./data/fruits-360/"
batch_size = 128
epochs = 4

# Import training dataset
training_fruit_img = []
training_label = []
for dir_path in glob.glob(directory_path + "Training/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_fruit_img.append(img)
        training_label.append(img_label)
training_fruit_img = np.array(training_fruit_img)
training_label = np.array(training_label)
print("Training set length: ", len(np.unique(training_label)))

# Import test dataset
test_fruit_img = []
test_label = []
for dir_path in glob.glob(directory_path + "Test/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_fruit_img.append(img)
        test_label.append(img_label)
test_fruit_img = np.array(test_fruit_img)
test_label = np.array(test_label)
print("Test set length: ", len(np.unique(test_label)))

# Import multiple-test dataset
test_fruits_img = []
tests_label = []
for img_path in glob.glob(os.path.join(directory_path + "test-multiple_fruits", "*.jpg")):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_fruits_img.append(img)
    tests_label.append(img_label)
test_fruits_img = np.array(test_fruits_img)
tests_label = np.array(tests_label)
print("Multiple Test set length: ", len(np.unique(tests_label)))

label_to_id = {v: k for k, v in enumerate(np.unique(training_label))}
label_to_id_test = {v: k for k, v in enumerate(np.unique(test_label))}
id_to_label = {v: k for k, v in label_to_id.items()}
id_to_label_test = {v: k for k, v in label_to_id_test.items()}
training_label_id = np.array([label_to_id[i] for i in training_label])
test_label_id = np.array([label_to_id_test[i] for i in test_label])
print("Training label id: ", training_label_id)
print("Training label id: ", test_label_id)

training_fruit_img, test_fruit_img = training_fruit_img / 255.0, test_fruit_img / 255.0

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(64, 64, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.01))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.01))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adadelta(), metrics=['accuracy'])
#tensorboard = keras.callbacks.TensorBoard(log_dir="./Graph", histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(training_fruit_img, training_label_id, validation_split=0.33, batch_size=batch_size, epochs=epochs)
history2 = model.fit(test_fruit_img, test_label_id, validation_split=0.33, batch_size=batch_size, epochs=epochs)

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history2.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history2.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

