import csv
import numpy as np
import cv2
import time
from sklearn.model_selection import train_test_split

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.optimizers import adam
from keras.utils.data_utils import Sequence
from keras.utils import plot_model, to_categorical
from keras import backend as K

def save_keras_model(save_model, path):
    """Saves keras model to given path."""
    save_model.save_weights(path + 'model.h5')

    with open(path + 'model.json', "w") as text_file:
        text_file.write(save_model.to_json())

def load_content(content, verbose=1):
    images = list()
    labels = list()
    paths = list()
    for path, label in content:
        paths.append(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        if label == "green":
            labels.append(2)
        elif label == "red":
            labels.append(0)
        elif label == "yellow":
            labels.append(1)
        else:
            labels.append(4)
            print("[ERROR] Label of image ", path, " neither red, yellow, or green!")
    images = np.asarray(images)
    labels = np.asarray(labels)
    if verbose ==1:
        return images, labels
    elif verbose ==2:
        return images, labels, paths

# ================================================================================
# standard CNN model
def build_model():
    '''
    Build CNN model for light color classification
    '''
    num_classes = 3
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same',
                     input_shape=(32,16,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()
    plot_model(model, to_file='TL_handling//tl_classifier_model.png')

    return model

if __name__ == '__main__':
    new_content = list()
    with open('images/signlabels_sign_only.csv','rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            new_content.append(row)
    new_content = np.array(new_content)

    # Load images with signs only for training and validation
    images, labels = load_content(new_content)

    print(images.shape, images.dtype)
    print(labels.shape, labels.dtype)

    seq_model = build_model()
    seq_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    # Load data and one hot encode
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Train
    seq_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, shuffle=True, validation_data=(X_test, y_test))

    # Save model
    model_path = 'TL_handling//tl_classifier_'
    save_keras_model(seq_model, 'TL_handling//tl_classifier_')
    time.sleep(10)
