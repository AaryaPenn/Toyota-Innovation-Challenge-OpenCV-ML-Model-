import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import joblib

# Load Excel data with pandas
data = pd.read_excel('data.xlsx')

# Apply Label Encoding to Colour column
colour_mapping = {'METAL': 0, 'RED': 1, 'WHITE': 2}
data['Colour'] = data['Colour'].map(colour_mapping)

# Load and resize images with OpenCV, and then flatten them
images = []
for path in data['Path Name']:
    img = cv2.imread(str(path))
    if img is not None:
        img = cv2.resize(img, (64, 64))  # resize the image to 256x256 pixels
        images.append(img.flatten())
    else:
        print(f'Failed to load image at path: {path}')

# Convert image data to numpy arrays
images = np.array(images)

# Define model
def create_model(num_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=images.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # One output node for each class
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# For each label, train a separate model
for label in ['Colour', 'Left 1', 'Left 2', 'Left 3', 'Right 1', 'Right 2', 'Right 3', 'Right 4']:
    print(f'Training model for label: {label}')

    le = LabelEncoder()
    encoded_labels = le.fit_transform(data[label])
    num_classes = len(le.classes_)
    joblib.dump(le, f'label_encoder_{label}.pkl')  # Save the label encoder

    model = create_model(num_classes)

    labels = to_categorical(encoded_labels, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    model.save(f'trained_model_{label}.h5')
