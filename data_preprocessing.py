import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load Excel data with pandas
data = pd.read_excel('data.xlsx')  # Replace 'data.xlsx' with your actual file name

# Preprocess the data
le = LabelEncoder()

# Apply Label Encoding to categorical columns
colour_mapping = {'METAL': 0, 'RED': 1, 'WHITE': 2}
data['Colour'] = data['Colour'].map(colour_mapping)

data[['Left 1', 'Left 2', 'Left 3', 'Right 1', 'Right 2', 'Right 3', 'Right 4']] = data[['Left 1', 'Left 2', 'Left 3', 'Right 1', 'Right 2', 'Right 3', 'Right 4']].apply(le.fit_transform)

# Load and resize images with OpenCV, and then flatten them
images = []
for path in data['Path Name']:
    img = cv2.imread(str(path))
    if img is not None:
        img = cv2.resize(img, (64, 64))  # resize the image to 64x64 pixels
        images.append(img.flatten())
    else:
        print(f'Failed to load image at path: {path}')

# Convert image data and labels to numpy arrays
images = np.array(images)
labels = np.array(data[['Left 1', 'Left 2', 'Left 3', 'Right 1', 'Right 2', 'Right 3', 'Right 4']])  # replace the column names as per your data

# Save preprocessed data to a file
np.savez('preprocessed_data.npz', images=images, labels=labels)
