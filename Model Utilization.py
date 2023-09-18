import cv2
import numpy as np
from keras.models import load_model
import joblib

# Load the trained models
models = {
    'Colour': load_model('trained_model_Colour.h5'),
    'Left 1': load_model('trained_model_Left 1.h5'),
    'Left 2': load_model('trained_model_Left 2.h5'),
    'Left 3': load_model('trained_model_Left 3.h5'),
    'Right 1': load_model('trained_model_Right 1.h5'),
    'Right 2': load_model('trained_model_Right 2.h5'),
    'Right 3': load_model('trained_model_Right 3.h5'),
    'Right 4': load_model('trained_model_Right 4.h5')
}

# Load the label encoders
label_encoders = {
    'Colour': joblib.load('label_encoder_Colour.pkl'),
    'Left 1': joblib.load('label_encoder_Left 1.pkl'),
    'Left 2': joblib.load('label_encoder_Left 2.pkl'),
    'Left 3': joblib.load('label_encoder_Left 3.pkl'),
    'Right 1': joblib.load('label_encoder_Right 1.pkl'),
    'Right 2': joblib.load('label_encoder_Right 2.pkl'),
    'Right 3': joblib.load('label_encoder_Right 3.pkl'),
    'Right 4': joblib.load('label_encoder_Right 4.pkl')
}

# Load and preprocess the image
image_path = 'testData/Metal_2.jpg'  # Replace with the path to your image
img = cv2.imread(image_path)
img = cv2.resize(img, (64, 64))  # Resize the image to match the model's expected input size
img = img.flatten()
img = np.array([img])

# Predict the values
predicted_classes = {}
for label, model in models.items():
    prediction = np.argmax(model.predict(img), axis=-1)[0]  # Get the class with the highest probability
    predicted_label = label_encoders[label].inverse_transform([prediction])[0]
    predicted_classes[label] = predicted_label

# Print the predicted class labels
print('Predicted Class Labels:')
for label, prediction in predicted_classes.items():
    print(f'{label}: {prediction}')
