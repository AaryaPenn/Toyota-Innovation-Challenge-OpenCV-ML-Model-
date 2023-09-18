# Toyota-Innovation-Challenge
Toyota Innovation Challenge, May 2023, @ University of Waterloo

## Team Members
- Aarya Penneru
- Ali Muizz
- Aviral Gupta
- Benedek Boda
- Prasheel Vellanki
- Saaniya Saraf
- Veronika Tadros
- Zain Parihar

# Machine Learning Project README

## Overview

This project is designed for various classification tasks using machine learning models. The project involves training separate models for different labels or classes based on data stored in an Excel file (`data.xlsx`) and making predictions for new data.

### Disclaimer

**Note**: The models, preprocessed data, and raw data files are not included in this repository due to file size constraints. You will need to obtain or generate these files separately to use the project effectively.

## Project Components

### 1. Data Preprocessing (data_preprocessing.py)

This script loads data from an Excel file, preprocesses it, encodes categorical columns, loads and resizes images using OpenCV, and saves the preprocessed data as numpy arrays in a `.npz` file. Key components include:

- Loading data from 'data.xlsx'.
- Label encoding for categorical data.
- Loading, resizing, and flattening images.
- Conversion of data and labels to numpy arrays.

### 2. Model Training (Model Training.py)

This script trains separate machine learning models for various classification tasks using Keras. It loads the preprocessed data, defines neural network models, trains each model, and saves both the models and label encoders. Key components include:

- Loading preprocessed data.
- Definition of neural network models for each label.
- Training models using labeled data.
- Saving trained models and label encoders.

### 3. Model Prediction (Model Utilization.py)

This script loads pre-trained models and label encoders, loads and preprocesses an input image, makes predictions for each trained model, and prints the predicted class labels. Key components include:

- Loading pre-trained machine learning models.
- Loading label encoders.
- Loading and preprocessing an input image.
- Making predictions and mapping them to class labels.

### Important Note

Please note that the models, preprocessed data, and raw data files are not included in this repository due to file size constraints. You must obtain or generate these files separately to use the project effectively.

## Usage

### 1. Clone this repository to your local machine:
   ```bash
    git clone https://github.com/Spazaldinho/Toyota-Innovation-Challenge.git
   ```
### 2. Data Preparation
#### Raw Data
  Obtain the raw data for your specific Toyota Innovation Challenge task. Ensure that the data is structured according to   your project's requirements.
#### Data Preprocessing
  1. Make sure that you have all the required python libraries installed.
  2. Use data_preprocessing.py to preprocess your raw data
  3. The script will encode categorical columns, load and resize images (if applicable), and save the preprocessed data as numpy arrays in a .npz file.
### 3. Model Training
  1. Train separate machine learning models for various classification tasks using the Model Training.py script. Customize the script to match your dataset and task requirements.
  2. The script will load the preprocessed data, define neural network models for each label, train the models using labeled data, and save both the trained models and label encoders.
### 4. Model Prediction
  1. Once you have trained models, you can use them for predictions. Run the Model Utilization.py script, ensuring that you customize it to specify the paths to the trained models and label encoders.
  2. The script will load the pre-trained machine learning models and label encoders, load and preprocess an input image (or data point), make predictions for each model, and print the predicted class labels.
