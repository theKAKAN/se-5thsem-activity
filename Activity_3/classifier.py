import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import cv2

# ====================== LOADING AND INITIALIZING ======================

def load_data(folder_path):
    X = []
    y = []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)

        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)

            # Read and resize the image (adjust the size as needed)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (64, 64))  # Adjust the size as needed

            # Flatten the image to a 1D array
            img_flattened = img.flatten()

            X.append(img)
            y.append(label)

    # Convert the labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return np.array(X).reshape(len(X), 3*64*64), np.array(y_encoded)

# Load training data
X_train, y_train = load_data('Train')

# Load test data
X_test, y_test = load_data('Test')


# ====================== TRAINING AND TESTING ======================

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# SVM
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Neural Network
nn_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
nn_classifier.fit(X_train, y_train)
nn_predictions = nn_classifier.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)


# ====================== RESULTS ======================
print(f'KNN Accuracy: {knn_accuracy}')
print(f'SVM Accuracy: {svm_accuracy}')
print(f'Random Forest Accuracy: {rf_accuracy}')
print(f'Neural Network Accuracy: {nn_accuracy}')
