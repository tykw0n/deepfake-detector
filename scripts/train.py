import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    return np.concatenate([hist_b, hist_g, hist_r]).flatten()

def load_data(data_dir, label):
    features, labels = [], []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        features.append(extract_features(fpath))
        labels.append(label)
    return np.array(features), np.array(labels)

real_features, real_labels = load_data("data/real_images", 1)  # 진짜: 1
fake_features, fake_labels = load_data("data/fake_images", 0)  # 가짜: 0

X = np.vstack([real_features, fake_features])
y = np.hstack([real_labels, fake_labels])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

import pickle
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

