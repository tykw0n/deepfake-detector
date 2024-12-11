import cv2
import numpy as np
import pickle

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    return np.concatenate([hist_b, hist_g, hist_r]).flatten()

with open("models/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_image(image_path):
    features = extract_features(image_path).reshape(1, -1)
    prediction = model.predict(features)
    return "Real" if prediction[0] == 1 else "Fake"

image_path = "path/to/new_image.jpg"
result = predict_image(image_path)
print(f"The image is: {result}")

