import cv2
import numpy as np
from keras.models import load_model
import os
import random

# Load pre-trained models
age_model = load_model('./utils/age.h5')
gender_model = load_model('./utils/gender.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (200, 200))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to get appropriate ad based on age and gender
def get_ad(age, gender):
    ad_directory = './adDirectory'
    if gender == 0:  # Male
        if 20 <= age <= 25:
            ad_path = os.path.join(ad_directory, 'Male AD 1 (20-35).jpeg')
    else:  # Female
        if 20 <= age <= 25:
            ad_path = os.path.join(ad_directory, 'female AD 1.jpg')
    return ad_path

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for predictions
    preprocessed_frame = preprocess_image(frame)

    # Predict age and gender
    predicted_age = age_model.predict(preprocessed_frame)
    predicted_gender = gender_model.predict(preprocessed_frame)

    age = int(predicted_age[0][0])
    gender = int(predicted_gender[0][0])

    # Get appropriate ad
    ad_path = get_ad(age, gender)

    if ad_path and os.path.exists(ad_path):
        ad_image = cv2.imread(ad_path)
        ad_image = cv2.resize(ad_image, (frame.shape[1], frame.shape[0]))

        # Overlay ad image on the frame
        combined_frame = cv2.addWeighted(frame, 0.5, ad_image, 0.5, 0)

        # Display the combined frame
        cv2.imshow('Ad Display', combined_frame)
    else:
        # Display the original frame if no ad is found
        cv2.imshow('Ad Display', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
