import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

# Path to the dataset
path = "/content/drive/MyDrive/UTKFace"

# Initialize lists for storing images and labels
images = []
genders = []

# Loop through the images in the dataset directory
for img_name in os.listdir(path):
    try:
        age, gender, _ = img_name.split('_')[:3]  # Extract age and gender from filename
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 200))  # Resize images to 200x200
        images.append(img)
        genders.append(int(gender))  # Convert gender to integer (0 for male, 1 for female)
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")
        continue

# Convert lists to numpy arrays
images = np.array(images)
genders = np.array(genders)

# Normalize image data
images = images / 255.0

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(images, genders, test_size=0.2, random_state=42)

# Define the gender detection model
gender_model = Sequential()
gender_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 3)))
gender_model.add(MaxPool2D(pool_size=(2, 2)))
gender_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
gender_model.add(MaxPool2D(pool_size=(2, 2)))
gender_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
gender_model.add(MaxPool2D(pool_size=(2, 2)))
gender_model.add(Flatten())
gender_model.add(Dense(128, activation='relu'))
gender_model.add(Dropout(0.5))
gender_model.add(Dense(1, activation='sigmoid'))

# Compile the model
gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
print(gender_model.summary())

# Train the model
history = gender_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=32)

# Save the trained model
gender_model.save('gender_model.h5')


#This is plot to see how the training performed overall, it can also be used for age trainig

# Plot the training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()
