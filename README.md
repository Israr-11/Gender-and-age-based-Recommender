## `Gender and Age-based Recommender Using Machine Learning`

This project leverages machine learning to recommend advertisements based on the detected gender and age of individuals captured in real-time from a camera feed. The system uses convolutional neural 
networks (CNNs) for gender and age detection.

## Step 1: Collect Dataset
To train the models, a comprehensive dataset is required. This project utilizes a dataset comprising approximately 0.6 million images of males and females. If your local system does not have GPU capabilities, it is recommended to use Google Colab for training.

## Step 2: Training and Evaluation
Train the models using supervised learning techniques. Once trained, evaluate the accuracy of the models by testing them with a subset of the dataset to ensure they are capable of correctly detecting gender and estimating age.

## Step 3: Real-time Implementation
A Python script is provided to:

1. Open the camera feed using OpenCV.
2. Detect the person in real-time.
3. Classify the person’s gender and estimate their age.
4. Recommend the most appropriate advertisement based on the detected age and gender.

## Usage

## Dataset Collection
Ensure you have the dataset of images for training. The dataset should be preprocessed appropriately for the model input.

## Model Training
Use the provided training scripts to train the models on the collected dataset.
Evaluate the models to ensure they meet the desired accuracy.

## Real-time Detection
Use the real-time detection script to capture video from the laptop camera.
The script processes each frame to detect the gender and age of the individual in front of the camera.
Based on the detected attributes, the script displays the appropriate advertisement.

## Requirements
Python 3.x
OpenCV
TensorFlow / Keras
NumPy
Pandas
Matplotlib

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/gender-age-recommender.git
cd gender-age-recommender

2. Install the required packages:

pip install -r requirements.txt

3. Place your dataset in the specified directory.

4. Run the training script to train the models.

5. Run the real-time detection script to start the application:

python detection.py

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

