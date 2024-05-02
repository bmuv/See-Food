# See-Food

## Overview
"See-Food" is inspired by the fictional "SeeFood" app featured in the comedy TV show *Silicon Valley* - https://www.youtube.com/watch?v=vIci3C4JkL0. Originally conceived as a "Shazam for food," the development in the show stalled when the algorithm could only detect hot dogs. Emulating this concept, See-Food is a playful take on image classification, designed to identify whether an image is a hot dog or not. This project serves as an educational tool to explore and understand the basics of image classification using deep learning.
Dataset : https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog

## Technology Stack
- **PyTorch**: Utilized for building and training the neural network.
- **Streamlit**: Used to create a user-friendly web interface that allows users to interact with the model.

## Setup
To get See-Food up and running, follow these steps:

### Prerequisites
Ensure you have Python installed on your machine. It is recommended to use a virtual environment to manage dependencies.

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/bmuv/See-Food.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage
To use SeeFud, you can train the model, test its accuracy, or launch the app for inference:

 **Training the model**:
  ```
  python ./src/train.py
  ```
 **Testing the model**:
  ```
  python ./src/test.py
  ```
 **Launching the app**:
  ```
  streamlit run ./src/app.py
  ```

## How to Use
Once the app is running, simply:
1. Navigate to the provided local URL (typically `http://localhost:8501`).
2. Upload an image using the upload button.
3. Let the classifier determine whether the uploaded image is a hot dog or not.

## Features
- **Educational Tool**: Great for beginners looking to dive into the world of machine learning and image classification.
- **Pretrained MobileNetV2**: Leverages a lightweight yet powerful neural network that achieves an accuracy of 82.50% after 50 epochs of training.
