import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import load_trained_model

# Load the pre-trained model with CUDA if available.
# This improves performance if running on a system with a compatible GPU.
use_cuda = True
model = load_trained_model('./model/best_model.pth', use_cuda=use_cuda)

# Define the image transformations to preprocess images before passing them to the model.
# These transformations standardize the size and color properties of the images,
# matching the requirements of the neural network.
transform = transforms.Compose([
    transforms.Resize(256),             # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),         # Crop the image to 224x224 pixels from the center
    transforms.ToTensor(),              # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize pixel values according to
                   std=[0.229, 0.224, 0.225]),        # the ImageNet training set
])

def predict(image):
    """
    Predict whether an image is a hot dog or not.

    Parameters:
    image (PIL.Image or str): A file path or a PIL.Image object to be classified.

    Returns:
    str: 'Hot Dog' if the model predicts the image is a hot dog, 'Not Hot Dog' otherwise.
    """
    device = next(model.parameters()).device  # Detect if running on GPU or CPU
    image = Image.open(image).convert('RGB')  # Open and convert image to RGB
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension
    output = model(image)  # Get model predictions
    _, predicted = torch.max(output.data, 1)  # Determine predicted class
    return 'Hot Dog' if predicted.item() == 0 else 'Not Hot Dog'

# Streamlit interface setup
st.title('See-Food')  # Set the title of the web app
st.image('src/images/image.png', caption=None, use_column_width=True)  # Display an image under the title
st.header('Hot Dog or Not Hot Dog Classifier')  # Set a header for the app
st.write("Upload an image and the classifier will tell you if it's a hot dog or not!")

# Create a file uploader to allow users to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)  # Display the uploaded image
    st.write("")  # Add a blank line for spacing
    st.write("Classifying...")  # Inform the user that classification is in progress
    label = predict(uploaded_file)  # Get the prediction
    st.write(f"Prediction: {label}")  # Display the prediction result
