import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Specify the path to the folder containing the model
models_dir = os.path.join(os.path.dirname(__file__), 'models')
model_path = os.path.join(models_dir, 'best_model_rf.joblib')

# Load the model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please check the path.")
    st.stop()

# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the uploaded image to match the MNIST dataset format.
    - Convert to grayscale.
    - Resize to 28x28 pixels.
    - Invert colors (MNIST uses white digits on a black background).
    - Normalize pixel values to the range [0, 1].
    - Flatten the image to a 1D array of 784 elements.
    """
    image = image.convert('L').resize((28, 28))  # Convert to grayscale and resize
    image_array = np.array(image)
    image_array = 255 - image_array  # Invert colors
    image_array = image_array / 255.0  # Normalize
    return image_array

# Streamlit app title
st.title("Digit Recognition with MNIST")

# Create three columns: left (file upload), middle (model's view), right (results)
col1, col2, col3 = st.columns([1, 1, 1])  # Three equal-width columns

# Left Column: File Uploader and Original Image Preview
with col1:
    st.write("### 1. Upload an Image")
    uploaded_file = st.file_uploader(
        "Upload an image of a digit", 
        type=["png", "jpg", "jpeg"], 
        key="file_uploader"
    )

    if uploaded_file is not None:
        # Display the original uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_container_width=True)

        # Add a reset button
        if st.button("Reset"):
            uploaded_file = None
            st.experimental_rerun()  # Reset the app

# Middle Column: What the Model Sees
with col2:
    if uploaded_file is not None:
        st.write("### 2. What the Model Sees")
        st.write("This is how the model processes your image:")
        
        # Preprocess the image
        image_array = preprocess_image(image)
        
        # Display the preprocessed image (what the model sees)
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap='gray')
        ax.axis('off')  # Hide axes
        st.pyplot(fig)

# Right Column: Prediction Results & Confusion Matrix
with col3:
    if uploaded_file is not None:
        st.write("### 3. Prediction Results")
        
        # Predict using the model
        prediction = model.predict(image_array.reshape(1, -1))
        prediction_proba = model.predict_proba(image_array.reshape(1, -1))  # Get probability scores
        confidence = np.max(prediction_proba) * 100  # Calculate confidence percentage

        # Display the prediction results
        st.write(f"**Predicted Digit:** {prediction[0]}")
        st.write(f"**Prediction Confidence:** {confidence:.2f}%")

        # Display the probability distribution for each class
        st.write("**Probability Distribution:**")
        fig, ax = plt.subplots()
        classes = np.arange(10)  # Digits 0-9
        ax.bar(classes, prediction_proba[0], color='skyblue')
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_xticks(classes)
        ax.set_title("Probability for Each Digit")
        st.pyplot(fig)

        # Display the confusion matrix image (if available)
        image_path = os.path.join(os.path.dirname(__file__), 'images', 'matrix.png')
        if os.path.exists(image_path):
            st.write("**Confusion Matrix:**")
            st.image(image_path, caption="Confusion Matrix", use_container_width=True)