import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict_disease(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    result_index = np.argmax(predictions)
    return result_index

# Streamlit App
def main():
    st.title("Plant Disease Detection App")
    st.sidebar.title("Options")

    # Option to upload image
    uploaded_file = st.sidebar.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"])

    # Load the validation set to access class names
    validation_set = tf.keras.utils.image_dataset_from_directory(
        'valid',
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    class_names = validation_set.class_names

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict_disease(image)
        class_name = class_names[prediction]

        st.write(f"Predicted Disease: {class_name}")

if __name__ == '__main__':
    main()
