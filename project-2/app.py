import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import h5py

def load_model_h5(filepath):
    parameters = {}
    with h5py.File(filepath, 'r') as file:
        for key in file.keys():
            parameters[key] = file[key][()]
    return parameters

# # Load the model
def custom_load_model(filepath):
    return load_model_h5(filepath)

# Load the model
# @st.cache(allow_output_mutation=True)
def load_model():
    model_parameters = custom_load_model('/media/f4h1m/Dell External/CSE3200/Binary_classification/project-2/my-model_epoch_400.h5')
    return model_parameters

# model = load_model()

# Load the model
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model('/media/f4h1m/Dell External/CSE3200/Binary_classification/project-2/my-model_epoch_400.h5')
#     return model



# Function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Assuming your model expects 224x224 images
    img_array = img.astype(np.float32) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
st.title("Fungus Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption="Uploaded Image.")
    st.write("")
    st.write("Classifying...")


    model = load_model()
    prediction = predict(image, model)
    


    # Assuming your model predicts probabilities for each class
    # You might need to adjust this part based on your model's output
    # If you have two classes, H5 and H6
    class_names = ['H5', 'H6']
    max_prob_class = np.argmax(prediction)
    predicted_class = class_names[max_prob_class]
    
    st.write("Predicted Class:", predicted_class)


































