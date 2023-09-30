import os
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import io
import numpy as np
import tempfile
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from keras.models import load_model

model= load_model('image_cap_model.h5')

def predict_image(frame):
    model = VGG16(weights='imagenet')
    # # load an image from file
    # image = load_img(image, target_size=(224, 224))
    # # convert the image pixels to a numpy array
    # image = img_to_array(image)
    # # reshape data for the model
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # # prepare the image for the VGG model
    # # expand dimensions to match the batch size of 1
    # image = preprocess_input(image)
    # # predict the probability across all output classes
    # yhat = model.predict(image)
    # # convert the probabilities to class labels
    # label = decode_predictions(yhat)
    # # retrieve the most likely result, e.g. highest probability
    # label = label[0][0]
    # # print the classification
    # return f"{label[1]} {label[2]*100}"
    # Preprocess the frame
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)

    # Predict the frame
    predictions = model.predict(frame)
    label = decode_predictions(predictions, top=1)[0][0]

    return f"{label[1]} ({label[2] * 100:.2f}%)"

    

st.title("Video description")
st.write("Upload a video (Max size: 2MB)")

# Video Upload
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

if uploaded_file is not None:
        # Check file size
        max_size = 2 * 1024 * 1024  # Maximum file size in bytes (2MB)
        file_size = len(uploaded_file.getvalue())
        if file_size <= max_size:
            st.success("Video uploaded successfully.")
            # Create frames directory
            os.makedirs("frames", exist_ok=True)
            # Split video into frames
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            vidcap = cv2.VideoCapture(tfile.name)
            success, image = vidcap.read()
            count = 0
            while success:
                # Save frame
                frame_path = os.path.join("frames", f"frame_{count}.jpg")
                cv2.imwrite(frame_path, image)
                image=Image.open(frame_path)
                image=image.resize((224, 224)) 
                #frame_image = load_img(frame_path)
                frame_label = predict_image(image)
                # Display prediction
                st.write(f"Frame {count + 1} prediction: {frame_label}")
                # Read the next frame
                success, image = vidcap.read()
                count += 1

            st.success("Frames saved successfully.")
        else:
            st.error("File size exceeds the limit of 2MB. Please upload a smaller video.")

            
            
            
            
            
            
