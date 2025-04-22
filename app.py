#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os

def colorize_image(img):
    # Convert to grayscale if not already grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale to RGB format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Load model files
    model_dir = "models"
    prototxt = os.path.join(model_dir, "models_colorization_deploy_v2.prototxt")
    model = os.path.join(model_dir, "colorization_release_v2.caffemodel")
    points = os.path.join(model_dir, "pts_in_hull.npy")
    
    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    
    # Add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    # Preprocess the image
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    
    # Resize the Lab image to 224x224 and extract the 'L' channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Pass the L channel through the network to predict 'a' and 'b' channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize the predicted 'ab' volume to match input image dimensions
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    
    # Combine original L channel with predicted ab channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    # Convert from Lab to RGB and normalize values
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    
    # Convert to 8-bit integer format
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

def main():
    st.title("B&W Image Colorizer")
    st.write("Upload a black and white image to colorize it with AI.")
    
    # Create a sidebar file uploader
    file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if file is None:
        st.info("Please upload an image file")
    else:
        # Display a spinner while processing
        with st.spinner("Colorizing image..."):
            image = Image.open(file)
            img = np.array(image)
            
            # Create two columns for display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                
            with col2:
                st.subheader("Colorized Image")
                colorized_img = colorize_image(img)
                st.image(colorized_img, use_container_width=True)
                
        st.success("Colorization complete!")
        
        # Add download button for the colorized image
        colorized_pil = Image.fromarray(colorized_img)
        buf = io.BytesIO()
        colorized_pil.save(buf, format="PNG")
        btn = st.download_button(
            label="Download Colorized Image",
            data=buf.getvalue(),
            file_name="colorized_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    import io  # Import here to avoid issues with download button
    main()