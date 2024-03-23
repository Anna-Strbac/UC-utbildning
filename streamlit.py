import streamlit as st
import numpy as np
import joblib
import cv2
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import center_of_mass

# Load the model 

model = 'C:/Users/46704/svm_clf.pkl'
svm_clf_model = joblib.load(model)



# Process the image

def processImage(input):
        
    # Read input
    data = input.getvalue()

    # Decode to grayscale
    img_gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Increase brightness
    brightness_increase = 50
    brightened_image = cv2.add(img_gray, brightness_increase)

    # Increase contrast
    contrast_increase = 1.5
    mean_gray = np.mean(brightened_image)
    adjusted_image = np.clip((contrast_increase * (brightened_image.astype(np.float32) - mean_gray) + mean_gray),
                                      0, 255).astype(np.uint8)
    
    # Invert the image
    inverted_img = 255 - adjusted_image

    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(inverted_img, (28, 28), interpolation=cv2.INTER_AREA)

    # Apply thresholding
    _, thresholded_img = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)

    # Applying cropping
    while np.sum(thresholded_img[0]) == 0:
        thresholded_img = thresholded_img[1:]
    while np.sum(thresholded_img[:,0]) == 0:
        thresholded_img = np.delete(thresholded_img,0,1)
    while np.sum(thresholded_img[-1]) == 0:
        thresholded_img = thresholded_img[:-1]
    while np.sum(thresholded_img[:,-1]) == 0:
        thresholded_img = np.delete(thresholded_img,-1,1)
    rows, cols = thresholded_img.shape

    if thresholded_img.size == 0:
        print("After cropping, the image turned out to be empty")
    else:
        rows, cols = thresholded_img.shape

    # Resize the image to fit within a 20x20 pixel box
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        thresholded_shifted = cv2.resize(thresholded_img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        thresholded_shifted = cv2.resize(thresholded_img, (cols, rows))

    # Create a new image with dimensions of 20x20 pixels
    new_image = np.zeros((20, 20))

    # Copy the modified image to the center of the new image
    row_offset = (20 - rows) // 2
    col_offset = (20 - cols) // 2
    new_image[row_offset:row_offset + rows, col_offset:col_offset + cols] = thresholded_shifted

    # Copy the resized image into the center of the new image
    row_offset = (20 - rows) // 2
    col_offset = (20 - cols) // 2
    new_image[row_offset:row_offset + rows, col_offset:col_offset + cols] = thresholded_shifted

    # Applying Gaussian Blur
    blurred = cv2.GaussianBlur(new_image, (5, 5), 0)

    rows, cols = blurred.shape

    if blurred.size == 0:
        print("After cropping, the image turned out to be empty")
    else:
        rows, cols = blurred.shape

    # Creating a new image with dimensions of 28x28 pixels
    final_image = np.zeros((28, 28), dtype=np.uint8)

    # Calculate offsets to place the 20x20 image in the center of the 28x28 image
    row_offset = (28 - 20) // 2
    col_offset = (28 - 20) // 2

    # Copy the 20x20 image into the center of the 28x28 image
    final_image[row_offset:row_offset + 20, col_offset:col_offset + 20] = blurred

    # Defining a function to calculate the best shift for centering the image
    def getBestShift(final_image):
        cy,cx = center_of_mass(final_image)
    
        rows,cols = final_image.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)

        return shiftx,shifty
  
    # Defining a function to shift the image
    def shift(final_image,sx,sy):
        rows,cols = final_image.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(final_image,M,(cols,rows))
        return shifted


    # Getting the necessary shifts for centering the object
    shiftx, shifty = getBestShift(final_image)

    # Shifting the image
    shifted = shift(final_image, shiftx, shifty)

    # Display the processed image
    st.write("Processed image: ")
    st.image(shifted, width=128 ,output_format="auto", clamp=True)

    if st.button('Prediction'):
       prediction(shifted)


    
def prediction(inputImg):
     
    # Model prediction
    flat = inputImg.flatten().reshape(1, -1)
    st.write("Predicted digit: ", svm_clf_model.predict(flat))




# Building a sidebar
with st.sidebar:

    option = st.radio(
        "Choose the option:",
        ("Camera", "Download"),
        captions = ["Use your camera", "Download a picture (jpg, jpeg, png)"]
    )

if option == 'Camera':
    buffer = st.camera_input("Take a picture of handwritten digit!")

    if buffer is not None:
        processImage(buffer)
        
else:
    upload = st.file_uploader("Download a picture of handwritten digit:", type=['png', 'jpeg', 'jpg'])

    if upload is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Uploaded image: ")
            st.image(upload, width=128)
        with col2:
            processImage(upload)