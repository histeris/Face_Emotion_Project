import streamlit as st
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from PIL import Image
import pickle

# Load Models
with open('HoG_model.pkl', 'rb') as file:
    hog_model = pickle.load(file)
    scaler_hog = hog_model['scaler_hog']
    model_hog = hog_model['model_hog']
    class_hog = hog_model['class_hog']

with open('LBP_model.pkl', 'rb') as file:
    lbp_model = pickle.load(file)
    scaler_lbp = lbp_model['scaler_lbp']
    model_lbp = lbp_model['model_lbp']
    class_lbp = lbp_model['class_lbp']

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# Feature Extraction Functions
def extract_lbp_features(image):
    """Extract LBP features from the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 8 * 1 + 3), range=(0, 8 * 1 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def preprocess_and_predict(image, model, scaler, method="LBP"):
    """Preprocess and predict the emotion using the chosen method."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image input.")

    input_size = (64, 64)
    resized_image = cv2.resize(image, input_size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) if len(resized_image.shape) == 3 else resized_image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    if method == "LBP":
        feature = extract_lbp_features(blurred_image)
    elif method == "HOG":
        feature = hog(blurred_image, **hog_params)
    else:
        raise ValueError(f"Unknown method: {method}")

    feature = np.array([feature])  # Convert to 2D
    feature_scaled = scaler.transform(feature)
    prediction = model.predict(feature_scaled)
    return prediction[0]

# Streamlit App
def main():
    st.title("Real-Time Emotion Detection with LBP and HOG")

    # Access webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    # Load Haar cascade for face detection
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam. Please check your camera settings.")
            break

        # Convert frame to grayscale and detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            try:
                # Predict emotion using LBP
                lbp_prediction = preprocess_and_predict(face_roi, model_lbp, scaler_lbp, method="LBP")
                lbp_label = class_lbp[int(lbp_prediction)]

                # Predict emotion using HOG
                hog_prediction = preprocess_and_predict(face_roi, model_hog, scaler_hog, method="HOG")
                hog_label = class_hog[int(hog_prediction)]

                # Draw bounding box and predictions
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"LBP: {lbp_label}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"HOG: {hog_label}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Display frame in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
