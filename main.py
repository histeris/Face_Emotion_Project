import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2
from skimage.feature import local_binary_pattern
from PIL import Image
from skimage.feature import hog

with open('HoG_model.pkl', 'rb') as file:
    hog_model = pickle.load(file)
    scaler_hog = hog_model['scaler_hog']
    model_hog = hog_model['model_hog']
    class_hog = hog_model['class_hog']
    input_size_hog = hog_model['input_size_hog']

with open('LBP_model.pkl', 'rb') as file:
    lbp_model = pickle.load(file)
    scaler_lbp = lbp_model['scaler_lbp']
    model_lbp = lbp_model['model_lbp']
    class_lbp = lbp_model['class_lbp']
    input_size_lbp = lbp_model['input_size_lbp']

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 8 * 1 + 3),
                             range=(0, 8 * 1 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def preprocess_and_predict_lbp(image_path, model, scaler):
    # Baca gambar
    # image = cv2.imread(image_path)
    if image_path is None:
        raise ValueError(f"Gambar tidak ditemukan atau tidak valid.")
    haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = haar_cascade.detectMultiScale(image_path, 1.2, 5)
    if len(face) < 1:
        cropped_img = image_path
    for face_rect in face:
        x,y,h,w = face_rect
        cropped_img = image_path[y:y+h, x:x+w]
    # Preprocessing
    input_size = (64, 64)
    resized_image = cv2.resize(cropped_img, input_size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Ekstraksi fitur LBP
    feature = extract_lbp_features(blurred_image)
    feature = np.array([feature])  # Ubah ke bentuk 2D (1, n_features)
    
    # Standarisasi
    feature_scaled = scaler.transform(feature)
    
    # Prediksi dengan model
    prediction = model.predict(feature_scaled)
    
    return prediction[0]

def preprocess_and_predict_hog(image_path, model, scaler, hog_params):
    # Baca gambar
    # image = cv2.imread(image_path)
    if image_path is None:
        raise ValueError(f"Gambar di {image_path} tidak ditemukan atau tidak valid.")
    haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = haar_cascade.detectMultiScale(image_path, 1.2, 5)
    if len(face) < 1:
        cropped_img = image_path
    for face_rect in face:
        x,y,h,w = face_rect
        cropped_img = image_path[y:y+h, x:x+w]
    # Preprocessing
    input_size = (64, 64)
    resized_image = cv2.resize(cropped_img, input_size)
    # Preprocessing
    input_size = (64, 64)
    resized_image = cv2.resize(image_path, input_size)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Ekstraksi fitur menggunakan HOG
    feature = hog(blurred_image, **hog_params)
    feature = np.array([feature])  # Ubah ke bentuk 2D (1, n_features)
    
    # Standarisasi
    feature_scaled = scaler.transform(feature)
    
    # Prediksi dengan model
    prediction = model.predict(feature_scaled)
    
    return prediction[0] 


def main():
    st.title("Deteksi Ekspresi Wajah Menggunakan LBP dan Hog")

    uploaded_file = st.file_uploader("Unggar gambar wajah Menggunakan LBP", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:

            image_data = Image.open(uploaded_file)
            st.image(image_data, caption='Gambar yang diunggah', use_column_width=True)
        
            if st.button('Submit (LBP)'):
                image_data = np.array(image_data)
                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                image = preprocess_and_predict_lbp(image_data, model_lbp, scaler_lbp)
                st.write(f"Prediction: {class_lbp[int(image)]}")
        except Exception as e:
            st.error(f"Terjadi kesalahan : {e}")   

    uploaded_file_hog = st.file_uploader("Unggar gambar wajah Menggunakan HoG", type=['jpg', 'jpeg', 'png'])
    if uploaded_file_hog is not None:
        try:

            image_datas = Image.open(uploaded_file_hog)
            st.image(image_datas, caption='Gambar yang diunggah', use_column_width=True)
        
            if st.button('Submit (HOG)'):
                image_datas = np.array(image_datas)
                image_datas = cv2.cvtColor(image_datas, cv2.COLOR_RGB2BGR)
                images = preprocess_and_predict_hog(image_datas, model_hog, scaler_hog, hog_params)
                st.write(f"Prediction: {class_lbp[int(images)]}")
        except Exception as e:
            st.error(f"Terjadi kesalahan : {e}")   
         


if __name__ == '__main__':
    main()