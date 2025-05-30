import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import math

# ----------------------------
# Build SRCNN model
# ----------------------------
def build_srcnn():
    input_img = Input(shape=(256, 256, 3), name='input_layer')
    l1 = Conv2D(64, 9, padding='same', activation='relu', name='conv2d_1')(input_img)
    l2 = Conv2D(32, 3, padding='same', activation='relu', name='conv2d_2')(l1)
    l3 = Conv2D(16, 1, padding='same', activation='relu', name='conv2d_3')(l2)
    l4 = Conv2D(3, 5, padding='same', activation='relu', name='conv2d_4')(l3)
    return Model(inputs=input_img, outputs=l4)

# Load model weights
try:
    model_srcnn = build_srcnn()
    model_srcnn.load_weights("srcnn_model.h5")
except Exception as e:
    st.error(f"Gagal memuat model SRCNN: {str(e)}")
    st.stop()

# ----------------------------
# Helper functions
# ----------------------------
def add_blur(image, blur_level):
    if blur_level == 0:
        return image
    ksize = 2 * blur_level + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def predict_image(model, image):
    # Model expects float32 input in [0,1]
    input_img = image.astype(np.float32)
    input_img = np.expand_dims(input_img, axis=0)  # add batch dim
    pred = model.predict(input_img, verbose=0)
    pred = np.clip(pred[0], 0.0, 1.0)
    return pred

def calculate_metrics(original, restored):
    mse_val = mean_squared_error(original, restored)
    rmse_val = math.sqrt(mse_val)
    psnr_val = peak_signal_noise_ratio(original, restored, data_range=1.0)
    try:
        ssim_val = structural_similarity(original, restored, channel_axis=-1, data_range=1.0)
    except Exception:
        ssim_val = 0.0
    return mse_val, rmse_val, psnr_val, ssim_val

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(layout="wide", page_title="Restorasi Citra dengan SRCNN")
st.title("🖼️ Restorasi Citra dengan SRCNN")

col1, col2, col3 = st.columns(3)

with col1:
    uploaded_file = st.file_uploader("Upload Citra (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])
    blur_level = st.slider("Tingkat Blur", 0, 10, 3)
    proses = st.button("Proses")

if proses and uploaded_file is not None:
    # Load dan resize image ke 256x256
    image = Image.open(uploaded_file).convert("RGB").resize((256,256))
    image_np = np.array(image) / 255.0  # normalisasi ke [0,1]

    # Buat gambar blur (uint8)
    blurred_uint8 = add_blur((image_np * 255).astype(np.uint8), blur_level)
    blurred_norm = blurred_uint8.astype(np.float32) / 255.0

    # Prediksi dengan SRCNN
    output_srcnn = predict_image(model_srcnn, blurred_norm)

    # Simpan hasil ke session state agar bisa dipakai di kolom lain
    st.session_state['original'] = image_np
    st.session_state['blurred'] = blurred_uint8  # tetap uint8 untuk display
    st.session_state['srcnn'] = (output_srcnn * 255).astype(np.uint8)  # convert ke uint8 juga untuk display

if 'original' in st.session_state:
    with col1:
        st.image(st.session_state['blurred'], caption="Citra Blur", use_container_width=True)
    with col2:
        st.image((st.session_state['original'] * 255).astype(np.uint8), caption="Original", use_container_width=True)
    with col3:
        st.image(st.session_state['srcnn'], caption="Hasil SRCNN", use_container_width=True)

    mse_b, rmse_b, psnr_b, ssim_b = calculate_metrics(st.session_state['original'], st.session_state['blurred'] / 255.0)
    mse_s, rmse_s, psnr_s, ssim_s = calculate_metrics(st.session_state['original'], st.session_state['srcnn'] / 255.0)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("### Metrik Sebelum Restorasi (Blurred)")
        st.markdown(f"MSE: {mse_b:.4f}")
        st.markdown(f"RMSE: {rmse_b:.4f}")
        st.markdown(f"PSNR: {psnr_b:.2f} dB")
        st.markdown(f"SSIM: {ssim_b:.4f}")
    with col5:
        st.markdown("### Metrik Setelah Restorasi (SRCNN)")
        st.markdown(f"MSE: {mse_s:.4f}")
        st.markdown(f"RMSE: {rmse_s:.4f}")
        st.markdown(f"PSNR: {psnr_s:.2f} dB")
        st.markdown(f"SSIM: {ssim_s:.4f}")
