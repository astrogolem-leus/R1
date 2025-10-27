import os

# ====== Atasi error OpenCV & GUI di Streamlit Cloud ======
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # cegah GUI error
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # cegah error backend video
os.environ["DISPLAY"] = ":0"  # cegah X11 error di cloud

# ====== Proteksi cv2 agar ultralytics tidak error di Streamlit Cloud ======
try:
    import cv2
except Exception as e:
    import types, sys

    def dummy_func(*args, **kwargs):
        return None

    # Dummy cv2 lengkap agar ultralytics tidak crash
    cv2 = types.SimpleNamespace(
        imshow=dummy_func,
        imread=dummy_func,
        imwrite=dummy_func,
        destroyAllWindows=dummy_func,
        waitKey=dummy_func,
        setNumThreads=dummy_func,
        getBuildInformation=dummy_func,
        IMREAD_COLOR=1,
        IMREAD_GRAYSCALE=0,
        IMREAD_UNCHANGED=-1,
        __version__="0.0"
    )

    sys.modules["cv2"] = cv2
    print("⚠️ OpenCV tidak aktif, menggunakan dummy cv2:", e)

# ====== Import utama ======
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        try:
            # Ambil input shape model
            input_shape = classifier.input_shape
            target_size = (input_shape[1] or 224, input_shape[2] or 224)

            # Pastikan gambar dalam format RGB
            img = img.convert("RGB")

            # Resize otomatis dengan padding agar proporsi tetap
            img_resized = ImageOps.pad(img, target_size, color=(0, 0, 0))

            # Ubah ke array dan normalisasi
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_idx = np.argmax(prediction, axis=1)[0]

            # Label (ubah sesuai dataset kamu)
            labels = ["Kelas 1", "Kelas 2", "Kelas 3"]
            st.success(f"Hasil Prediksi: {labels[class_idx]}")

            st.write("Probabilitas:", np.max(prediction))

        except Exception as e:
            st.error(f"❌ Terjadi error saat klasifikasi: {e}")
            st.stop()
