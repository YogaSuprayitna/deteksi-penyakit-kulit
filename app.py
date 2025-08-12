import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

model = load_model('model_dt_kulit.h5')
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

tab1, tab2 = st.tabs(["🩺 Deteksi", "👤 Tentang"])

with tab1:
    st.title("🩺 Deteksi Penyakit Kulit dengan CNN")
    st.write("Upload gambar kulit atau gunakan kamera untuk prediksi penyakit kulit.")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    input_method = st.radio("Pilih metode input gambar:", ["📁 Upload Gambar", "📷 Kamera"])

    img = None
    if input_method == "📁 Upload Gambar":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")

    elif input_method == "📷 Kamera":
        camera_image = st.camera_input("Ambil foto kulit")
        if camera_image is not None:
            img = Image.open(camera_image).convert("RGB")

    st.markdown('</div>', unsafe_allow_html=True)

    if img is not None:
        st.image(img, caption="Gambar yang dipilih", use_container_width=True)

        img_resized = img.resize((200, 200))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(prediction[0])) * 100

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"### 🔍 Prediksi: **{predicted_label.capitalize()}**")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.title("👤 Tentang Aplikasi")
    st.markdown("""
    <div class="upload-section">
        <h3>Tentang Aplikasi</h3>
        <p>Aplikasi ini dirancang untuk membantu pengguna dalam mendeteksi berbagai jenis penyakit kulit menggunakan teknologi <strong>Convolutional Neural Network (CNN)</strong>. Cukup dengan mengunggah gambar kulit atau menggunakan kamera secara langsung, aplikasi ini akan menganalisis gambar dan memberikan prediksi berdasarkan data pelatihan yang telah dikembangkan.</p>
        <p>Jenis penyakit kulit yang dapat dikenali oleh aplikasi ini meliputi:</p>
        <ul>
            <li><strong>Acne</strong> – Jerawat akibat pori-pori tersumbat oleh minyak dan sel kulit mati</li>
            <li><strong>Eksim</strong> – Peradangan kulit yang menyebabkan kemerahan, gatal, dan kering</li>
            <li><strong>Herpes</strong> – Infeksi kulit akibat virus yang ditandai dengan lepuhan</li>
            <li><strong>Panu</strong> – Infeksi jamur pada kulit yang menyebabkan bercak putih atau coklat</li>
            <li><strong>Rosacea</strong> – Kondisi kulit kronis yang menyebabkan kemerahan pada wajah</li>
        </ul>
    </div>

    """, unsafe_allow_html=True)
