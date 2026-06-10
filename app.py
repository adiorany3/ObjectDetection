import streamlit as st
import cv2
from PIL import Image
import tempfile
import os
import glob
from ultralytics import YOLO

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Object Detection App", layout="wide", page_icon="🎯")

# 2. CSS untuk Menghilangkan Emblem & Elemen Bawaan Streamlit
hide_streamlit_style = """
<style>
    /* Sembunyikan menu hamburger (toolbar) di kanan atas */
    div[data-testid="stToolbar"] {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Sembunyikan footer "Made with Streamlit" di bawah */
    footer {visibility: hidden;}
    
    /* Sembunyikan tombol Deploy (jika muncul) */
    .stAppDeployButton {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    
    /* Opsional: Rapikan padding agar lebih rapi */
    .block-container {padding-top: 1rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 3. Judul dan Deskripsi
st.title("🎯 Aplikasi Object Detection Real-time")
st.markdown("Aplikasi ini menggunakan model **YOLOv8** untuk mendeteksi objek dalam gambar dan video secara akurat.")

# 4. Muat Model YOLOv8 (di-cache agar tidak dimuat ulang setiap interaksi)
@st.cache_resource
def load_model():
    # 'yolov8n.pt' adalah model nano (tercepat). Bisa diganti 'yolov8s.pt' jika butuh akurasi lebih tinggi
    return YOLO("yolov8m.pt")

model = load_model()

# 5. Sidebar untuk Pengaturan
st.sidebar.header("⚙️ Pengaturan")
conf_threshold = st.sidebar.slider("Ambang Batas Kepercayaan (Confidence)", 0.0, 1.0, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.info("Model: **YOLOv8 Nano**\n\nDikembangkan oleh Ultralytics.")

# 6. Tab untuk Gambar dan Video
tab1, tab2 = st.tabs(["📷 Deteksi Gambar", "🎥 Deteksi Video"])

# ==========================================
# TAB 1: DETEKSI GAMBAR
# ==========================================
with tab1:
    st.header("Unggah Gambar untuk Dideteksi")
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
            
        if st.button("🔍 Deteksi Objek pada Gambar", type="primary"):
            with st.spinner("Sedang memproses gambar..."):
                # Jalankan inferensi YOLO
                results = model(image, conf=conf_threshold)
                
                # results[0].plot() mengembalikan array numpy dalam format BGR (OpenCV)
                res_plotted = results[0].plot()
                
                # Konversi BGR ke RGB agar warna tampil benar di Streamlit
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.image(res_plotted_rgb, caption="Hasil Deteksi", use_container_width=True)
                
                # Tampilkan detail objek
                st.subheader("📋 Detail Objek Terdeteksi:")
                boxes = results[0].boxes
                if len(boxes) == 0:
                    st.info("Tidak ada objek yang terdeteksi dengan tingkat kepercayaan tersebut.")
                else:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = model.names[cls]
                        st.write(f"- **{name.capitalize()}**: {conf:.2%} confidence")

# ==========================================
# TAB 2: DETEKSI VIDEO
# ==========================================
with tab2:
    st.header("Unggah Video untuk Dideteksi")
    uploaded_video = st.file_uploader("Pilih file video...", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🎬 Deteksi Objek pada Video", type="primary"):
            with st.spinner("Sedang memproses video (mohon tunggu, ini memakan waktu)..."):
                # Simpan video yang diunggah ke file sementara
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile_path = tfile.name
                
                # Jalankan inferensi YOLO pada video dan simpan hasilnya
                model.predict(source=tfile_path, conf=conf_threshold, save=True)
                
                # Cari file video hasil prediksi yang baru saja dibuat
                output_dir = "runs/detect/predict"
                list_of_files = glob.glob(f"{output_dir}/*.mp4")
                
                if list_of_files:
                    # Ambil file terbaru
                    latest_file = max(list_of_files, key=os.path.getctime)
                    st.success("Pemrosesan selesai!")
                    st.video(latest_file)
                    
                    # Bersihkan file sementara
                    os.remove(tfile_path)
                else:
                    st.error("Gagal memproses video. Pastikan format video didukung.")

# Footer Kustom (karena footer Streamlit asli sudah disembunyikan)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Aplikasi Deteksi Objek Kustom by Galuh Adi Insani</p>", unsafe_allow_html=True)
