import streamlit as st
import tensorflow as tf
from keras.layers import TextVectorization
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import string
from nltk.tokenize import word_tokenize
import nltk
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Config
st.set_page_config(
    page_title="Deteksi Berita Hoax Indonesia",
    page_icon="📰",
    layout="wide"
)

# ============================================================
# HELPER: Check if model has built-in TextVectorization
# ============================================================

def model_has_text_vectorization(model):
    """Check if model already has TextVectorization layer"""
    for layer in model.layers:
        if isinstance(layer, TextVectorization):
            return True
        if hasattr(layer, 'layers'):  # Check nested layers
            for sublayer in layer.layers:
                if isinstance(sublayer, TextVectorization):
                    return True
    return False

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models_and_vectorizer():
    try:
        # ✅ CRITICAL: Load EXACT same data yang digunakan saat training
        clean_df = pd.read_parquet(r"clean_data.parquet")
        
        # ✅ Gunakan EXACT same split
        X_train = clean_df["combined_features"].sample(frac=0.7, random_state=42)
        
        # ✅ CRITICAL: Parameters HARUS SAMA dengan training
        max_vocab_size = 10000  # Cek di notebook training!
        
        # Calculate avg tokens
        avg_tokens = round(sum([len(str(text).split()) for text in X_train]) / len(X_train))
                
        # Create vectorizer
        text_vectorizer = TextVectorization(
            max_tokens=max_vocab_size,
            output_mode="int",
            output_sequence_length=avg_tokens,
            pad_to_max_tokens=True
        )
        
        # Adapt
        text_vectorizer.adapt(X_train)
        
        # Load models
        models = {}
        model_info = {}
        model_files = {
            "GRU (92.1%)": r"model/gru_model.keras",
            "LSTM (91.3%)": r"model/lstm_model.keras",
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                model = tf.keras.models.load_model(path)
                models[name] = model
                
                has_vectorization = model_has_text_vectorization(model)
                model_info[name] = {
                    'has_vectorization': has_vectorization,
                    'input_shape': model.input_shape
                }
        
        return models, text_vectorizer, avg_tokens, model_info
    
    except Exception as e:
        st.error(f"Error: {e}")
        return {}, None, 0, {}

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_text(text):
    """MUST MATCH training preprocessing EXACTLY"""
    
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ''
    
    try:
        # ✅ CRITICAL: Harus sama persis dengan training
        
        # 1. Lowercase DULU (urutan penting!)
        text = text.lower()
        
        # 2. Remove punctuation
        for p in string.punctuation:
            text = text.replace(p, ' ')
        
        # 3. Tokenize
        tokens = word_tokenize(text)
        
        # 4. Filter alphabetic only
        words_only = [w for w in tokens if w.isalpha()]
        
        # 5. Remove stopwords (pastikan list sama dengan training!)
        stopwords = {
            'ada', 'adalah', 'adanya', 'adapun', 'agak', 'akan', 'akhir', 
            'aku', 'amat', 'anda', 'antar', 'antara', 'apa', 'apabila', 
            'apakah', 'apalagi', 'asal', 'atau', 'atas', 'awal', 'bagai', 
            'bagaimana', 'bagi', 'bahkan', 'bahwa', 'baik', 'bakal', 'banyak',
            'baru', 'bawah', 'beberapa', 'begini', 'begitu', 'belum', 'benar',
            'berada', 'berakhir', 'berapa', 'berbagai', 'beri', 'berikan',
            'berikut', 'berjumlah', 'bermacam', 'bersama', 'besar', 'betul',
            'biasa', 'biasanya', 'bila', 'bisa', 'boleh', 'buat', 'bukan',
            'bulan', 'cara', 'cukup', 'cuma', 'dahulu', 'dalam', 'dan',
            'dapat', 'dari', 'daripada', 'datang', 'dekat', 'demi', 'demikian',
            'dengan', 'depan', 'di', 'dia', 'diantara', 'diberi', 'dibuat',
            'didapat', 'digunakan', 'diingat', 'dijawab', 'dijelaskan',
            'dikarenakan', 'dikatakan', 'dikerjakan', 'diketahui', 'dikira',
            'dilakukan', 'dilihat', 'dimaksud', 'diminta', 'dimulai',
            'dimungkinkan', 'dini', 'dipastikan', 'diperlukan', 'dipersoalkan',
            'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disini',
            'ditambahkan', 'ditanya', 'ditunjukkan', 'diucapkan', 'dong',
            'dua', 'dulu', 'empat', 'enggak', 'entah', 'guna', 'hal', 'hampir',
            'hanya', 'hari', 'harus', 'hendak', 'hingga', 'ia', 'ialah',
            'ibarat', 'ibu', 'ikut', 'ingat', 'ingin', 'ini', 'itu', 'jadi',
            'jangan', 'jauh', 'jawab', 'jelas', 'jika', 'jikalau', 'juga',
            'jumlah', 'justru', 'kala', 'kalau', 'kalian', 'kami', 'kamu',
            'kan', 'kapan', 'karena', 'kasus', 'kata', 'ke', 'keadaan',
            'kebetulan', 'kecil', 'kedua', 'keinginan', 'kelihatan', 'kelima',
            'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kenapa', 'kepada',
            'kesampaian', 'keseluruhan', 'keterlaluan', 'ketika', 'khususnya',
            'kini', 'kira', 'kita', 'kok', 'kurang', 'lagi', 'lah', 'lain',
            'lainnya', 'lalu', 'lama', 'lanjut', 'lebih', 'lewat', 'lima',
            'luar', 'macam', 'maka', 'makin', 'malah', 'mampu', 'mana',
            'masa', 'masalah', 'masih', 'masing', 'mau', 'maupun', 'melainkan',
            'melakukan', 'melalui', 'melihat', 'memang', 'memberi', 'membuat',
            'memerlukan', 'meminta', 'memperbuat', 'mempergunakan',
            'memperkirakan', 'memperlihatkan', 'mempersoalkan', 'mempunyai',
            'memulai', 'memungkinkan', 'menambahkan', 'menanti', 'menanya',
            'mendapat', 'mendatang', 'mengatakan', 'mengenai', 'mengerjakan',
            'mengetahui', 'menggunakan', 'menghendaki', 'mengingat', 'mengira',
            'mengucapkan', 'menjadi', 'menjawab', 'menjelaskan', 'menuju',
            'menunjuk', 'menunjukkan', 'menurut', 'menuturkan', 'menyampaikan',
            'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'merasa',
            'mereka', 'merupakan', 'meski', 'meskipun', 'meyakini', 'minta',
            'mirip', 'misal', 'misalkan', 'misalnya', 'mula', 'mulai', 'mungkin',
            'nah', 'naik', 'namun', 'nanti', 'nyaris', 'nyatanya', 'oleh',
            'pada', 'padahal', 'pak', 'paling', 'panjang', 'pantas', 'para',
            'pasti', 'penting', 'per', 'percuma', 'perlu', 'pernah', 'persoalan',
            'pertama', 'pertanyaan', 'pihak', 'pukul', 'pula', 'pun', 'punya',
            'rasa', 'rasanya', 'rata', 'rupanya', 'saat', 'saja', 'saling',
            'sama', 'sambil', 'sampai', 'sana', 'sangat', 'saya', 'se',
            'sebab', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik',
            'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu',
            'sebelum', 'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar',
            'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'secara', 'sedang',
            'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'segala',
            'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak',
            'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekali', 'sekalian',
            'sekaligus', 'sekalipun', 'sekarang', 'sekecil', 'seketika',
            'sekiranya', 'sekitar', 'sela', 'selain', 'selaku', 'selalu',
            'selama', 'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya',
            'semacam', 'semakin', 'semampu', 'semasa', 'semasih', 'semata',
            'semaunya', 'sementara', 'semisal', 'sempat', 'semua', 'semuanya',
            'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seorang',
            'sepanjang', 'sepantasnya', 'seperlunya', 'seperti', 'sepertinya',
            'sepihak', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat',
            'sesama', 'sesegera', 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya',
            'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah',
            'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidaknya',
            'setinggi', 'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah',
            'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'sudah',
            'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun',
            'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas',
            'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi',
            'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu',
            'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak',
            'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya',
            'teringat', 'terjadi', 'terjadilah', 'terjadinya', 'terkira',
            'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata',
            'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju',
            'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tidak',
            'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk', 'turut',
            'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum',
            'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'waduh',
            'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong',
            'yaitu', 'yakin', 'yakni', 'yang'
        }
        
        # 6. Filter stopwords
        filtered = [w for w in words_only if w not in stopwords]
        
        # 7. Join back
        result = ' '.join(filtered)
        
        # ✅ DEBUGGING
        st.sidebar.info(f"""
        **Preprocessing Steps:**
        - Original words: {len(text.split())}
        - After tokenization: {len(tokens)}
        - After alphabetic filter: {len(words_only)}
        - After stopword removal: {len(filtered)}
        - Final result: {len(result.split())} words
        """)
        
        return result
    
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return ''

# ============================================================
# PREDICTION - ADAPTIVE VERSION
# ============================================================

def predict_news(text, title, model, text_vectorizer, has_built_in_vectorization, threshold=0.5):
    """Prediction with adjustable threshold"""
    
    combined = f"{title} {text}"
    cleaned_text = preprocess_text(combined)
    
    if not cleaned_text.strip():
        return None, None, None, None
    
    try:
        input_array = np.array([cleaned_text], dtype=object)
        
        if has_built_in_vectorization:
            prob = model.predict(input_array, verbose=0)[0][0]
        else:
            vectorized = text_vectorizer(input_array)
            prob = model.predict(vectorized, verbose=0)[0][0]
        
        raw_prob = float(prob)
        
        # ✅ Adjustable threshold
        label = "REAL" if raw_prob > threshold else "HOAX"
        conf = raw_prob if raw_prob > threshold else (1 - raw_prob)
        
        # Show probabilities
        st.info(f"""
        🔍 **Probability Analysis:**
        - Raw Output: {raw_prob:.6f}
        - REAL score: {(1-raw_prob)*100:.2f}%
        - HOAX score: {raw_prob*100:.2f}%
        - Threshold: {threshold}
        - Decision: **{label}**
        """)
        
        return label, conf * 100, cleaned_text, raw_prob
    
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None

# ============================================================
# GAUGE CHART
# ============================================================

def create_gauge_chart(confidence, label):
    color = "#e74c3c" if label == "HOAX" else "#2ecc71"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': f"<b>{label}</b>", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#ecf0f1'},
                {'range': [50, 75], 'color': '#f39c12'},
                {'range': [75, 100], 'color': color}
            ]
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# ============================================================
# MAIN APP
# ============================================================

# Tambahkan CSS di awal file
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;900&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"]  {
    font-family: 'Roboto', Arial, sans-serif;
}
.header-minimal {
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 2px 16px rgba(34,34,59,0.06);
    padding: 2em 2em 1.5em 2em;
    margin-bottom: 2em;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}
.header-minimal h1 {
    font-family: 'Montserrat', Arial, sans-serif;
    font-size: 2.1em;
    font-weight: 800;
    color: #22223b;
    margin-bottom: 0.2em;
    text-align: center;
    letter-spacing: 0.5px;
}
.header-minimal p {
    color: #4a4e69;
    font-size: 1.08em;
    text-align: center;
    margin-bottom: 0.2em;
}
.main-header {
    background: linear-gradient(90deg, #f2e9e4 60%, #c9ada7 100%);
    border-radius: 18px;
    box-shadow: 0 4px 24px rgba(34,34,59,0.08);
    padding: 2.5em 2em 1.5em 2em;
    margin-bottom: 2em;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}
.main-header h1 {
    font-family: 'Montserrat', Arial, sans-serif;
    font-size: 2.7em;
    font-weight: 900;
    color: #22223b;
    margin-bottom: 0.1em;
    letter-spacing: 1px;
    text-align: center;
}
.main-header p {
    color: #4a4e69;
    font-size: 1.15em;
    text-align: center;
    margin-bottom: 0.5em;
}
.stTextInput > div > div > input, .stTextArea > div > textarea {
    background: #f8f9fa;
    border-radius: 10px;
    border: 1.5px solid #c9ada7;
    font-size: 1.08em;
    font-family: 'Roboto', Arial, sans-serif;
}
.stButton > button {
    background-color: #22223b;
    color: #fff;
    border-radius: 10px;
    font-size: 1.1em;
    font-family: 'Montserrat', Arial, sans-serif;
    padding: 0.6em 2.2em;
    margin-top: 1em;
    font-weight: 700;
    letter-spacing: 1px;
    transition: 0.2s;
}
.stButton > button:hover {
    background-color: #4a4e69;
    color: #f2e9e4;
}
.result-box {
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(34,34,59,0.10);
    padding: 2em 1em 1.5em 1em;
    margin-top: 2em;
    text-align: center;
    font-family: 'Montserrat', Arial, sans-serif;
}
.stSidebar {
    font-family: 'Roboto', Arial, sans-serif;
    background: #f8f9fa;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3 {
    font-family: 'Montserrat', Arial, sans-serif;
}
.stSidebar > div {
    margin-bottom: 1.5em;
}
</style>
<div class="header-minimal">
    <h1>Deteksi Berita Hoax Indonesia</h1>
    <p>
        Aplikasi sederhana untuk memeriksa apakah sebuah berita termasuk <b>hoax</b> atau <b>fakta</b>.<br>
        Masukkan judul dan isi berita, lalu klik <b>Deteksi Berita</b> untuk melihat hasil analisis.
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading models..."):
    models, text_vectorizer, avg_tokens, model_info = load_models_and_vectorizer()
    if not models:
        st.error("❌ Model tidak ditemukan!")
        st.stop()

# Bungkus form dan hasil dengan div CSS
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1>Mulai Deteksi Sekarang</h1>", unsafe_allow_html=True)
st.markdown("<h6>Masukkan judul dan isi berita, lalu klik deteksi.</h6>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📰 Deteksi Hoax")
    st.markdown("""
    <div style='font-size:1em; color:#444;'>
        <b>Tentang:</b><br>
        Aplikasi ini membantu Anda memeriksa apakah berita termasuk <b>hoax</b> atau <b>fakta</b>.<br><br>
        <b>Cara Pakai:</b>
        <ul>
            <li>Masukkan judul & isi berita</li>
            <li>Klik <b>Deteksi Berita</b></li>
            <li>Lihat hasil & visualisasi</li>
        </ul>
        <b>Catatan:</b> Hasil prediksi bersifat otomatis, tetap lakukan verifikasi manual jika ragu.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Kelompok 5 Deep Learning - ITERA 2024")
    st.header("Pengaturan")
    selected_model_name = st.selectbox("Pilih Model", list(models.keys()))
    st.subheader("Threshold")
    threshold = st.slider(
        "Sesuaikan threshold prediksi",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Geser untuk mengatur sensitivitas deteksi"
    )
    st.markdown("---")
    st.caption("Kelompok 5 Deep Learning - ITERA 2024")
    st.markdown("""
    <div style='font-size:1em; color:#444;'>
        <b>Tentang Aplikasi</b><br>
        Sistem ini dikembangkan untuk membantu masyarakat dan jurnalis dalam memverifikasi kebenaran berita secara cepat dan mudah.<br><br>
        <b>Cara Kerja:</b>
        <ul>
            <li>Masukkan judul & isi berita</li>
            <li>Model AI akan menganalisis konten</li>
            <li>Hasil prediksi & visualisasi akan muncul</li>
        </ul>   
        <b>Catatan:</b> Hasil prediksi bersifat otomatis dan tidak menggantikan verifikasi manual.
    </div>
    """, unsafe_allow_html=True)

# Input
with st.form("input_form"):
    title = st.text_input("Judul Berita", placeholder="Masukkan judul...")
    text = st.text_area("Isi Berita", placeholder="Masukkan isi berita...", height=200)
    submitted = st.form_submit_button("🔍 Deteksi Berita")

# Definisikan stopwords (copy dari fungsi preprocess_text Anda)
stopwords = {
    'ada', 'adalah', 'adanya', 'adapun', 'agak', 'akan', 'akhir', 
    'aku', 'amat', 'anda', 'antar', 'antara', 'apa', 'apabila', 
    'apakah', 'apalagi', 'asal', 'atau', 'atas', 'awal', 'bagai', 
    'bagaimana', 'bagi', 'bahkan', 'bahwa', 'baik', 'bakal', 'banyak',
    'baru', 'bawah', 'beberapa', 'begini', 'begitu', 'belum', 'benar',
    'berada', 'berakhir', 'berapa', 'berbagai', 'beri', 'berikan',
    'berikut', 'berjumlah', 'bermacam', 'bersama', 'besar', 'betul',
    'biasa', 'biasanya', 'bila', 'bisa', 'boleh', 'buat', 'bukan',
    'bulan', 'cara', 'cukup', 'cuma', 'dahulu', 'dalam', 'dan',
    'dapat', 'dari', 'daripada', 'datang', 'dekat', 'demi', 'demikian',
    'dengan', 'depan', 'di', 'dia', 'diantara', 'diberi', 'dibuat',
    'didapat', 'digunakan', 'diingat', 'dijawab', 'dijelaskan',
    'dikarenakan', 'dikatakan', 'dikerjakan', 'diketahui', 'dikira',
    'dilakukan', 'dilihat', 'dimaksud', 'diminta', 'dimulai',
    'dimungkinkan', 'dini', 'dipastikan', 'diperlukan', 'dipersoalkan',
    'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disini',
    'ditambahkan', 'ditanya', 'ditunjukkan', 'diucapkan', 'dong',
    'dua', 'dulu', 'empat', 'enggak', 'entah', 'guna', 'hal', 'hampir',
    'hanya', 'hari', 'harus', 'hendak', 'hingga', 'ia', 'ialah',
    'ibarat', 'ibu', 'ikut', 'ingat', 'ingin', 'ini', 'itu', 'jadi',
    'jangan', 'jauh', 'jawab', 'jelas', 'jika', 'jikalau', 'juga',
    'jumlah', 'justru', 'kala', 'kalau', 'kalian', 'kami', 'kamu',
    'kan', 'kapan', 'karena', 'kasus', 'kata', 'ke', 'keadaan',
    'kebetulan', 'kecil', 'kedua', 'keinginan', 'kelihatan', 'kelima',
    'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kenapa', 'kepada',
    'kesampaian', 'keseluruhan', 'keterlaluan', 'ketika', 'khususnya',
    'kini', 'kira', 'kita', 'kok', 'kurang', 'lagi', 'lah', 'lain',
    'lainnya', 'lalu', 'lama', 'lanjut', 'lebih', 'lewat', 'lima',
    'luar', 'macam', 'maka', 'makin', 'malah', 'mampu', 'mana',
    'masa', 'masalah', 'masih', 'masing', 'mau', 'maupun', 'melainkan',
    'melakukan', 'melalui', 'melihat', 'memang', 'memberi', 'membuat',
    'memerlukan', 'meminta', 'memperbuat', 'mempergunakan',
    'memperkirakan', 'memperlihatkan', 'mempersoalkan', 'mempunyai',
    'memulai', 'memungkinkan', 'menambahkan', 'menanti', 'menanya',
    'mendapat', 'mendatang', 'mengatakan', 'mengenai', 'mengerjakan',
    'mengetahui', 'menggunakan', 'menghendaki', 'mengingat', 'mengira',
    'mengucapkan', 'menjadi', 'menjawab', 'menjelaskan', 'menuju',
    'menunjuk', 'menunjukkan', 'menurut', 'menuturkan', 'menyampaikan',
    'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh', 'merasa',
    'mereka', 'merupakan', 'meski', 'meskipun', 'meyakini', 'minta',
    'mirip', 'misal', 'misalkan', 'misalnya', 'mula', 'mulai', 'mungkin',
    'nah', 'naik', 'namun', 'nanti', 'nyaris', 'nyatanya', 'oleh',
    'pada', 'padahal', 'pak', 'paling', 'panjang', 'pantas', 'para',
    'pasti', 'penting', 'per', 'percuma', 'perlu', 'pernah', 'persoalan',
    'pertama', 'pertanyaan', 'pihak', 'pukul', 'pula', 'pun', 'punya',
    'rasa', 'rasanya', 'rata', 'rupanya', 'saat', 'saja', 'saling',
    'sama', 'sambil', 'sampai', 'sana', 'sangat', 'saya', 'se',
    'sebab', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian', 'sebaik',
    'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu',
    'sebelum', 'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar',
    'sebetulnya', 'sebisanya', 'sebuah', 'sebut', 'secara', 'sedang',
    'sedangkan', 'sedemikian', 'sedikit', 'sedikitnya', 'segala',
    'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat', 'sejak',
    'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekali', 'sekalian',
    'sekaligus', 'sekalipun', 'sekarang', 'sekecil', 'seketika',
    'sekiranya', 'sekitar', 'sela', 'selain', 'selaku', 'selalu',
    'selama', 'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya',
    'semacam', 'semakin', 'semampu', 'semasa', 'semasih', 'semata',
    'semaunya', 'sementara', 'semisal', 'sempat', 'semua', 'semuanya',
    'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seorang',
    'sepanjang', 'sepantasnya', 'seperlunya', 'seperti', 'sepertinya',
    'sepihak', 'sering', 'seringnya', 'serta', 'serupa', 'sesaat',
    'sesama', 'sesegera', 'sesekali', 'seseorang', 'sesuatu', 'sesuatunya',
    'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah',
    'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidaknya',
    'setinggi', 'seusai', 'sewaktu', 'siap', 'siapa', 'siapakah',
    'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu', 'sudah',
    'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun',
    'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya', 'tandas',
    'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya', 'tapi',
    'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu',
    'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa', 'terbanyak',
    'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya',
    'teringat', 'terjadi', 'terjadilah', 'terjadinya', 'terkira',
    'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata',
    'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju',
    'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tidak',
    'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk', 'turut',
    'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya', 'umum',
    'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'waduh',
    'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun', 'wong',
    'yaitu', 'yakin', 'yakni', 'yang'
}

if submitted:
    word_count = len(text.split())
    if not title.strip() or not text.strip():
        st.warning("Input tidak boleh kosong!")
    elif word_count < 20:
        st.warning("Minimal 20 kata pada isi berita.")
    else:
        with st.spinner("Menganalisis..."):
            model = models[selected_model_name]
            has_vectorization = model_info[selected_model_name]['has_vectorization']
            label, conf, cleaned, raw_prob = predict_news(
                text, title, model, text_vectorizer, has_vectorization, threshold=threshold
            )
            if label:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(
                        f"<h2 style='color: {('#e74c3c' if label=='HOAX' else '#2ecc71')};'>"
                        f"{'🚨' if label=='HOAX' else '✅'} {label}</h2>",
                        unsafe_allow_html=True
                    )
                    st.metric("Confidence", f"{conf:.2f}%")
                with col2:
                    st.plotly_chart(create_gauge_chart(conf, label), width="stretch")
                with st.expander("Lihat Hasil Preprocessing"):
                    st.text_area("Cleaned Text", cleaned, height=100, disabled=True)

                # Gunakan hasil dari fungsi predict_news
                cleaned_text = cleaned if cleaned else ""

                # Tampilkan WordCloud
                st.markdown("#### Visualisasi WordCloud dari Teks Berita")
                wordcloud = WordCloud(width=700, height=300, background_color='white', colormap='viridis').generate(cleaned_text)
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#888; font-size:0.95em; margin-top:2em;'>
    <i>
    Hasil analisis ini menggunakan model deep learning yang telah dilatih pada ribuan berita Indonesia.<br>
    Untuk hasil terbaik, pastikan teks berita yang dimasukkan lengkap dan jelas.
    </i>
</div>
""", unsafe_allow_html=True)