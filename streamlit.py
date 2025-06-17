import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Konfigurasi Penting ---
MODEL_PATH = 'model_naive_bayes.pkl'
DATA_PATH = 'processed_heart_disease_data.csv'

# Kolom target Anda: 'num'pip show pandas
TARGET_COLUMN = 'num' 

# Kolom-kolom yang DI-DROP saat training model (tidak digunakan sebagai fitur)
# Termasuk kolom yang tidak diinput user DAN kolom target 'num'
DROPPED_COLUMNS = [
    'chol',
    'trestbps',
    'restecg',
    'fbs',
    TARGET_COLUMN 
]

# --- Fungsi Pembantu (Tidak Memanggil st.* secara langsung) ---
@st.cache_resource
def load_model(path):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return {"success": True, "data": model}
    except FileNotFoundError:
        return {"success": False, "message": f"Error: File model tidak ditemukan di '{path}'. Pastikan file 'model_naive_bayes.pkl' ada di direktori yang sama."}
    except Exception as e:
        return {"success": False, "message": f"Error saat memuat model: {e}"}

@st.cache_data
def load_data(path):
    try:
        data = pd.read_csv(path)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        return {"success": True, "data": data}
    except FileNotFoundError:
        return {"success": False, "message": f"Error: File data tidak ditemukan di '{path}'. Pastikan file 'processed_heart_disease_data.csv' ada di direktori yang sama."}
    except pd.errors.EmptyDataError:
        return {"success": False, "message": f"Error: File '{path}' kosong. Pastikan ada data di dalam CSV."}
    except Exception as e:
        return {"success": False, "message": f"Error saat memuat data CSV: {e}"}

def get_model_accuracy(model, data_df, target_col, dropped_cols_from_features):
    if target_col not in data_df.columns:
        return {"success": False, "message": f"Error: Kolom target '{target_col}' tidak ditemukan dalam data untuk menghitung akurasi."}

    X = data_df.drop(columns=[target_col])
    y_raw = data_df[target_col]

    y_binary = (y_raw > 0).astype(int) 

    actual_dropped_cols_for_X = [col for col in dropped_cols_from_features if col in X.columns]
    if actual_dropped_cols_for_X:
        X = X.drop(columns=actual_dropped_cols_for_X)

    try:
        class_counts = y_binary.value_counts()
        stratify_needed = False
        
        min_samples_for_stratify = 2 
        if len(class_counts) > 1 and class_counts.min() >= min_samples_for_stratify: 
            stratify_needed = True
            
        if stratify_needed:
            X_train, X_test_for_acc, y_train, y_test_for_acc = train_test_split(
                X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
            )
        else:
            st.info(f"Catatan: Pembagian data untuk akurasi dilakukan tanpa stratifikasi karena populasi kelas terlalu kecil atau hanya ada satu kelas ({class_counts.min() if len(class_counts) > 1 else 'N/A'} sampel di kelas minoritas).")
            X_train, X_test_for_acc, y_train, y_test_for_acc = train_test_split(
                X, y_binary, test_size=0.2, random_state=42
            )
        
        if X_test_for_acc.empty:
            return {"success": False, "message": "Dataset terlalu kecil untuk pembagian test set (setelah di-split). Akurasi tidak dapat dihitung."}

        y_pred_raw = model.predict(X_test_for_acc)
        y_pred_binary = (y_pred_raw > 0).astype(int) 

        accuracy = accuracy_score(y_test_for_acc, y_pred_binary) 
        return {"success": True, "accuracy": f"{accuracy:.2f}"}
    except Exception as e:
        return {"success": False, "message": f"Terjadi kesalahan saat menghitung akurasi: {e}. Periksa apakah dimensi dan kolom X_test ({X_test_for_acc.shape if 'X_test_for_acc' in locals() else 'N/A'}) sesuai dengan yang diharapkan model. Kolom X_test: {X_test_for_acc.columns.tolist() if 'X_test_for_acc' in locals() else 'N/A'}"}

def main():
    model_result = load_model(MODEL_PATH)
    data_result = load_data(DATA_PATH)

    if not model_result["success"]:
        st.error(model_result["message"])
        st.stop()
    model = model_result["data"]

    if not data_result["success"]:
        st.error(data_result["message"])
        st.stop()
    processed_data_for_features = data_result["data"]

    all_features_in_original_data = processed_data_for_features.columns.tolist()
    feature_names_for_model = [col for col in all_features_in_original_data if col not in DROPPED_COLUMNS]

    if not feature_names_for_model:
        st.error("Error Konfigurasi: Tidak ada fitur yang tersisa untuk model setelah menghilangkan kolom yang di-drop dan kolom target. Harap periksa `DROPPED_COLUMNS` dan `processed_heart_disease_data.csv` Anda.")
        st.stop()

    required_user_inputs = ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    missing_required_features = [f for f in required_user_inputs if f not in feature_names_for_model]
    if missing_required_features:
        st.error(f"Error Konfigurasi: Fitur-fitur penting yang harus diinput user tidak ditemukan di data setelah dropping: {', '.join(missing_required_features)}. Harap periksa file `processed_heart_disease_data.csv` dan daftar `DROPPED_COLUMNS` Anda.")
        st.stop()
    
    st.sidebar.title("Informasi Aplikasi")
    st.sidebar.markdown(
        """
        Aplikasi ini adalah alat bantu prediksi risiko penyakit jantung.
        Model dilatih menggunakan data pasien.
        """
    )
    st.sidebar.markdown("Kelompok 5")
    st.sidebar.markdown("- Muhammad Akhtar Perwira (1103220150)")
    st.sidebar.markdown("- Enrico Chandra Nugroho (1103220234)")
    st.sidebar.markdown("- Khalif Aziz Prawira (1103223045)")

    st.title("Aplikasi Prediksi Penyakit Jantung")
    st.markdown("Gunakan aplikasi ini untuk memprediksi risiko penyakit jantung berdasarkan data pasien.")

    st.header("Informasi Model & Performa") 
    st.write(f"**Tipe Model:** Naive Bayes Classifier")

    col1_info, col2_info = st.columns([1, 2])
    
    with col1_info:
        accuracy_result = get_model_accuracy(model, processed_data_for_features.copy(), TARGET_COLUMN, [c for c in DROPPED_COLUMNS if c != TARGET_COLUMN])
        if accuracy_result["success"]:
            st.metric(label="Akurasi Model", value=f"{float(accuracy_result['accuracy'])*100:.0f}%")
        else:
            st.error(f"Gagal menghitung akurasi: {accuracy_result['message']}")
    
    with col2_info:
        st.info(
            f"""
            Model ini memprediksi apakah seseorang terkena penyakit jantung atau tidak.
            Nilai target asli (`{TARGET_COLUMN}`) diinterpretasikan sebagai:
            - **0:** Tidak ada penyakit jantung
            - **>0:** Ada penyakit jantung (digabungkan menjadi satu kelas 'Ada')
            """
        )
            
    st.markdown("---") 

    st.header("Masukkan Data Pasien")
    st.write("Silakan isi detail pasien di bawah ini untuk mendapatkan prediksi penyakit jantung.")

    num_cols = 2 
    cols = st.columns(num_cols)
    col_idx = 0

    user_input = {}
    
    placeholders = {
        'age': "Contoh: 45",
        'thalach': "Contoh: 150",
        'oldpeak': "Contoh: 1.5",
    }

    tooltip_info = {
        'age': "Usia pasien dalam tahun. Rentang: {min_val} - {max_val}.",
        'sex': "Jenis kelamin pasien. 0: Perempuan, 1: Laki-laki.",
        'cp': "Jenis nyeri dada. 0: Tipe khas angina, 1: Tipe atipikal angina, 2: Nyeri non-angina, 3: Asimptomatik.",
        'thalach': "Detak jantung maksimum yang dicapai. Rentang: {min_val} - {max_val}.",
        'exang': "Angina yang diinduksi olahraga. 0: Tidak, 1: Ya.",
        'oldpeak': "Depresi ST yang diinduksi oleh latihan relatif terhadap istirahat. Rentang: {min_val} - {max_val} (dengan satu angka di belakang koma).",
        'slope': "Kemiringan puncak segmen ST latihan. 0: Upsloping, 1: Flat, 2: Downsloping.",
        'ca': "Jumlah pembuluh darah utama (0-3) yang diwarnai oleh fluoroskopi. Rentang: {min_val} - {max_val}.",
        'thal': "Hasil tes stres talasemia: 1=Normal, 2:Cacat tetap, 3:Cacat reversibel. Rentang: {min_val} - {max_val}."
    }

    min_max_values = {}
    for feature in feature_names_for_model:
        if feature in processed_data_for_features.columns:
            min_val = processed_data_for_features[feature].min()
            max_val = processed_data_for_features[feature].max()
            min_max_values[feature] = {
                'min': f"{min_val:.0f}" if 'int' in str(processed_data_for_features[feature].dtype) and pd.notna(min_val) else f"{min_val:.1f}" if pd.notna(min_val) else 'N/A',
                'max': f"{max_val:.0f}" if 'int' in str(processed_data_for_features[feature].dtype) and pd.notna(max_val) else f"{max_val:.1f}" if pd.notna(max_val) else 'N/A'
            }

    # Definisikan fitur numerik murni dan kategorikal secara eksplisit untuk konsistensi key
    NUMERIC_FEATURES = ['age', 'thalach', 'oldpeak']
    CATEGORICAL_FEATURES = ['sex', 'exang', 'cp', 'slope', 'ca', 'thal']

    for feature in feature_names_for_model:
        with cols[col_idx]: 
            if feature in processed_data_for_features.columns: 
                dtype = processed_data_for_features[feature].dtype
                
                current_tooltip = tooltip_info.get(feature, "Masukkan nilai untuk fitur ini.")
                if '{min_val}' in current_tooltip or '{max_val}' in current_tooltip:
                     current_tooltip = current_tooltip.format(
                        min_val=min_max_values.get(feature, {}).get('min'), 
                        max_val=min_max_values.get(feature, {}).get('max')
                     )
                
                if feature in CATEGORICAL_FEATURES:
                    unique_vals = processed_data_for_features[feature].unique()
                    options_raw = sorted([str(x) for x in unique_vals if pd.notna(x)])
                    
                    display_options = ["Pilih..."] 
                    value_options_map = {} 
                    
                    if feature == 'sex':
                        display_options.extend(['Perempuan', 'Laki-laki']) 
                        value_options_map = {'Perempuan': '0', 'Laki-laki': '1'}
                    elif feature == 'exang':
                        display_options.extend(['Tidak', 'Ya']) 
                        value_options_map = {'Tidak': '0', 'Ya': '1'}
                    elif feature == 'cp':
                        display_options.extend(['Tipe Khas Angina (0)', 'Tipe Atipikal Angina (1)', 'Nyeri Non-Angina (2)', 'Asimptomatik (3)'])
                        value_options_map = {'Tipe Khas Angina (0)': '0', 'Tipe Atipikal Angina (1)': '1', 'Nyeri Non-Angina (2)': '2', 'Asimptomatik (3)': '3'}
                    elif feature == 'slope':
                        display_options.extend(['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'])
                        value_options_map = {'Upsloping (0)': '0', 'Flat (1)': '1', 'Downsloping (2)': '2'}
                    elif feature == 'ca':
                        display_options.extend(['0', '1', '2', '3'])
                        value_options_map = {'0': '0', '1': '1', '2': '2', '3': '3'}
                    elif feature == 'thal':
                        display_options.extend(['Normal (1)', 'Cacat Tetap (2)', 'Cacat Reversibel (3)'])
                        value_options_map = {'Normal (1)': '1', 'Cacat Tetap (2)': '2', 'Cacat Reversibel (3)': '3'}
                    else: 
                        display_options.extend(options_raw)
                        value_options_map = {disp: val for disp, val in zip(display_options[1:], options_raw)}


                    selected_display_value = st.selectbox(
                        f"**{feature.replace('_', ' ').title()}**",
                        options=display_options,
                        index=0, 
                        key=f"input_{feature}", # Consistent key
                        help=current_tooltip 
                    )
                    
                    if selected_display_value == "Pilih...":
                        user_input[feature] = None 
                    else:
                        user_input[feature] = value_options_map.get(selected_display_value)
                        if user_input[feature] is not None and (processed_data_for_features[feature].dtype == 'int64' or processed_data_for_features[feature].dtype == 'float64'):
                             user_input[feature] = float(user_input[feature]) 
                             if 'int' in str(processed_data_for_features[feature].dtype):
                                 user_input[feature] = int(user_input[feature])

                elif feature in NUMERIC_FEATURES: # Hanya untuk fitur numerik murni
                    
                    placeholder_text = placeholders.get(feature, f"Masukkan nilai")
                    
                    user_val_str = st.text_input(
                        f"**{feature.replace('_', ' ').title()}**",
                        value="",
                        placeholder=placeholder_text,
                        key=f"input_{feature}_text", # Consistent key
                        help=current_tooltip 
                    )

                    if user_val_str: 
                        try:
                            val_to_convert = float(user_val_str)
                            if 'int' in str(dtype):
                                user_input[feature] = int(val_to_convert) 
                            else: 
                                user_input[feature] = val_to_convert
                            
                            min_val = processed_data_for_features[feature].min()
                            max_val = processed_data_for_features[feature].max()
                            if pd.notna(min_val) and pd.notna(max_val):
                                if user_input[feature] < min_val or user_input[feature] > max_val:
                                    st.warning(f"Nilai untuk {feature.replace('_', ' ').title()} di luar rentang historis ({min_max_values[feature]['min']}-{min_max_values[feature]['max']}).")

                        except ValueError:
                            st.error(f"Input untuk {feature.replace('_', ' ').title()} harus berupa angka.")
                            user_input[feature] = None 
                    else:
                        user_input[feature] = None 
                    
                else: # Fallback jika fitur tidak dikenali (harusnya tidak terjadi dengan definisi NUMERIC_FEATURES/CATEGORICAL_FEATURES)
                    st.warning(f"Fitur '{feature}' tidak ditemukan di `processed_heart_disease_data.csv` atau tidak dikenali. Menggunakan input teks generik.")
                    user_input[feature] = st.text_input(f"**{feature.replace('_', ' ').title()}", value="", key=f"input_{feature}_fallback", help="Masukkan nilai untuk fitur ini.") 

        col_idx = (col_idx + 1) % num_cols 

    st.markdown("<br>", unsafe_allow_html=True) 
    input_df = pd.DataFrame([user_input])
    
    input_df = input_df.replace({None: np.nan})

    if input_df.isnull().values.any():
        st.warning("Harap lengkapi semua input sebelum melakukan prediksi.")
        predict_button = st.button("Lakukan Prediksi", disabled=True) 
    else:
        predict_button = st.button("Lakukan Prediksi", help="Klik untuk mendapatkan hasil prediksi penyakit jantung.")


    if predict_button:
        try:
            for feature in feature_names_for_model:
                if pd.notna(input_df.loc[0, feature]):
                    original_dtype = processed_data_for_features[feature].dtype
                    if 'int' in str(original_dtype):
                        input_df.loc[0, feature] = int(input_df.loc[0, feature])
                    elif 'float' in str(original_dtype):
                        input_df.loc[0, feature] = float(input_df.loc[0, feature])
            
            input_df = input_df[feature_names_for_model] 
            
            prediction_raw = model.predict(input_df)
            
            final_prediction = 1 if prediction_raw[0] > 0 else 0

            proba_available = False
            if hasattr(model, 'predict_proba'):
                prediction_proba_raw = model.predict_proba(input_df)
                proba_available = True

            st.subheader("Ringkasan Hasil:")
            
            if final_prediction == 0:
                st.success("Prediksi: **Tidak Terkena Penyakit Jantung**")
            else:
                st.error("Prediksi: **Terkena Penyakit Jantung**")
            
            if proba_available:
                st.write("**Probabilitas:**")
                
                proba_no_disease = 0.0
                if 0 in model.classes_: 
                    proba_no_disease = prediction_proba_raw[0][list(model.classes_).index(0)]
                
                proba_disease = 0.0
                for i, class_label in enumerate(model.classes_):
                    if class_label > 0:
                        proba_disease += prediction_proba_raw[0][i]

                st.write(f"  - **Tidak Terkena Penyakit Jantung:** {proba_no_disease*100:.2f}%")
                st.write(f"  - **Terkena Penyakit Jantung:** {proba_disease*100:.2f}%")


            st.markdown("""
            ---
            **Penting:**
            Prediksi ini dihasilkan oleh model machine learning dan BUKAN pengganti diagnosis medis profesional.
            Selalu konsultasikan dengan dokter atau profesional kesehatan yang berkualitas untuk setiap masalah kesehatan.
            """)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.write("Harap pastikan semua bidang input terisi dengan benar dan sesuai dengan tipe data yang diharapkan oleh model Anda.")
            st.write(f"Detail kesalahan: {e}")
            st.write(f"Fitur yang diharapkan oleh model (berdasarkan `feature_names_for_model`): {feature_names_for_model}")
            st.write(f"Fitur yang disediakan oleh input user: {input_df.columns.tolist()}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Prediksi Penyakit Jantung",
        layout="centered",
        initial_sidebar_state="auto",
    )

    main()