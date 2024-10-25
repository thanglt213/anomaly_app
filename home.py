import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

# Constants
KMEANS_MODEL_FILE = "kmeans_model.pkl"
ISOLATION_FOREST_MODEL_FILE = 'isolation_forest_model.pkl'
KMEANS_NUMERIC_FEATURES = ['credit_account', 'debit_account', 'so_tien_chi_tiet', 'id_loai_giao_dich']
ISOLATION_NUMERIC_FEATURES = ['days_to_report', 'requested_amount_per_day']

# Image Handling Functions
def display_resized_image(image_path, new_height_divider=2):
    image = Image.open(image_path)
    width, height = image.size
    resized_image = image.resize((width, height // new_height_divider))
    st.image(resized_image, use_column_width=True)

# Data Preprocessing Functions
def preprocess_kmeans_data(train_data, predict_data, numeric_cols):
    combined_data = pd.concat([train_data, predict_data], ignore_index=True)
    combined_data[numeric_cols] = combined_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    label_encoders = {}
    for col in combined_data.columns:
        if combined_data[col].dtype == 'object':
            le = LabelEncoder()
            combined_data[col] = le.fit_transform(combined_data[col].fillna('Unknown'))
            label_encoders[col] = le
            
    scaler = StandardScaler()
    combined_data[numeric_cols] = scaler.fit_transform(combined_data[numeric_cols])
    
    return combined_data, label_encoders

def preprocess_isolation_forest_data(train_data, predict_data, numeric_cols):
    combined_data = pd.concat([train_data, predict_data], ignore_index=True)
    combined_data[numeric_cols] = combined_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    label_encoders = {}
    for col in combined_data.columns:
        if combined_data[col].dtype == 'object':
            le = LabelEncoder()
            combined_data[col] = le.fit_transform(combined_data[col].fillna('Unknown'))
            label_encoders[col] = le
            
    scaler = StandardScaler()
    combined_data[numeric_cols] = scaler.fit_transform(combined_data[numeric_cols])
    
    return combined_data, label_encoders

# Model Training Functions
def train_isolation_forest_model(train_data, contamination_rate=0.05):
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
    model.fit(train_data.select_dtypes(include=[np.number]))
    return model

def train_and_save_kmeans_model(data, features, optimal_k=4):
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    kmeans.fit(data[features])

    with open(KMEANS_MODEL_FILE, 'wb') as f:
        pickle.dump((kmeans, scaler), f)

    st.success(f"Mô hình KMeans đã được huấn luyện và lưu vào {KMEANS_MODEL_FILE}.")

# Model Loading Functions
def load_kmeans_model():
    with open(KMEANS_MODEL_FILE, 'rb') as f:
        kmeans, scaler = pickle.load(f)
    st.success("Mô hình KMeans đã được tải thành công.")
    return kmeans, scaler

def load_isolation_forest_model():
    model = joblib.load(ISOLATION_FOREST_MODEL_FILE)
    st.success("Mô hình Isolation Forest đã được tải thành công.")
    return model

# Prediction Functions
def predict_with_kmeans_model(kmeans, scaler, new_data, features):
    new_data[features] = scaler.transform(new_data[features])
    new_data['cluster'] = kmeans.predict(new_data[features])
    new_data['distance_to_centroid'] = np.min(kmeans.transform(new_data[features]), axis=1)
    
    threshold = np.percentile(new_data['distance_to_centroid'], 95)
    new_data['k_anomaly'] = new_data['distance_to_centroid'] > threshold
    return new_data

def predict_with_isolation_forest_model(model, predict_encoded):
    predictions = model.predict(predict_encoded)
    return predictions

# Chart Plotting Functions
def plot_prediction_chart(data, group_by_col, title, ylabel, key):
    prediction_counts = data.groupby([group_by_col, 'Prediction']).size().reset_index(name='Count')
    prediction_counts = prediction_counts.sort_values(by='Count', ascending=False)
    
    fig = px.bar(prediction_counts, x=group_by_col, y='Count', color='Prediction', title=title, text_auto=True,
                 color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, key=key)

def plot_prediction_percent_chart(data, group_by_col, title, ylabel, key):
    prediction_counts = data.groupby(group_by_col)['Prediction'].value_counts(normalize=True).unstack().fillna(0)
    prediction_counts['Bất thường'] = prediction_counts.get('Bất thường', 0)
    prediction_counts = prediction_counts.reset_index()
    prediction_counts = prediction_counts.sort_values(by='Bất thường', ascending=False)

    fig = px.bar(prediction_counts, x=group_by_col, y='Bất thường', title=title, 
                 labels={group_by_col: ylabel, 'Bất thường': 'Tỷ lệ phần trăm'}, 
                 text=prediction_counts['Bất thường'].map('{:.1%}'.format))
    
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, key=key)

# Streamlit Pages
def ke_toan_option():
    if not os.path.exists(KMEANS_MODEL_FILE):
        st.info("Chưa có mô hình. Vui lòng tải dữ liệu để huấn luyện.")
        uploaded_file = st.file_uploader("Tải file CSV để huấn luyện mô hình", type=['csv'])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data.head())
            if st.button("Huấn luyện và lưu mô hình KMeans"):
                train_and_save_kmeans_model(data, KMEANS_NUMERIC_FEATURES)
    else:
        st.success("Mô hình KMeans đã tồn tại.")
        if st.button("Huấn luyện lại mô hình KMeans"):
            retrain_file = st.file_uploader("Tải file CSV để huấn luyện lại mô hình", type=['csv'])
            if retrain_file:
                data = pd.read_csv(retrain_file)
                st.dataframe(data.head())
                if st.button("Huấn luyện lại và lưu mô hình KMeans"):
                    train_and_save_kmeans_model(data, KMEANS_NUMERIC_FEATURES)
        else:
            kmeans, scaler = load_kmeans_model()
            new_file = st.file_uploader("Tải file CSV để dự đoán với mô hình KMeans", type=['csv'])
            if new_file:
                new_data = pd.read_csv(new_file)
                st.dataframe(new_data.head())
                predicted_data = predict_with_kmeans_model(kmeans, scaler, new_data, KMEANS_NUMERIC_FEATURES)
                st.dataframe(predicted_data.head())
                if st.button("Lưu kết quả dự đoán KMeans ra CSV"):
                    st.download_button("Tải CSV kết quả dự đoán", 
                                       data=predicted_data.to_csv(index=False).encode('utf-8'), 
                                       file_name='kmeans_prediction_results.csv', 
                                       mime='text/csv')

def suc_khoe_option():
    with st.expander("Tải dữ liệu huấn luyện và dự đoán", expanded=True):
        train_file = st.file_uploader("Chọn file CSV huấn luyện Isolation Forest", type=["csv"], key='train_isolation_forest')
        predict_file = st.file_uploader("Chọn file CSV dự đoán Isolation Forest", type=["csv"], key='predict_isolation_forest')

    if train_file and predict_file:
        train_data = pd.read_csv(train_file).dropna().astype(str)
        predict_data = pd.read_csv(predict_file).dropna().astype(str)

        if 'days_to_report' not in train_data.columns or 'requested_amount_per_day' not in train_data.columns:
            st.error("Dữ liệu huấn luyện Isolation Forest thiếu cột 'days_to_report' hoặc 'requested_amount_per_day'.")
            return

        combined_data, label_encoders = preprocess_isolation_forest_data(train_data, predict_data, ISOLATION_NUMERIC_FEATURES)
        train_encoded = combined_data.iloc[:len(train_data)]
        predict_encoded = combined_data.iloc[len(train_data):]

        if os.path.exists(ISOLATION_FOREST_MODEL_FILE):
            st.info("Mô hình Isolation Forest đã tồn tại. Dùng để dự đoán.")
            model = load_isolation_forest_model()
        else:
            if st.button("Huấn luyện mô hình Isolation Forest"):
                model = train_isolation_forest_model(train_encoded)
                joblib.dump(model, ISOLATION_FOREST_MODEL_FILE)
                st.success(f"Mô hình Isolation Forest đã được lưu vào {ISOLATION_FOREST_MODEL_FILE}.")

        predictions = predict_with_isolation_forest_model(model, predict_encoded)
        predict_data['Prediction'] = np.where(predictions == -1, 'Bất thường', 'Bình thường')
        st.dataframe(predict_data)

        if st.button("Lưu kết quả dự đoán ra CSV"):
            st.download_button("Tải CSV kết quả dự đoán", 
                               data=predict_data.to_csv(index=False).encode('utf-8'), 
                               file_name='isolation_forest_predictions.csv', 
                               mime='text/csv')

# Main Application
def app():
    selected_option = option_menu(menu_title=None, options=['Kế toán', 'Sức khoẻ'], 
                                  icons=['currency-exchange', 'activity'], menu_icon="cast", 
                                  default_index=0, orientation="horizontal")

    if selected_option == 'Kế toán':
        ke_toan_option()
    elif selected_option == 'Sức khoẻ':
        suc_khoe_option()

if __name__ == "__main__":
    main()
