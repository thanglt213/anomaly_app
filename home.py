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
KMEANS_NUMERIC_FEATURES = ['ma_don_vi','credit_account', 'debit_account', 'so_tien_chi_tiet', 'id_loai_giao_dich']
ISOLATION_NUMERIC_FEATURES = ['days_to_report', 'requested_amount_per_day']


# Hàm xử lý giá trị NaN và inf
def handle_missing_and_inf(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace([np.inf, -np.inf], np.nan)  # Thay thế Inf bằng NaN
    data = data.dropna()  # Loại bỏ các dòng chứa NaN
    return data

# Hàm chuyển đổi, mã hóa và chuẩn hóa dữ liệu
def transform_and_scale_data(data: pd.DataFrame, numeric_features: list) -> np.ndarray:
    # Chuyển tất cả dữ liệu về kiểu str
    data = data.astype(str)

    # Chuyển các cột số sang kiểu Numeric
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Loại bỏ các dòng có giá trị NaN trong các cột số (nếu có)
    data = data.dropna(subset=numeric_features)

    # Mã hóa các cột không phải số bằng Label Encoding
    label_encoder = LabelEncoder()
    non_numeric_features = [col for col in data.columns if col not in numeric_features]
    for col in non_numeric_features:
        data[col] = label_encoder.fit_transform(data[col])

    # Chuẩn hóa dữ liệu bằng StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled

# Huấn luyện mô hình KMeans.
def train_kmeans_model(data: pd.DataFrame, numeric_features: list):
    # Bước 1: Xử lý NaN và Inf
    st.write("Số lượng dòng trước khi xử lý NaN và Inf:", data.shape[0])
    data_cleaned = handle_missing_and_inf(data)
    st.write("Số lượng dòng sau khi xử lý NaN và Inf:", data_cleaned.shape[0])

    if data_cleaned.empty:
        st.error("Dữ liệu sau khi xử lý không còn dòng nào! Vui lòng kiểm tra dữ liệu.")
        return None, None

    # Bước 2: Chuyển đổi, mã hóa và chuẩn hóa dữ liệu
    st.write("Bắt đầu chuyển đổi và chuẩn hóa dữ liệu...")
    data_preprocessed = transform_and_scale_data(data_cleaned, numeric_features)
    st.write("Dữ liệu sau khi chuyển đổi và chuẩn hóa có dạng:", data_preprocessed.shape)

    # Bước 3: Huấn luyện mô hình KMeans
    st.write("Bắt đầu huấn luyện mô hình KMeans...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data_preprocessed)

    return kmeans, data_preprocessed

# Dự đoán cụm và khoảng cách tới tâm cụm cho dữ liệu mới
def predict_with_kmeans_model(kmeans, scaler, new_data: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """
    Dự đoán với mô hình KMeans.
    - Xử lý dữ liệu mới.
    - Chuẩn hóa và gán cụm vào dữ liệu.
    """
    if kmeans is None or scaler is None:
        raise ValueError("Mô hình KMeans hoặc scaler chưa được khởi tạo.")

    # Xử lý NaN và Inf
    data_cleaned = handle_missing_and_inf(new_data)

    # Lọc các cột số
    if not all(feature in data_cleaned.columns for feature in numeric_features):
        raise ValueError("Thiếu các đặc trưng số trong dữ liệu mới.")

    data_numeric = data_cleaned[numeric_features]

    # Chuẩn hóa dữ liệu
    data_scaled = scaler.transform(data_numeric)

    # Dự đoán cụm và tính khoảng cách tới tâm cụm
    data_cleaned['cluster'] = kmeans.predict(data_scaled)
    data_cleaned['distance_to_centroid'] = np.min(kmeans.transform(data_scaled), axis=1)

    return data_cleaned



# Hàm tiền xử lý dữ liệu Isolation forest
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

# Hàm training isolation forest
def train_isolation_forest_model(train_data, contamination_rate=0.05):
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
    model.fit(train_data.select_dtypes(include=[np.number]))
    return model


def load_isolation_forest_model():
    model = joblib.load(ISOLATION_FOREST_MODEL_FILE)
    st.success("Mô hình đã được tải thành công.")
    return model
 
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
# Module Kế toán
def ke_toan_option():
    """
    Tùy chọn phân tích dữ liệu kế toán bằng KMeans.
    """
    # Khởi tạo session_state nếu chưa tồn tại
    if 'kt_kmeans_model' not in st.session_state:
        st.session_state['kt_kmeans_model'] = None
    if 'kt_predicted_data' not in st.session_state:
        st.session_state['kt_predicted_data'] = None
    if 'anomaly_percentile' not in st.session_state:
        st.session_state['anomaly_percentile'] = 3.0  # Default anomaly percentile

    # Tải file dữ liệu huấn luyện
    kt_train_file = st.file_uploader("Tải file CSV dữ liệu huấn luyện để xây dựng mô hình", type=['csv'])
    if kt_train_file is not None:
        # Đọc dữ liệu huấn luyện
        kt_train_data = pd.read_csv(kt_train_file)
        
        # Tiến hành huấn luyện mô hình KMeans ngay khi tải dữ liệu mới
        st.write("Đang xử lý và huấn luyện lại mô hình...")
        kt_kmeans, processed_data = train_kmeans_model(kt_train_data, KMEANS_NUMERIC_FEATURES)
        if kt_kmeans is None:
            st.error("Không thể huấn luyện mô hình. Vui lòng kiểm tra dữ liệu!")
            return
        
        # Lưu mô hình và dữ liệu đã xử lý vào session_state
        st.session_state['kt_kmeans_model'] = kt_kmeans
        st.session_state['kt_predicted_data'] = predict_with_kmeans_model(
            st.session_state['kt_kmeans_model'],
            st.session_state['kt_scaler'],
            kt_train_data,
            KMEANS_NUMERIC_FEATURES
        )

        if st.session_state['kt_predicted_data'] is not None:
            st.success("Mô hình đã được huấn luyện và dữ liệu đã được dự đoán.")
        else:
            st.error("Không thể dự đoán dữ liệu! Vui lòng kiểm tra đầu vào.")

    # Xử lý và hiển thị dữ liệu dự đoán
    if st.session_state['kt_predicted_data'] is not None:
        # Chọn tỷ lệ bất thường từ 0% đến 10%
        anomaly_percentile = st.slider(
            label="Chọn tỷ lệ bất thường (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=st.session_state['anomaly_percentile'],  # Giá trị mặc định
            step=0.5,  # Bước nhảy
            format="%.1f%%"  # Hiển thị giá trị theo phần trăm
        )
        st.session_state['anomaly_percentile'] = anomaly_percentile
        
        # Xác định ngưỡng bất thường
        kt_predicted_data = st.session_state['kt_predicted_data']
        threshold = np.percentile(kt_predicted_data['distance_to_centroid'], 100 - anomaly_percentile)
        kt_predicted_data['k_anomaly'] = kt_predicted_data['distance_to_centroid'] > threshold    

        # Hiển thị kết quả dự đoán
        st.write("Dữ liệu dự đoán hiện tại:")
        st.dataframe(kt_predicted_data.head())
        
        # Tổng hợp kết quả theo đơn vị
        result = (
            kt_predicted_data[kt_predicted_data['k_anomaly'] == True]
            .groupby(['ten_don_vi', 'debit_account_name'])
            .agg({'so_chung_tu': 'count', 'so_tien_chi_tiet': 'sum'})
            .reset_index()
        )
        
        result['total_count'] = result.groupby('ten_don_vi')['so_chung_tu'].transform('sum')
        result = result.sort_values(by='total_count', ascending=False).drop(columns='total_count')
        
        st.write("\nTổng hợp kết quả bộ chứng từ bất thường theo đơn vị:")
        st.dataframe(result, use_container_width=True)

        # Download kết quả dự đoán
        st.download_button(
            "Tải CSV kết quả dự đoán", 
            data=kt_predicted_data.to_csv(index=False).encode('utf-8'), 
            file_name='kmeans_prediction_results.csv', 
            mime='text/csv'
        )


# Modul bảo hiểm sức khỏe        
def suc_khoe_option():
    # Khởi tạo session nếu chưa tồn tại
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None
    if 'predict_data' not in st.session_state:
        st.session_state.predict_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    # Load file dữ liệu huấn luyện và dự đoán
    with st.expander("Tải dữ liệu huấn luyện và dự đoán", expanded=False):
        train_file = st.file_uploader("Chọn file CSV huấn luyện", type=["csv"])
        predict_file = st.file_uploader("Chọn file CSV dự đoán", type=["csv"])

        # Nếu có thay đổi dữ liệu huấn luyện hoặc dữ liệu dự đoán, tính lại `combined_data`
        if train_file:
            st.session_state.train_data = pd.read_csv(train_file).dropna().astype(str)
            st.session_state.combined_data = None  # Đặt lại để tính toán lại khi có thay đổi dữ liệu

        if predict_file:
            st.session_state.predict_data = pd.read_csv(predict_file).dropna().astype(str)
            st.session_state.combined_data = None  # Đặt lại để tính toán lại khi có thay đổi dữ liệu

        # Nếu cả dữ liệu huấn luyện và dữ liệu dự đoán đều có mặt, tiến hành xử lý
        if st.session_state.train_data is not None and st.session_state.predict_data is not None:
            train_data = st.session_state.train_data
            predict_data = st.session_state.predict_data

            # Chỉ tính toán lại `combined_data` nếu chưa có hoặc dữ liệu đã thay đổi
            if st.session_state.combined_data is None:
                combined_data, label_encoders = preprocess_isolation_forest_data(train_data, predict_data, ISOLATION_NUMERIC_FEATURES)
                st.session_state.combined_data = combined_data
                st.session_state.label_encoders = label_encoders

            combined_data = st.session_state.combined_data
            train_encoded = combined_data.iloc[:len(train_data)]
            predict_encoded = combined_data.iloc[len(train_data):]

            # Tải/Huấn luyện mô hình
            if st.session_state.model is None:
                if os.path.exists(ISOLATION_FOREST_MODEL_FILE):
                    st.session_state.model = load_isolation_forest_model()
                elif st.button("Huấn luyện mô hình"):
                    st.session_state.model = train_isolation_forest_model(train_encoded)
                    joblib.dump(st.session_state.model, ISOLATION_FOREST_MODEL_FILE)

            # Dự đoán
            if st.session_state.model:
                predictions = predict_with_isolation_forest_model(st.session_state.model, predict_encoded)
                
                # Kiểm tra chiều dài của predictions và predict_data trước khi gán
                if len(predictions) == len(predict_data):
                    predict_data['Prediction'] = np.where(predictions == -1, 'Bất thường', 'Bình thường')
                    st.session_state.predictions = predict_data

                    # Hiển thị kết quả
                    st.write(f"Số lượng bất thường: {sum(predict_data['Prediction'] == 'Bất thường')}/{len(predict_data)}")
                    st.dataframe(predict_data[['Prediction', 'branch', 'claim_no', 'distribution_channel', 'hospital']], use_container_width=True)
                else:
                    st.error("Chiều dài của dữ liệu dự đoán không khớp với chiều dài của dữ liệu đầu vào. Vui lòng kiểm tra lại dữ liệu đầu vào.")

                # Download kết quả dự đoán
                if st.button("Lưu kết quả dự đoán ra CSV"):
                    st.download_button("Tải CSV kết quả dự đoán", 
                                       data=predict_data.to_csv(index=False).encode('utf-8'), 
                                       file_name='isolation_forest_predictions.csv', 
                                       mime='text/csv')

    # Phần Trực quan hóa kết quả
    with st.expander("Trực quan hóa kết quả...", expanded=True):
        # Lấy predict_data
        predict_data = st.session_state.predict_data if 'predict_data' in st.session_state else None
        
        # Nếu dữ liệu dùng để dự đoán đã tồn tại
        if predict_data is not None:
            # Initialize charts data if not already done
            if 'charts_data' not in st.session_state:
                st.session_state.charts_data = {
                    'distribution_channel': ('distribution_channel', 'Số lượng bất thường theo kênh khai thác:', 'Kênh khai thác'),
                    'distribution_channel_percent': ('distribution_channel', 'Tỷ lệ % bất thường theo kênh khai thác:', 'Kênh khai thác'),
                    'branch': ('branch', 'Số lượng bất thường theo chi nhánh:', 'Chi nhánh'),
                    'branch_percent': ('branch', 'Tỷ lệ % bất thường theo chi nhánh:', 'Chi nhánh'),
                    'hospital': ('hospital', 'Số lượng bất thường theo bệnh viện:', 'Bệnh viện'),
                    'hospital_percent': ('hospital', 'Tỷ lệ % bất thường theo bệnh viện:', 'Bệnh viện')
                }
    
            # Plot the charts from stored session state data
            for chart_key, chart_info in st.session_state.charts_data.items():
                if 'percent' in chart_key:
                    plot_prediction_percent_chart(predict_data, *chart_info, key=chart_key)
                else:
                    plot_prediction_chart(predict_data, *chart_info, key=chart_key)

# Main Application
def app():
    selected_option = option_menu(menu_title=None, options=['Sức khoẻ','Xe cơ gới','Kế toán'], 
                                  icons=['activity','car-front-fill','currency-exchange'], menu_icon="cast", 
                                  default_index=0, orientation="horizontal")

    if selected_option == 'Kế toán':
        ke_toan_option()
    elif selected_option == 'Sức khoẻ':
        suc_khoe_option()

if __name__ == "__main__":
    app()
