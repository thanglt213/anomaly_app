import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
import numpy as np
import plotly.express as px
from PIL import Image
from streamlit_option_menu import option_menu

# Hàm tải và xử lý ảnh
def display_resized_image(image_path, new_height_divider=2):
    image = Image.open(image_path)
    width, height = image.size
    resized_image = image.resize((width, height // new_height_divider))
    st.image(resized_image, use_column_width=True)

# Hàm mã hóa và chuẩn hóa dữ liệu
def preprocess_data(train_data, predict_data, numeric_cols):
    combined_data = pd.concat([train_data, predict_data], ignore_index=True)
    for col in numeric_cols:
        combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
    
    label_encoders = {}
    for col in combined_data.columns:
        if combined_data[col].dtype == 'object':
            le = LabelEncoder()
            combined_data[col] = le.fit_transform(combined_data[col].fillna('Unknown'))
            label_encoders[col] = le
            
    scaler = StandardScaler()
    combined_data[numeric_cols] = scaler.fit_transform(combined_data[numeric_cols])
    
    return combined_data, label_encoders

# Hàm huấn luyện IsolationForest
def train_isolation_forest(train_data, contamination_rate=0.05):
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
    model.fit(train_data.select_dtypes(include=[np.number]))
    return model

# Hàm hiển thị biểu đồ stacked bar
def plot_prediction_chart(data, group_by_col, title, ylabel, key):
    prediction_counts = data.groupby([group_by_col, 'Prediction']).size().reset_index(name='Count')
    prediction_counts = prediction_counts.sort_values(by='Count', ascending=False)
    
    custom_colors = ['#1f77b4', '#ff7f0e']
    
    fig = px.bar(
        prediction_counts, 
        x=group_by_col, 
        y='Count', 
        color='Prediction',
        title=title, 
        labels={group_by_col: ylabel}, 
        text_auto=True,
        color_discrete_sequence=custom_colors
    )

    fig.update_layout(
        xaxis_title="",  
        yaxis_title="",  
    )
    
    st.plotly_chart(fig, key=key)

# Hàm hiển thị biểu đồ tỷ lệ phần trăm
def plot_prediction_percent_chart(data, group_by_col, title, ylabel, key):
    prediction_counts = data.groupby(group_by_col)['Prediction'].value_counts(normalize=True).unstack().fillna(0)
    prediction_counts['Bất thường'] = prediction_counts.get('Bất thường', 0)
    prediction_counts = prediction_counts.reset_index()
    prediction_counts = prediction_counts.sort_values(by='Bất thường', ascending=False)

    fig = px.bar(prediction_counts, 
                 x=group_by_col, 
                 y='Bất thường',
                 title=title, 
                 labels={group_by_col: ylabel, 'Bất thường': 'Tỷ lệ phần trăm'}, 
                 text=prediction_counts['Bất thường'].map('{:.1%}'.format))
    
    fig.update_layout(
        xaxis_title="",  
        yaxis_title="",  
    )
    
    st.plotly_chart(fig, key=key)

# Tên file lưu trữ mô hình
MODEL_FILE = "kmeans_model.pkl"

# Hàm huấn luyện và lưu mô hình
def train_and_save_model(data, features, optimal_k=4):
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    # Huấn luyện mô hình KMeans
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    kmeans.fit(data[features])

    # Lưu mô hình và bộ chuẩn hóa vào file
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump((kmeans, scaler), f)

    st.success(f"Mô hình đã được huấn luyện và lưu vào {MODEL_FILE}.")

# Hàm tải mô hình đã lưu
def load_model():
    with open(MODEL_FILE, 'rb') as f:
        kmeans, scaler = pickle.load(f)
    st.success("Mô hình đã được tải thành công.")
    return kmeans, scaler

# Hàm để thực hiện dự đoán
def predict_with_model(kmeans, scaler, new_data, features):
    new_data[features] = scaler.transform(new_data[features])  # Chuẩn hóa dữ liệu mới
    new_data['cluster'] = kmeans.predict(new_data[features])
    new_data['distance_to_centroid'] = np.min(kmeans.transform(new_data[features]), axis=1)
    
    # Xác định các giao dịch bất thường
    threshold = np.percentile(new_data['distance_to_centroid'], 95)
    new_data['k_anomaly'] = new_data['distance_to_centroid'] > threshold
    return new_data

# Chức năng chính của mục 'Kế toán'
def ke_toan_option():
    st.title("Phân cụm KMeans và phát hiện bất thường")

    # Kiểm tra xem mô hình đã được lưu trước đó chưa
    if not os.path.exists(MODEL_FILE):
        st.info("Chưa có mô hình. Vui lòng tải dữ liệu để huấn luyện.")
        
        # Tải file dữ liệu train CSV
        uploaded_file = st.file_uploader("Tải file CSV để huấn luyện mô hình", type=['csv'])
        if uploaded_file is not None:
            # Đọc dữ liệu từ file thành DataFrame
            data = pd.read_csv(uploaded_file)
            
            # Hiển thị dữ liệu
            st.write("Xem trước dữ liệu huấn luyện:")
            st.dataframe(data.head())

            # Các đặc trưng cần huấn luyện
            features = ['credit_account', 'debit_account', 'so_tien_chi_tiet', 'id_loai_giao_dich']

            # Huấn luyện và lưu mô hình
            if st.button("Huấn luyện và lưu mô hình"):
                train_and_save_model(data, features)

    else:
        st.success("Mô hình đã tồn tại. Bạn có thể tải dữ liệu để dự đoán hoặc huấn luyện lại mô hình.")

        # Nút để huấn luyện lại mô hình
        if st.button("Huấn luyện lại mô hình"):
            st.info("Vui lòng tải file dữ liệu mới để huấn luyện lại mô hình.")
            retrain_file = st.file_uploader("Tải file CSV để huấn luyện lại mô hình", type=['csv'])
            
            if retrain_file is not None:
                data = pd.read_csv(retrain_file)
                st.write("Xem trước dữ liệu huấn luyện lại:")
                st.dataframe(data.head())

                # Các đặc trưng cần huấn luyện
                features = ['credit_account', 'debit_account', 'so_tien_chi_tiet', 'id_loai_giao_dich']

                # Huấn luyện và lưu mô hình mới
                if st.button("Huấn luyện lại và lưu mô hình"):
                    train_and_save_model(data, features)
        else:
            # Tải mô hình đã lưu
            kmeans, scaler = load_model()

            # Tải file dữ liệu mới để dự đoán
            new_file = st.file_uploader("Tải file CSV để dự đoán", type=['csv'])
            if new_file is not None:
                new_data = pd.read_csv(new_file)
                st.write("Xem trước dữ liệu dự đoán:")
                st.dataframe(new_data.head())

                # Các đặc trưng cần dự đoán
                features = ['credit_account', 'debit_account', 'so_tien_chi_tiet', 'id_loai_giao_dich']

                # Thực hiện dự đoán
                predicted_data = predict_with_model(kmeans, scaler, new_data, features)

                # Hiển thị kết quả dự đoán
                st.subheader("Kết quả dự đoán kèm các bất thường")
                st.dataframe(predicted_data.head())

                # Nút để lưu kết quả dự đoán ra file CSV
                if st.button("Lưu kết quả dự đoán ra CSV"):
                    pred_csv_data = predicted_data.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Tải CSV kết quả dự đoán", data=pred_csv_data, file_name='prediction_results.csv', mime='text/csv')

# Chức năng chính của mục 'Bảo hiểm sức khỏe'
def suc_khoe_option():
        model_file = 'isolation_forest_model.pkl'
        model_exists = os.path.exists(model_file)

        with st.expander("Tải dữ liệu huấn luyện và dự đoán", expanded=True):
            train_file = st.file_uploader("Chọn file CSV huấn luyện", type=["csv"], key='train')
            predict_file = st.file_uploader("Chọn file CSV dự đoán", type=["csv"], key='predict')

        if train_file and predict_file:
            try:
                train_data = pd.read_csv(train_file).dropna().astype(str)
                predict_data = pd.read_csv(predict_file).dropna().astype(str)

                if 'days_to_report' not in train_data.columns or 'requested_amount_per_day' not in train_data.columns:
                    st.error("Dữ liệu huấn luyện thiếu cột 'days_to_report' hoặc 'requested_amount_per_day'. Vui lòng kiểm tra lại file.")
                    st.stop()

                numeric_columns = ['days_to_report', 'requested_amount_per_day']
                combined_data, label_encoders = preprocess_data(train_data, predict_data, numeric_columns)
                
                num_train_rows = train_data.shape[0]
                train_encoded = combined_data.iloc[:num_train_rows]
                predict_encoded = combined_data.iloc[num_train_rows:]

                model = None

                if model_exists:
                    with st.expander("Huấn luyện và tải mô hình", expanded=True):
                        col0, col1, col2 = st.columns(3)
                        with col0:
                            st.info("File mô hình đã tồn tại.")
                        with col1:
                            if st.button("Tải mô hình"):
                                with st.spinner('Đang tải mô hình...'):
                                    model = joblib.load(model_file)
                                st.success("Mô hình đã được tải!")
                        with col2:
                            if st.button("Huấn luyện lại"):
                                with st.spinner('Đang huấn luyện mô hình...'):
                                    model = train_isolation_forest(train_encoded)
                                st.success("Mô hình đã được huấn luyện!")
                                joblib.dump(model, model_file)  
                else:
                    with st.spinner('Đang huấn luyện mô hình...'):
                        model = train_isolation_forest(train_encoded)
                    st.success("Mô hình đã được huấn luyện thành công!")
                    joblib.dump(model, model_file)  

                if model is not None:
                    with st.expander("Thực hiện dự đoán...", expanded=True):
                        with st.spinner('Đang thực hiện dự đoán...'):
                            predictions = model.predict(predict_encoded)
                        predict_data['Prediction'] = np.where(predictions == -1, 'Bất thường', 'Bình thường')

                        st.write(f"Số lượng bất thường: {sum(predict_data['Prediction'] == 'Bất thường')}/{len(predict_data)}")
                        st.dataframe(predict_data[['Prediction', 'branch', 'claim_no', 'distribution_channel', 'hospital']], use_container_width=True)

                        csv = predict_data.to_csv(index=False)
                        st.download_button("Tải xuống kết quả", csv, "predictions.csv", "text/csv")

                    with st.expander("Trực quan hóa kết quả...", expanded=True):
                        plot_prediction_chart(predict_data, 'distribution_channel', 'Số lượng bất thường theo kênh khai thác:', 'Kênh khai thác', key='key1')
                        plot_prediction_percent_chart(predict_data, 'distribution_channel', 'Tỷ lệ % bất thường theo kênh khai thác:', 'Kênh khai thác', key='key2')
                        plot_prediction_chart(predict_data, 'branch', 'Số lượng bất thường theo chi nhánh:', 'Chi nhánh', key='key3')
                        plot_prediction_percent_chart(predict_data, 'branch', 'Tỷ lệ % bất thường theo chi nhánh:', 'Chi nhánh', key='key4')
                        plot_prediction_chart(predict_data, 'hospital', 'Số lượng bất thường theo bệnh viện:', 'Bệnh viện', key='key5')
                        plot_prediction_percent_chart(predict_data, 'hospital', 'Tỷ lệ % bất thường theo bệnh viện:', 'Bệnh viện', key='key6')

            except Exception as e:
                st.error(f"Có lỗi xảy ra khi xử lý tệp: {e}")
    
# Main Streamlit app
def app():
    st.title("Một ứng dụng AI: Phát hiện bất thường trong hoạt động bảo hiểm ")
    #display_resized_image("ica.jpg")
    st.info("\nBất thường không có nghĩa là gian lận, nhưng gian lận là bất thường!\n", icon="ℹ️")
    
    # Tạo option-menu không có tiêu đề
    selected_option = option_menu(
        menu_title=None,  # Ẩn tiêu đề menu
        options=["Bảo hiểm sức khỏe", "Bảo hiểm xe", "Kế toán"],  # Các tùy chọn
        icons=["heart", "car", "book"],  # Icon cho từng mục
        default_index=0,  # Mục mặc định
        orientation="horizontal"  # Hiển thị menu ngang
    )

    if selected_option == 'Bảo hiểm sức khỏe':
        suc_khoe_option()
    elif selected_option == 'Bảo hiểm xe':
        st.write("Tính năng sẽ được cập nhật sau.")

    elif selected_option == 'Kế toán':
        #st.write("Tính năng sẽ được cập nhật sau.")
        ke_toan_option()

if __name__ == "__main__":
    app()


