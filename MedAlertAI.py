import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import numpy as np
import plotly.express as px
from PIL import Image

# Hàm tải và xử lý ảnh
def display_resized_image(image_path, new_height_divider=2):
    image = Image.open(image_path)
    width, height = image.size
    resized_image = image.resize((width, height // new_height_divider))
    st.image(resized_image, use_column_width=True)

# Hàm mã hóa và chuẩn hóa dữ liệu
def preprocess_data(train_data, predict_data, numeric_cols):
    # Gộp dữ liệu huấn luyện và dự đoán
    combined_data = pd.concat([train_data, predict_data], ignore_index=True)

    # Chuyển các cột kiểu numeric và xử lý lỗi
    for col in numeric_cols:
        combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')

    # Mã hóa các cột phân loại
    label_encoders = {}
    for col in combined_data.columns:
        if combined_data[col].dtype == 'object':
            le = LabelEncoder()
            combined_data[col] = le.fit_transform(combined_data[col].fillna('Unknown'))
            label_encoders[col] = le
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    combined_data[numeric_cols] = scaler.fit_transform(combined_data[numeric_cols])
    
    return combined_data, label_encoders

# Hàm huấn luyện IsolationForest
def train_isolation_forest(train_data, contamination_rate=0.05):
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
    model.fit(train_data.select_dtypes(include=[np.number]))  # Chỉ sử dụng các cột số để huấn luyện
    return model

# Hàm hiển thị biểu đồ stacked bar
def plot_prediction_chart(data, group_by_col, title, ylabel, key):
    # Nhóm dữ liệu theo group_by_col và Prediction
    prediction_counts = data.groupby([group_by_col, 'Prediction']).size().reset_index(name='Count')
    
    # Sắp xếp dữ liệu
    prediction_counts = prediction_counts.sort_values(by='Count', ascending=False)
    
    # Đặt hệ màu tùy chỉnh
    custom_colors = ['#1f77b4', '#ff7f0e']
      
    # Tạo biểu đồ stacked bar
    fig = px.bar(
        prediction_counts, 
        x=group_by_col, 
        y='Count', 
        color='Prediction',  # Sử dụng cột 'Prediction' để tạo màu sắc cho từng nhóm
        title=title, 
        labels={group_by_col: ylabel}, 
        text_auto=True,
        color_discrete_sequence=custom_colors  # Sử dụng hệ màu tùy chỉnh
    )

    # Bỏ tiêu đề trục x và y
    fig.update_layout(
        xaxis_title="",  # Bỏ tiêu đề trục x
        yaxis_title="",  # Bỏ tiêu đề trục y
    )
    
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig, key=key)


# Hàm hiển thị biểu đồ tỷ lệ phần trăm
def plot_prediction_percent_chart(data, group_by_col, title, ylabel, key):
    # Nhóm dữ liệu theo group_by_col và tính tỷ lệ bất thường
    prediction_counts = data.groupby(group_by_col)['Prediction'].value_counts(normalize=True).unstack().fillna(0)
    
    # Chỉ giữ lại tỷ lệ phần trăm cho "Bất thường"
    prediction_counts['Bất thường'] = prediction_counts.get('Bất thường', 0)
    
    # Chuyển đổi thành DataFrame cho dễ xử lý
    prediction_counts = prediction_counts.reset_index()
    
    # Sắp xếp theo tỷ lệ phần trăm "Bất thường" giảm dần
    prediction_counts = prediction_counts.sort_values(by='Bất thường', ascending=False)

    # Tạo biểu đồ cột theo tỷ lệ phần trăm
    fig = px.bar(prediction_counts, 
                 x=group_by_col, 
                 y='Bất thường',
                 title=title, 
                 labels={group_by_col: ylabel, 'Bất thường': 'Tỷ lệ phần trăm'}, 
                 text=prediction_counts['Bất thường'].map('{:.1%}'.format))  # Định dạng nhãn phần trăm
    
    # Bỏ tiêu đề trục x và y
    fig.update_layout(
        xaxis_title="",  # Bỏ tiêu đề trục x
        yaxis_title="",  # Bỏ tiêu đề trục y
    )
    
    # Hiển thị biểu đồ trong Streamlit với key duy nhất
    st.plotly_chart(fig, key=key)


# Main Streamlit app
st.title("Phát hiện bất thường trong bồi thường bảo hiểm sức khỏe")
display_resized_image("ica.jpg")
st.info("Bất thường không có nghĩa là gian lận, nhưng gian lận là bất thường!", icon="ℹ️")

# Kiểm tra tệp mô hình
model_file = 'isolation_forest_model.pkl'
model_exists = os.path.exists(model_file)

# Tải dữ liệu huấn luyện và dự đoán
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

        numeric_columns = ['days_to_report', 'requested_amount_per_day']  # Cột cần chuẩn hóa
        combined_data, label_encoders = preprocess_data(train_data, predict_data, numeric_columns)
        
        # Tách dữ liệu huấn luyện và dự đoán
        num_train_rows = train_data.shape[0]
        train_encoded = combined_data.iloc[:num_train_rows]
        predict_encoded = combined_data.iloc[num_train_rows:]
        
        model = None  # Khởi tạo biến mô hình

        if model_exists:
            with st.expander("Huấn luyện và tải mô hình", expanded=True):
                with st.container():
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
                            joblib.dump(model, model_file)  # Lưu mô hình
        else:
            with st.spinner('Đang huấn luyện mô hình...'):
                model = train_isolation_forest(train_encoded)
            st.success("Mô hình đã được huấn luyện thành công!")
            joblib.dump(model, model_file)  # Lưu mô hình
        
        # Thực hiện dự đoán và hiển thị kết quả nếu mô hình đã được tải
        if model is not None:
            with st.expander("Thực hiện dự đoán...", expanded=True):
                with st.spinner('Đang thực hiện dự đoán...'):
                    predictions = model.predict(predict_encoded)
                predict_data['Prediction'] = np.where(predictions == -1, 'Bất thường', 'Bình thường')
                
                # Hiển thị kết quả dự đoán
                st.write(f"Số lượng bất thường: {sum(predict_data['Prediction'] == 'Bất thường')}/{len(predict_data)}")
                st.dataframe(predict_data[['Prediction', 'branch', 'claim_no', 'distribution_channel', 'hospital']], use_container_width=True)
    
                # Nút tải xuống kết quả
                csv = predict_data.to_csv(index=False)
                st.download_button("Tải xuống kết quả", csv, "predictions.csv", "text/csv")

            with st.expander("Trực quan hóa kết quả...", expanded=True):
                # Biểu đồ
                plot_prediction_chart(predict_data, 'distribution_channel', 'Số lượng bất thường theo kênh khai thác:', 'Kênh khai thác', key='key1')
                plot_prediction_percent_chart(predict_data, 'distribution_channel', 'Tỷ lệ % bất thường theo kênh khai thác:', 'Kênh khai thác', key='key2')
              
                plot_prediction_chart(predict_data, 'branch', 'Số lượng bất thường theo chi nhánh:', 'Chi nhánh', key='key3')
                plot_prediction_percent_chart(predict_data, 'branch', 'Tỷ lệ % bất thường theo chi nhánh:', 'Chi nhánh', key='key4')
                
                plot_prediction_chart(predict_data, 'hospital', 'Số lượng bất thường theo bệnh viện:', 'Bệnh viện', key='key5')
                plot_prediction_percent_chart(predict_data, 'hospital', 'Tỷ lệ % bất thường theo bệnh viện:', 'Bệnh viện', key='key6')

    except Exception as e:
        st.error(f"Có lỗi xảy ra khi xử lý tệp: {e}")
