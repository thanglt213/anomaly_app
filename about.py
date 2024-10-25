import streamlit as st

def app():
  # Nội dung
  st.write("""
  Gian lận trong bảo hiểm là một vấn đề nghiêm trọng ảnh hưởng đến cả các công ty bảo hiểm và người tiêu dùng. 
  Theo một báo cáo của tổ chức Coalition Against Insurance Fraud, khoảng 10% tất cả các yêu cầu bồi thường bảo hiểm 
  ở Mỹ có thể được coi là gian lận, dẫn đến thiệt hại lên tới 80 tỷ USD mỗi năm. Tình trạng này không chỉ làm tăng chi phí 
  cho các công ty bảo hiểm mà còn dẫn đến việc tăng phí bảo hiểm cho người tiêu dùng hợp pháp.
  
  Để giải quyết vấn đề này, nhiều công ty bảo hiểm đã áp dụng các giải pháp công nghệ tiên tiến như phân tích dữ liệu lớn (big data analytics) 
  và trí tuệ nhân tạo (AI) để phát hiện các mẫu hành vi gian lận. Việc triển khai hệ thống cảnh báo sớm và xác minh thông tin trực tuyến 
  cũng giúp giảm thiểu rủi ro gian lận. Ứng dụng máy học trong việc phát hiện gian lận bảo hiểm đang ngày càng trở nên phổ biến và hiệu quả. 
  Bằng cách sử dụng các thuật toán máy học, các công ty bảo hiểm có thể phân tích và xử lý một lượng lớn dữ liệu 
  để phát hiện các mẫu hành vi bất thường có thể chỉ ra gian lận.
  """)
  
  st.subheader("Các phương pháp và ứng dụng chính:")
  st.write("""
  1. **Phân tích dữ liệu lớn**: Máy học cho phép phân tích dữ liệu từ nhiều nguồn khác nhau, 
  bao gồm hồ sơ yêu cầu bồi thường, thông tin khách hàng và lịch sử giao dịch. Qua đó, 
  các mô hình máy học có thể tìm ra các mối liên hệ mà con người có thể bỏ lỡ.
  
  2. **Mô hình phân loại**: Các thuật toán như Logistic Regression, Decision Trees, 
  hoặc Random Forest có thể được sử dụng để phân loại các yêu cầu bồi thường thành 
  "hợp lệ" hoặc "nghi ngờ". Điều này giúp giảm thiểu số lượng yêu cầu cần xem xét thủ công.
  
  3. **Phát hiện bất thường**: Kỹ thuật như KMeans Clustering hoặc Isolation Forest 
  giúp xác định các yêu cầu bồi thường có đặc điểm khác biệt so với các trường hợp thông thường. 
  Những yêu cầu này có thể là dấu hiệu của hành vi gian lận.
  
  4. **Phân tích mạng lưới**: Sử dụng graph analytics để phát hiện các mối liên hệ giữa 
  các cá nhân và tổ chức. Việc phát hiện các mạng lưới gian lận, nơi mà nhiều yêu cầu bồi thường 
  được thực hiện bởi những người có liên hệ với nhau, có thể giúp ngăn chặn các hoạt động gian lận 
  quy mô lớn.
  
  5. **Học sâu (Deep Learning)**: Các mạng nơ-ron sâu có thể xử lý các dữ liệu phức tạp hơn, 
  chẳng hạn như hình ảnh từ hiện trường tai nạn hoặc các tài liệu chứng từ, để xác định tính hợp lệ 
  của yêu cầu.
  """)
  
  st.subheader("Lợi ích:")
  st.write("""
  - **Tăng độ chính xác**: Máy học cải thiện khả năng phát hiện gian lận, 
  giảm thiểu số lượng các yêu cầu bồi thường hợp lệ bị từ chối sai.
  
  - **Tiết kiệm thời gian**: Các hệ thống tự động giúp rút ngắn thời gian xử lý yêu cầu 
  và cho phép nhân viên tập trung vào các trường hợp nghi ngờ.
  
  - **Tối ưu hóa chi phí**: Giảm thiểu thiệt hại do gian lận có thể giúp các công ty bảo hiểm 
  giảm chi phí và giữ giá bảo hiểm ổn định cho khách hàng hợp pháp.
  """)
  
  st.write("""
  Nhờ vào sự tiến bộ của công nghệ và khả năng phân tích mạnh mẽ, máy học đang trở thành 
  một công cụ quan trọng trong cuộc chiến chống lại gian lận bảo hiểm, mang lại lợi ích 
  cho cả công ty và khách hàng.
  """)
