import streamlit as st
from streamlit_option_menu import option_menu

def app():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")

    # Tạo option-menu không có tiêu đề
    selected_option = option_menu(
        menu_title=None,  # Ẩn tiêu đề menu
        options=["Bồi thường sức khỏe", "Kế toán", "Xe cơ giới"],  # Các tùy chọn
        icons=["box", "book", "gear"],  # Icon cho từng mục
        default_index=0,  # Mục mặc định
        orientation="horizontal"  # Hiển thị menu ngang
    )

    # Hiển thị nội dung dựa trên lựa chọn
    if selected_option == "Bồi thường sức khỏe":
        MedAlertAI.app()
    elif selected_option == "Kế toán":
        st.write("You selected Option 2!")
    elif selected_option == "Xe cơ giới":
        st.write("You selected Option 3!")

