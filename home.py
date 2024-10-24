import streamlit as st
from streamlit_option_menu import option_menu

def app():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")

    # Tạo option-menu không có tiêu đề
    selected_option = option_menu(
        menu_title=None,  # Ẩn tiêu đề menu
        options=["Option 1", "Option 2", "Option 3"],  # Các tùy chọn
        icons=["box", "book", "gear"],  # Icon cho từng mục
        default_index=0,  # Mục mặc định
        orientation="horizontal"  # Hiển thị menu ngang
    )

    # Hiển thị nội dung dựa trên lựa chọn
    if selected_option == "Option 1":
        st.write("You selected Option 1!")
    elif selected_option == "Option 2":
        st.write("You selected Option 2!")
    elif selected_option == "Option 3":
        st.write("You selected Option 3!")

