import streamlit as st
from streamlit_option_menu import option_menu

# Tạo menu điều hướng
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # Tiêu đề menu
        options=["Home", "About", "Contact"],  # Các tùy chọn trong menu
        icons=["house", "info-circle", "envelope"],  # Icon cho từng mục
        menu_icon="cast",  # Icon cho menu chính
        default_index=0,  # Mục mặc định được chọn
    )

# Hiển thị nội dung dựa trên lựa chọn
if selected == "Home":
    st.title("Welcome to Home Page!")
    st.write("This is the home page content.")
elif selected == "About":
    st.title("About Us")
    st.write("This is the about page content.")
elif selected == "Contact":
    st.title("Contact Us")
    st.write("This is the contact page content.")
