import streamlit as st
from streamlit_option_menu import option_menu
import home
import about
import contact

# Tạo menu điều hướng
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "About"],
        icons=["house", "info-circle", "envelope"],
        menu_icon="cast",
        default_index=0,
    )

# Gọi app tương ứng theo lựa chọn của người dùng
if selected == "Home":
    home.app()
elif selected == "About":
    about.app()


