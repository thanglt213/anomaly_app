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
        icons=["house", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

st.title("Một ứng dụng AI: Phát hiện bất thường trong hoạt động bảo hiểm")
st.info("\nBất thường không có nghĩa là gian lận, nhưng gian lận là bất thường!\n", icon="ℹ️")
    
# Gọi app tương ứng theo lựa chọn của người dùng
if selected == "Home":
    home.app()
elif selected == "About":
    about.app()


