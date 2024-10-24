import streamlit as st
from streamlit_option_menu import option_menu

def app():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")

    # Tạo option-menu đầu tiên
    option1 = option_menu(
        menu_title="First Menu",
        options=["Option 1", "Option 2"],
        icons=["box", "book"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"  # Menu nằm ngang
    )

    if option1 == "Option 1":
        st.write("You selected Option 1 from the First Menu!")
    elif option1 == "Option 2":
        st.write("You selected Option 2 from the First Menu!")

    # Tạo option-menu thứ hai
    option2 = option_menu(
        menu_title="Second Menu",
        options=["Option A", "Option B"],
        icons=["calendar", "camera"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"  # Menu nằm ngang
    )

    if option2 == "Option A":
        st.write("You selected Option A from the Second Menu!")
    elif option2 == "Option B":
        st.write("You selected Option B from the Second Menu!")

