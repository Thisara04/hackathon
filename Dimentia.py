# filename: app.py
import streamlit as st

st.title("Simple Streamlit Demo App")
st.write("This app calculates the square and cube of a number you enter.")

# User input
num = st.number_input("Enter a number:", value=1, step=1)

# Calculations
square = num ** 2
cube = num ** 3

# Display results
st.write(f"The square of {num} is {square}")
st.write(f"The cube of {num} is {cube}")

# Add a simple chart
st.write("### Visualization")
st.bar_chart({"Square": [square], "Cube": [cube]})
