import streamlit as st
import requests

st.title("Upload Gene Expression Data")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Show file details
    st.write("Filename:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size:", uploaded_file.size)

    # Send the file to FastAPI backend
    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post("http://localhost:8000/load/upload_data", files=files)

    if response.status_code == 200:
        st.success("File uploaded successfully.")
    else:
        st.error(f"Error uploading file: {response.text}")