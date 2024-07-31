import streamlit as st
import pandas as pd
import requests

st.header("Upload Gene Expression Data")

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
        
# List uploaded files section
st.header("List of uploaded files")

subdir = st.text_input("Enter subdirectory", value="raw")

def fetch_file_list(subdir: str):
    try:
        response = requests.get("http://localhost:8000/load/filenames", params={"subdir": subdir})
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching file list: {str(e)}")
        return []

if st.button("Refresh file list"):
    filenames = fetch_file_list(subdir)
    if filenames:
        st.write("Files:", filenames)
    else:
        st.write("No files found.")


st.header("Raw DataFrames:")

tab1, tab2 = st.tabs(["Expression data", "Event data"])

with tab1:

    # Selectbox for files
    filenames = fetch_file_list(subdir)
    if filenames:
        exp_file = st.selectbox("Select expression file", filenames)
    else:
        st.write("No files available to select.")

    def load_exp_data(filename):
        try:
            response = requests.post("http://localhost:8000/load/load_expression", json={"filename": filename})
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            # Convert response dictionary back to DataFrame
            df = pd.DataFrame(data['data'], columns=data['columns'], index=data['index'])
            return df
        except requests.RequestException as e:
            st.error(f"Error loading expression data: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error

    with st.expander("Show expression dataframe"):
        if exp_file:
            exp_df = load_exp_data(exp_file)
            if not exp_df.empty:
                st.write(exp_df)
            else:
                st.write("No data available.")

with tab2:

    if filenames:
        event_file = st.selectbox("Select Event file", filenames)
    else:
        st.write("No files available to select.")

    # Function to load expression data
    def load_event_data(filename):
        try:
            response = requests.post("http://localhost:8000/load/load_event", json={"filename": filename})
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            # Convert response dictionary back to DataFrame
            df = pd.DataFrame(data['data'], columns=data['columns'], index=data['index'])
            return df
        except requests.RequestException as e:
            st.error(f"Error loading event data: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error

    with st.expander("Show event dataframe"):
        if event_file:
            event_df = load_event_data(event_file)
            if not event_df.empty:
                st.write(event_df)
            else:
                st.write("No data available.")

st.header("Sync Patients for expression and event dataframes:")

sync = st.button("Sync", type="primary")

if sync:
    try:
        response = requests.get("http://localhost:8000/load/sync_data")
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        
        # Extract and convert expression dataframe
        exp_dict = data["exp_df"]
        exp_df = pd.DataFrame(
            exp_dict["data"], 
            columns=exp_dict["columns"], 
            index=exp_dict["index"]
        )
        
        # Extract and convert event dataframe
        event_dict = data["event_df"]
        event_df = pd.DataFrame(
            event_dict["data"], 
            columns=event_dict["columns"], 
            index=event_dict["index"]
        )
        
        with st.expander("Show synced expression dataframe"):
            st.write(exp_df)
        
        with st.expander("Show synced event dataframe"):
            st.write(event_df)
    
    except requests.RequestException as e:
        st.error(f"Error fetching synced data: {e}")
    except KeyError as e:
        st.error(f"Error processing response data: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")