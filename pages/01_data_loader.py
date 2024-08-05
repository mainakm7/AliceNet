import streamlit as st
import pandas as pd
import requests

st.header("Upload Gene Expression Data")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size:", uploaded_file.size)

    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post("http://localhost:8000/load/upload_data", files=files)

    if response.status_code == 200:
        st.success("File uploaded successfully.")
    else:
        st.error(f"Error uploading file: {response.text}")


st.divider()
st.header("List of uploaded files")

subdir = st.text_input("Enter subdirectory", value="raw")

def fetch_file_list(subdir: str):
    try:
        response = requests.get("http://localhost:8000/load/filenames", params={"subdir": subdir})
        response.raise_for_status()
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

st.divider()
st.header("Raw DataFrames:")

tab1, tab2 = st.tabs(["Expression data", "Event data"])

def load_exp_data(filename):
    try:
        response = requests.post("http://localhost:8000/load/load_expression", json={"filename": filename})
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data['data'], columns=data['columns'], index=data['index'])
    except requests.RequestException as e:
        st.error(f"Error loading expression data: {str(e)}")
        return pd.DataFrame()

with tab1:
    filenames = fetch_file_list(subdir)
    if filenames:
        exp_file = st.selectbox("Select expression file", filenames)
        if st.button("Load Expression Data"):
            with st.expander("Show expression dataframe"):
                if exp_file:
                    exp_df = load_exp_data(exp_file)
                    if not exp_df.empty:
                        st.write(exp_df)
                        st.session_state._temp_exp_dict = exp_df.to_dict(orient="split")
                    else:
                        st.write("No data available.")
    else:
        st.write("No files available to select.")

def load_event_data(filename):
    try:
        response = requests.post("http://localhost:8000/load/load_event", json={"filename": filename})
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data['data'], columns=data['columns'], index=data['index'])
    except requests.RequestException as e:
        st.error(f"Error loading event data: {str(e)}")
        return pd.DataFrame()

with tab2:
    filenames = fetch_file_list(subdir)
    if filenames:
        event_file = st.selectbox("Select Event file", filenames)
        if st.button("Load Event Data"):
            with st.expander("Show event dataframe"):
                if event_file:
                    event_df = load_event_data(event_file)
                    if not event_df.empty:
                        st.write(event_df)
                        st.session_state._temp_event_dict = event_df.to_dict(orient="split")
                    else:
                        st.write("No data available.")
    else:
        st.write("No files available to select.")

def sync_data(exp_df, event_df):
    try:
        response = requests.post("http://localhost:8000/load/sync_data", json={"sf_exp_df": exp_df, "sf_events_df": event_df})
        response.raise_for_status()
        data = response.json()
        exp_dict = data["exp_df"]
        event_dict = data["event_df"]
        exp_df = pd.DataFrame(exp_dict["data"], columns=exp_dict["columns"], index=exp_dict["index"])
        event_df = pd.DataFrame(event_dict["data"], columns=event_dict["columns"], index=event_dict["index"])
        return exp_df, event_df
    except requests.RequestException as e:
        st.error(f"Error syncing data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except KeyError as e:
        st.error(f"Error processing response data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return pd.DataFrame(), pd.DataFrame()

if st.checkbox("Sync the patients of expression and event DataFrames"):
    if '_temp_exp_dict' in st.session_state and '_temp_event_dict' in st.session_state:
        exp_df, event_df = sync_data(st.session_state._temp_exp_dict, st.session_state._temp_event_dict)
        
        # Create tabs after syncing
        tab1, tab2 = st.tabs(["Expression data", "Event data"])
        
        with tab1:
            with st.expander("Show synced expression dataframe"):
                st.write(exp_df)
                st.session_state.exp_dict = exp_df.to_dict(orient="split")
        
        with tab2:
            with st.expander("Show synced event dataframe"):
                st.write(event_df)
                st.session_state.event_dict = event_df.to_dict(orient="split")
    else:
        st.error("Expression or event data not available to sync.")

st.divider()

