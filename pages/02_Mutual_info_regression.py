import streamlit as st
import pandas as pd
import requests

# Ensure DataFrames are available in session_state
if 'exp_dict' not in st.session_state or 'event_dict' not in st.session_state:
    st.error("Expression and event data not found. Please load them first.")
else:
    if '_temp_mi_melted_dict' not in st.session_state:
        st.session_state._temp_mi_melted_dict = None
    if '_temp_mi_dict' not in st.session_state:
        st.session_state._temp_mi_dict = None
        
    exp_df = st.session_state.exp_dict
    event_df = st.session_state.event_dict

    # Compute MI Section
    st.header("Compute MI:")
    mi_btn = st.button("Compute MI", type="primary")

    if mi_btn:
        with st.spinner("Computing MI..."):
            try:
                response = requests.post("http://localhost:8000/mi/compute_mi", json={"sf_exp_df": exp_df, "sf_events_df": event_df})
                response.raise_for_status()
                data = response.json()

                mi_dict = data["raw_mi_data"]
                mi_df = pd.DataFrame(mi_dict["data"], columns=mi_dict["columns"], index=mi_dict["index"])

                # Write mi_df to session_state
                st.session_state._temp_mi_dict = mi_dict

                st.write("Computed MI Data:")
                st.write(mi_df)
                
            except requests.RequestException as e:
                st.error(f"Error fetching MI data: {e}")

st.divider()
# Subdirectory Input
subdir = st.text_input("Enter subdirectory", value="MI")

def fetch_file_list(subdir: str):
    try:
        response = requests.get("http://localhost:8000/load/filenames", params={"subdir": subdir})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching file list: {str(e)}")
        return []

# Refresh File List Button
if st.button("Refresh file list"):
    filenames = fetch_file_list(subdir)
    if filenames:
        st.write("Files:", filenames)
    else:
        st.write("No files found.")

st.divider()
# Load MI Data Section
st.header("Load raw MI value from file")

def load_mi_data(filename):
    try:
        response = requests.post("http://localhost:8000/load/raw_mi", json={"filename": filename})
        response.raise_for_status()
        mi_dict = response.json()
        return mi_dict
    except requests.RequestException as e:
        st.error(f"Error loading MI data: {str(e)}")
        return {}

def melt_raw_mi(mi_dict):
    try:
        response = requests.post("http://localhost:8000/mi/melt_mi", json={"mi_raw_data": mi_dict})
        response.raise_for_status()
        mi_melted_dict = response.json()

        mi_melted_df = pd.DataFrame(mi_melted_dict["data"], columns=mi_melted_dict["columns"], index=mi_melted_dict["index"])
        if not mi_melted_df.empty:
            st.success("MI data melted successfully.")
            requests.post("http://localhost:8000/mi/melted_mi_data_to_db", json={"mi_melted_data": mi_melted_dict})
        return mi_melted_dict
    except requests.RequestException as e:
        st.error(f"Error in melting MI raw DataFrame: {str(e)}")
        return {}

# Selectbox for files
filenames = fetch_file_list(subdir)
MI_file = st.selectbox("Select raw MI file", filenames if filenames else ["No files available"])

melt_check = st.checkbox("Melt DataFrame after loading")

# Load Button for MI Data
load_btn = st.button("Load MI Data", type="primary")
with st.spinner("Loading MI data..."):
    if load_btn:
        if MI_file and MI_file != "No files available":
            mi_dict = load_mi_data(MI_file)
            st.session_state._temp_mi_dict = mi_dict
            
            
            mi_df = pd.DataFrame(mi_dict['data'], columns=mi_dict['columns'], index=mi_dict['index'])
            if not mi_df.empty:
                st.write("Raw MI Data:")
                st.write(mi_df)
                if '_temp_mi_dict' in st.session_state and melt_check:
                    mi_melted_dict = melt_raw_mi(st.session_state._temp_mi_dict)
                    st.session_state._temp_mi_melted_dict = mi_melted_dict
                    
                    
                    
                    if mi_melted_dict:
                        st.success("Melted MI data has been saved.")
                        
                else:
                    st.warning("No MI dataframe to melt or melting not requested.")
            else:
                st.warning("Loaded MI data is empty.")
        else:
            st.warning("Select a valid MI file to display.")

# Ensure session_state variables are updated
st.session_state.mi_dict = st.session_state.get('_temp_mi_dict', {})
st.session_state.mi_melted_dict = st.session_state.get('_temp_mi_melted_dict', {})



st.divider()
