import streamlit as st
import pandas as pd
import requests

# Ensure DataFrames are available in session_state
if "exp_df" in st.session_state and "event_df" in st.session_state:
    exp_df = st.session_state.exp_df
    event_df = st.session_state.event_df

    # Convert DataFrames to dictionary
    exp_dict = exp_df.to_dict(orient="split")
    event_dict = event_df.to_dict(orient="split")

    # Button to compute MI
    st.header("Compute MI:")
    mi_btn = st.button("Compute MI", type="primary")

    if mi_btn:
        try:
            response = requests.post("http://localhost:8000/mi/compute_mi", json={"sf_exp_df": exp_dict, "sf_events_df": event_dict})
            response.raise_for_status()
            data = response.json()
            
            mi_dict = data["raw_mi_data"]
            mi_df = pd.DataFrame(mi_dict["data"], columns=mi_dict["columns"], index=mi_dict["index"])
            
            # Write mi_df to session_state
            st.session_state.mi_df = mi_df
            
            with st.expander("Raw MI Dataframe:"):
                st.write(mi_df)
        except requests.RequestException as e:
            st.error(f"Error fetching MI data: {e}")
else:
    st.write("Expression and event dataframes are not available. First Load them!")

# Subdirectory Input
subdir = st.text_input("Enter subdirectory", value="MI")

def fetch_file_list(subdir: str):
    try:
        response = requests.get("http://localhost:8000/load/filenames", params={"subdir": subdir})
        response.raise_for_status()  # Raise an exception for HTTP errors
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

# Load MI Data
st.header("Load raw MI value from file")

def load_mi_data(filename):
    try:
        response = requests.post("http://localhost:8000/load/raw_mi", json={"filename": filename})
        response.raise_for_status()  # Raise an exception for HTTP errors
        mi_data = response.json()
        df = pd.DataFrame(mi_data['data'], columns=mi_data['columns'], index=mi_data['index'])
        return df
    except Exception as e:
        st.error(f"Error loading MI data: {str(e)}")
        # Return an empty DataFrame instead of None
        return pd.DataFrame()

# Selectbox for files
filenames = fetch_file_list(subdir)
MI_file = st.selectbox("Select raw MI file", filenames if filenames else ["No files available"])

# Display MI DataFrame
with st.expander("Raw MI Dataframe:"):
    if MI_file and MI_file != "No files available":
        mi_df = load_mi_data(MI_file)
        if not mi_df.empty:
            # Write mi_df to session_state
            st.session_state.mi_df = mi_df
            st.write(mi_df)
        else:
            st.write("No data available.")
    else:
        st.write("Select a valid MI file to display.")

# Melt MI Dataframe
st.header("Melt the dataframe:")

def melt_raw_mi():
    if "mi_df" in st.session_state:
        mi_df = st.session_state.mi_df
        mi_dict = mi_df.to_dict(orient="split")
        try:
            response = requests.post("http://localhost:8000/mi/melt_mi", json={"mi_raw_data": mi_dict})
            response.raise_for_status()  # Raise an exception for HTTP errors
            mi_melted_dict = response.json()

            mi_melted_df = pd.DataFrame(mi_melted_dict["data"], columns=mi_melted_dict["columns"], index=mi_melted_dict["index"])
            if not mi_melted_df.empty:
                st.session_state.mi_melt_df = mi_melted_df
                st.success("MI data melted successfully.")
        except Exception as e:
            st.error(f"Error in melting MI raw DataFrame: {str(e)}")
    else:
        st.error("No raw MI data available to melt.")

melt_btn = st.button("Melt MI", type="primary", on_click=melt_raw_mi)

with st.expander("Melted MI"):
    if "mi_melt_df" in st.session_state:
        st.write(st.session_state.mi_melt_df)
    else:
        st.write("No melted MI data available.")