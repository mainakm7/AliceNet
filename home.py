import streamlit as st 


st.header("Welcome to AliceNet")

for key in ['exp_dict', 'event_dict', 'mi_dict','mi_melt_df']:
    if key not in st.session_state.keys():
        st.session_state[key] = None