import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import pandas as pd

# Check if session state variables exist
if 'exp_dict' not in st.session_state or 'event_dict' not in st.session_state:
    st.error("Expression and event data not found. Please load them first.")
else:
    # Initialize session state variables
    if 'gene' not in st.session_state:
        st.session_state.gene = None
    if 'specific_event' not in st.session_state:
        st.session_state.specific_event = None
    if 'best_params' not in st.session_state:
        st.session_state.best_params = None
    if 'adj_mat_dict' not in st.session_state:
        st.session_state.adj_mat_dict = None
    if 'num_clusters_heirarchical' not in st.session_state:
        st.session_state.num_clusters_heirarchical = 10
    if 'num_clusters_spectral' not in st.session_state:
        st.session_state.num_clusters_spectral = 10

    exp_dict = st.session_state.exp_dict
    event_dict = st.session_state.event_dict

    st.header("Clustering Splicing factors for each Splicing event!")

    st.subheader("Choose Gene and Specific Splicing Event")

    # Fetch gene list from the server
    gene_response = requests.post("http://localhost:8000/network/event_gene_select", json={"sf_events_df": event_dict})

    if gene_response.status_code == 201:
        gene_list = gene_response.json()
        gene = st.selectbox("Select specific gene for network", gene_list, index=0)

        gene_btn =  st.button("Select Gene")
        if gene_btn:  
            st.session_state._temp_gene = gene
            st.session_state.gene = gene

    else:
        st.error(f"Error fetching gene list: {gene_response.text}")

    # Fetch specific events for the selected gene
    if st.session_state._temp_gene: 
        event_response = requests.post(f"http://localhost:8000/network/specific_event_select/{st.session_state.gene}", json={"sf_events_df": event_dict})
        if event_response.status_code == 201:
            event_list = event_response.json()
            specific_event = st.selectbox("Select specific event for the chosen gene", event_list, index=0)

            if st.button("Select Specific Event"):
                st.session_state.specific_event = specific_event  # Update session state on button click

        else:
            st.error(f"Error fetching specific events: {event_response.text}")

    try:
        response = requests.post("http://localhost:8000/network/xgboostnetquery", json={
                                    "specific_gene": st.session_state.gene,
                                    "eventname": st.session_state.specific_event
                                })
        if response.status_code == 201:
            st.success("Best fit parameters are available in the Database")
        else:
            st.error("Best fit parameters not available. Please fit the network first")
    except Exception as e:
        st.error(f"Error querying the network Database: {e}")

    st.divider()
    
    st.subheader("Feature Clustering!")

    st.markdown("*SHAP calculation and performing Hierarchical clustering for each Latent variable to generate adjacency matrix \
            of all splicing factors for selected event across all samples*")

    st.markdown("*Generate Hierarchical cluster elbow plot:*")
    if st.button("generate elbow", type="primary"):
        with st.spinner("Generating elbow plot..."):
            try:
                response = requests.post("http://localhost:8000/network/hcluster_elbow", json={
                                    "specific_gene": st.session_state.gene,
                                    "eventname": st.session_state.specific_event
                                })
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                st.image(image, caption=f'Hierarchical cluster Elbow plot for event: {st.session_state.specific_event}', use_column_width=True)
            except Exception as e:
                st.error(f"Error occurred while obtaining Hierarchical elbow plot: {e}")

    hnum_cluster = st.number_input("Please select a cluster size based on the elbow plot", min_value=1, max_value=262, step=1, value=st.session_state.num_clusters_heirarchical)
    st.session_state.num_clusters_heirarchical = hnum_cluster  # Update cluster size in session state

    st.markdown("*Generate adjacency matrix of splicing factors for the selected event*")
    if st.button("generate hcluster", type="primary"):
        with st.spinner("Generating Adjacency Matrix..."):
            try:
                response = requests.post("http://localhost:8000/network/hcluster", json={
                                    "specific_gene": st.session_state.gene,
                                    "eventname": st.session_state.specific_event,
                                    "num_cluster": st.session_state.num_clusters_heirarchical
                                })
                response.raise_for_status()
                adj_mat_dict = response.json()
                st.session_state.adj_mat_dict = adj_mat_dict  # Update session state
                adj_mat_df = pd.DataFrame(adj_mat_dict["data"], columns=adj_mat_dict["columns"], index=adj_mat_dict["index"])
                st.text(f"Adjacency matrix for event: {st.session_state.specific_event}")
                st.write(adj_mat_df)
            except Exception as e:
                st.error(f"Error occurred while obtaining Hierarchical elbow plot: {e}")

    st.divider()
    st.subheader("Spectral clustering of splicing factors:")
    if st.session_state.adj_mat_dict is None:
        st.warning("First generate the adjacency matrix for the event")
    else:
        st.markdown("*Generate spectral cluster elbow plot:*")

        try:
            response = requests.post("http://localhost:8000/network/scluster_elbow", json={
                                    "paramreq": {
                                "specific_gene": st.session_state.gene,
                                "eventname": st.session_state.specific_event
                            }})
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            st.image(image, caption=f'Spectral cluster Elbow plot for event: {st.session_state.specific_event}', use_column_width=True)
        except Exception as e:
            st.error(f"Error occurred while obtaining Spectral elbow plot: {e}")

        snum_cluster = st.number_input("Please select a cluster size based on the elbow plot", min_value=1, max_value=262, step=1, value=st.session_state.num_clusters_spectral)
        st.session_state.num_clusters_spectral = snum_cluster  # Update cluster size in session state

        st.markdown(f"*Clustered Splicing factors for the selected event: {st.session_state.specific_event}*")
        try:
            response = requests.post("http://localhost:8000/network/scluster", json={
                                    "paramreq": {
                                "specific_gene": st.session_state.gene,
                                "eventname": st.session_state.specific_event,
                                "num_cluster": st.session_state.num_clusters_spectral
                            }})
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            st.image(image, caption=f'Spectral cluster for event: {st.session_state.specific_event}', use_column_width=True)
        except Exception as e:
            st.error(f"Error occurred while obtaining Spectral elbow plot: {e}")
