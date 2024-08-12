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
    st.session_state.setdefault('gene', None)
    st.session_state.setdefault('specific_event', None)
    st.session_state.setdefault('best_params', None)
    st.session_state.setdefault('adj_mat_dict', None)
    st.session_state.setdefault('num_clusters_heirarchical', None)
    st.session_state.setdefault('num_clusters_spectral', None)

    exp_dict = st.session_state.exp_dict
    event_dict = st.session_state.event_dict

    st.header("Clustering Splicing Factors for Each Splicing Event")

    st.subheader("Choose Gene and Specific Splicing Event")

    # Fetch gene list from the server
    gene_response = requests.post("http://localhost:8000/network/event_gene_select", json={"sf_events_df": event_dict})

    if gene_response.status_code == 201:
        gene_list = gene_response.json()
        gene = st.selectbox("Select specific gene for network", gene_list, index=0)
        if st.button("Select Gene"):  
            st.session_state.gene = gene

    else:
        st.error(f"Error fetching gene list: {gene_response.text}")

    # Fetch specific events for the selected gene
    if st.session_state.gene: 
        event_response = requests.post(f"http://localhost:8000/network/specific_event_select/{st.session_state.gene}", json={"sf_events_df": event_dict})
        if event_response.status_code == 201:
            event_list = event_response.json()
            specific_event = st.selectbox("Select specific event for the chosen gene", event_list, index=0)
            if st.button("Select Specific Event"):
                st.session_state.specific_event = specific_event

        else:
            st.error(f"Error fetching specific events: {event_response.text}")

    # Query the network Database for best fit parameters
    if st.session_state.gene and st.session_state.specific_event:
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
    
    st.subheader("Feature Clustering")

    st.markdown("Perform SHAP calculation and hierarchical clustering for each latent variable to generate the adjacency matrix of all splicing factors for the selected event across all samples.")

    # Generate Hierarchical cluster elbow plot
    if st.button("Generate Elbow Plot"):
        with st.spinner("Generating elbow plot..."):
            try:
                response = requests.post("http://localhost:8000/network/hcluster_elbow", json={
                    "specific_gene": st.session_state.gene,
                    "eventname": st.session_state.specific_event
                })
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                st.image(image, caption=f'Hierarchical Cluster Elbow Plot for Event: {st.session_state.specific_event}', use_column_width=True)
            except Exception as e:
                st.error(f"Error occurred while obtaining Hierarchical elbow plot: {e}")

    # Hierarchical cluster number input
    hnum_cluster = st.number_input("Select cluster size based on the elbow plot", min_value=1, max_value=262, step=1, value=10)
    st.session_state.num_clusters_heirarchical = hnum_cluster

    # Generate Adjacency Matrix
    if st.button("Generate Adjacency Matrix"):
        with st.spinner("Generating Adjacency Matrix..."):
            try:
                response = requests.post("http://localhost:8000/network/hcluster", json={
                    "specific_gene": st.session_state.gene,
                    "eventname": st.session_state.specific_event,
                    "num_cluster": st.session_state.num_clusters_heirarchical
                })
                response.raise_for_status()
                adj_mat_dict = response.json()
                st.session_state.adj_mat_dict = adj_mat_dict
                adj_mat_df = pd.DataFrame(adj_mat_dict["data"], columns=adj_mat_dict["columns"], index=adj_mat_dict["index"])
                st.text(f"Adjacency Matrix for Event: {st.session_state.specific_event}")
                st.write(adj_mat_df)
            except Exception as e:
                st.error(f"Error occurred while obtaining Adjacency Matrix: {e}")

    st.divider()
    
    st.subheader("Spectral Clustering of Splicing Factors")

    if st.session_state.adj_mat_dict is None:
        st.warning("Generate the adjacency matrix for the event first.")
    else:
        # Generate Spectral cluster elbow plot
        if st.button("Generate Spectral Clustering Elbow Plot"):
            with st.spinner("Generating Spectral Clustering Elbow Plot..."):
                try:
                    response = requests.post("http://localhost:8000/network/scluster_elbow", json={
                        "specific_gene": st.session_state.gene,
                        "eventname": st.session_state.specific_event
                    })
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption=f'Spectral Cluster Elbow Plot for Event: {st.session_state.specific_event}', use_column_width=True)
                except Exception as e:
                    st.error(f"Error occurred while obtaining Spectral elbow plot: {e}")

        # Spectral cluster number input
        snum_cluster = st.number_input("Select cluster size based on the spectral elbow plot", min_value=1, max_value=262, step=1, value=10)
        st.session_state.num_clusters_spectral = snum_cluster

        # Generate Spectral Clusters
        if st.button("Generate Splicing Factor Clusters"):
            with st.spinner("Generating clusters of splicing factors..."):
                try:
                    response = requests.post("http://localhost:8000/network/scluster", json={
                        "specific_gene": st.session_state.gene,
                        "eventname": st.session_state.specific_event,
                        "num_cluster": st.session_state.num_clusters_spectral
                    })
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption=f'Spectral Clustering for Event: {st.session_state.specific_event}', use_column_width=True)
                except Exception as e:
                    st.error(f"Error occurred while obtaining Spectral clusters: {e}")
