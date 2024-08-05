import streamlit as st
import pandas as pd
import requests

# Ensure DataFrames are available in session_state
if 'exp_dict' not in st.session_state or 'event_dict' not in st.session_state:
    st.error("Expression and event data not found. Please load them first.")
else:
    # Initialize session state variables if not already set
    if 'gene' not in st.session_state:
        st.session_state.gene = None
    if 'specific_event' not in st.session_state:
        st.session_state.specific_event = None
    if '_temp_gene' not in st.session_state:
        st.session_state._temp_gene = None
    if '_temp_specific_event' not in st.session_state:
        st.session_state._temp_specific_event = None
    if '_test_size' not in st.session_state:
        st.session_state._test_size = 0.3

    exp_dict = st.session_state.exp_dict
    event_dict = st.session_state.event_dict

    st.header("XGBoost Network")

    st.subheader("Choose Gene and Specific Splicing Event")
    gene_response = requests.post("http://localhost:8000/network/event_gene_select", json={"sf_events_df": event_dict})

    if gene_response.status_code == 200:
        gene_list = gene_response.json()
        gene = st.selectbox("Select specific gene for network", gene_list)

        if st.button("Select Gene"):
            st.session_state._temp_gene = gene

            event_response = requests.post(f"http://localhost:8000/network/specific_event_select/{st.session_state._temp_gene}", json={"sf_events_df": event_dict})
            if event_response.status_code == 200:
                event_list = event_response.json()
                specific_event = st.selectbox("Select specific event for the chosen gene", event_list)

                if st.button("Select Specific Event"):
                    st.session_state._temp_specific_event = specific_event
                    st.session_state.gene = st.session_state._temp_gene
                    st.session_state.specific_event = st.session_state._temp_specific_event
            else:
                st.error(f"Error fetching specific events: {event_response.text}")
    else:
        st.error(f"Error fetching gene list: {gene_response.text}")

    st.divider()

    st.subheader("Data Preparation")

    st.text("Choose a split ratio for train and test dataset")
    test_size = st.slider("Test Size", min_value=0.0, max_value=1.0, step=0.1, value=st.session_state._test_size)
    st.session_state._test_size = test_size

    if st.button("Data Preparation"):
        try:
            data_response = requests.post("http://localhost:8000/network/data_prepare", 
                                          json={"specific_gene": st.session_state.gene, "event": st.session_state.specific_event, "test_size": st.session_state._test_size})
            if data_response.status_code == 201:
                st.success("Data prepared successfully.")
                st.session_state.data_dict = data_response.json()
            else:
                st.error(f"Error in preparing data: {data_response.text}")
        except Exception as e:
            st.error(f"Error in preparing data: {e}")

    st.divider()

    st.subheader("Hyperparameter Optimization")

    if "data_dict" in st.session_state:
        st.text("Select ranges for hyperparameters")

        n_estimators = st.slider("n_estimators", 50, 200, (50, 200))
        max_depth = st.slider("max_depth", 3, 9, (3, 9))
        learning_rate = st.slider("learning_rate", 0.01, 0.3, (0.01, 0.3))
        min_child_weight = st.slider("min_child_weight", 0.001, 10.0, (0.001, 10.0))
        gamma = st.slider("gamma", 0.001, 10.0, (0.001, 10.0))
        subsample = st.slider("subsample", 0.5, 1.0, (0.5, 1.0))
        colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, (0.5, 1.0))
        reg_alpha = st.slider("reg_alpha", 0.001, 10.0, (0.001, 10.0))
        reg_lambda = st.slider("reg_lambda", 0.001, 10.0, (0.001, 10.0))

        hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda
        }

        if st.button("Optimize Hyperparameters"):
            try:
                hptuning_response = requests.post(
                    "http://localhost:8000/network/hptuning",
                    json={
                        "train_X": st.session_state.data_dict["train_X"],
                        "train_y": st.session_state.data_dict["train_y"],
                        "test_X": st.session_state.data_dict["test_X"],
                        "test_y": st.session_state.data_dict["test_y"],
                        "hparams": hyperparameters
                    }
                )
                if hptuning_response.status_code == 201:
                    st.success("Hyperparameter optimization completed successfully.")
                    optimized_params = hptuning_response.json()
                    st.write("Best Parameters:", optimized_params["best_params"])
                    st.write("Best Value:", optimized_params["best_value"])
                else:
                    st.error(f"Error in hyperparameter optimization: {hptuning_response.text}")
            except Exception as e:
                st.error(f"Error in hyperparameter optimization: {e}")
    else:
        st.warning("Prepare the data first before optimizing hyperparameters.")
