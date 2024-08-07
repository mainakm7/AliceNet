import streamlit as st
import requests

# Check if session state variables exist
if 'exp_dict' not in st.session_state or 'event_dict' not in st.session_state:
    st.error("Expression and event data not found. Please load them first.")
elif 'mi_dict' not in st.session_state or 'mi_melted_dict' not in st.session_state:
    st.error("MI data data not found. Please load them first.")
else:
    # Initialize session state variables
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

    # Fetch gene list from the server
    gene_response = requests.post("http://localhost:8000/network/event_gene_select", json={"sf_events_df": event_dict})

    if gene_response.status_code == 201:
        gene_list = gene_response.json()
        gene = st.selectbox("Select specific gene for network", gene_list)

        if st.button("Select Gene"):
            st.session_state._temp_gene = gene
            st.session_state.gene = gene
    else:
        st.error(f"Error fetching gene list: {gene_response.text}")

    # Fetch specific events for the selected gene
    if st.session_state._temp_gene:
        event_response = requests.post(f"http://localhost:8000/network/specific_event_select/{st.session_state._temp_gene}", json={"sf_events_df": event_dict})
        if event_response.status_code == 201:
            event_list = event_response.json()
            specific_event = st.selectbox("Select specific event for the chosen gene", event_list)

            if st.button("Select Specific Event"):
                st.session_state._temp_specific_event = specific_event
                st.session_state.specific_event = specific_event
        else:
            st.error(f"Error fetching specific events: {event_response.text}")

    st.divider()

    st.subheader("Data Preparation and Hyperparameter tuning")

    st.text("Choose a split ratio for train and test dataset")
    test_size = st.slider("Test Size", min_value=0.0, max_value=1.0, step=0.1, value=st.session_state._test_size)
    st.session_state._test_size = test_size

    # if st.button("Data Preparation and Hyperparameter tuning"):
    #     if st.session_state.specific_event:
    #         try:
    #             data_response = requests.post(
    #                 "http://localhost:8000/network/data_prepare",
    #                 json={
    #                     "paramreq": {
    #                         "eventname": st.session_state.specific_event,
    #                         "test_size": st.session_state._test_size
    #                     },
    #                     "datareq": {
    #                         "mi_melted_data": st.session_state.mi_melted_dict,
    #                         "sf_exp_df": st.session_state.exp_dict,
    #                         "sf_events_df": st.session_state.event_dict
    #                     }
    #                 }
    #             )
    #             if data_response.status_code == 201:
    #                 st.success("Data prepared successfully.")
    #                 st.session_state.data_dict = data_response.json()
    #             else:
    #                 st.error(f"Error in preparing data: {data_response.text}")
    #         except Exception as e:
    #             st.error(f"Error in preparing data: {e}")
    #     else:
    #         st.error("Please select both a gene and a specific event before preparing data.")

    st.divider()

    
    with st.form(key='hyperparameter_form'):
        st.text("Select ranges for hyperparameters")
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators_min = st.number_input("n_estimators - Min", min_value=1, max_value=1000, value=50)
            max_depth_min = st.number_input("max_depth - Min", min_value=1, max_value=50, value=3)
            learning_rate_min = st.number_input("learning_rate - Min", min_value=0.001, max_value=1.0, step=0.001, value=0.01) 
            min_child_weight_min = st.number_input("min_child_weight - Min", min_value=0.001, max_value=10.0, step=0.001, value=0.001)
            gamma_min = st.number_input("gamma - Min", min_value=0.001, max_value=10.0, step=0.001, value=0.001)
            subsample_min = st.number_input("subsample - Min", min_value=0.1, max_value=1.0, step=0.01, value=0.5)
            colsample_bytree_min = st.number_input("colsample_bytree - Min", min_value=0.1, max_value=1.0, step=0.01, value=0.5)
            reg_alpha_min = st.number_input("reg_alpha - Min", min_value=0.001, max_value=10.0, step=0.001, value=0.001)
            reg_lambda_min = st.number_input("reg_lambda - Min", min_value=0.001, max_value=10.0, step=0.001, value=0.001)
            
        
        with col2:
            n_estimators_max = st.number_input("n_estimators - Max", min_value=1, max_value=1000, value=200)
            max_depth_max = st.number_input("max_depth - Max", min_value=1, max_value=50, value=9)
            learning_rate_max = st.number_input("learning_rate - Max", min_value=0.001, max_value=1.0, step=0.001, value=0.3)
            min_child_weight_max = st.number_input("min_child_weight - Max", min_value=0.001, max_value=10.0, step=0.001, value=10.0)
            gamma_max = st.number_input("gamma - Max", min_value=0.001, max_value=10.0, step=0.001, value=10.0)
            subsample_max = st.number_input("subsample - Max", min_value=0.1, max_value=1.0, step=0.01, value=1.0)
            colsample_bytree_max = st.number_input("colsample_bytree - Max", min_value=0.1, max_value=1.0, step=0.01, value=1.0)
            reg_alpha_max = st.number_input("reg_alpha - Max", min_value=0.001, max_value=10.0, step=0.001, value=10.0)
            reg_lambda_max = st.number_input("reg_lambda - Max", min_value=0.001, max_value=10.0, step=0.001, value=10.0)
            

        submit_button = st.form_submit_button("Optimize Hyperparameters")

        if submit_button:
            hyperparameters = {
                "n_estimators": (n_estimators_min, n_estimators_max),
                "max_depth": (max_depth_min, max_depth_max),
                "learning_rate": (learning_rate_min, learning_rate_max),
                "min_child_weight": (min_child_weight_min, min_child_weight_max),
                "gamma": (gamma_min, gamma_max),
                "subsample": (subsample_min, subsample_max),
                "colsample_bytree": (colsample_bytree_min, colsample_bytree_max),
                "reg_alpha": (reg_alpha_min, reg_alpha_max),
                "reg_lambda": (reg_lambda_min, reg_lambda_max)
            }

            with st.spinner("Optimizing hyperparameters..."):
                try:
                    hptuning_response = requests.post(
                        "http://localhost:8000/network/hptuning",
                        json={
                            "paramreq": {
                        "eventname": st.session_state.specific_event,
                        "test_size": st.session_state._test_size
                    },
                    "datareq": {
                        "mi_melted_data": st.session_state.mi_melted_dict,
                        "sf_exp_df": st.session_state.exp_dict,
                        "sf_events_df": st.session_state.event_dict
                    },
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

st.divider()
st.subheader("Fit the Network with the optimum parameters and add to Database")
if st.button("Model Fit", type="primary"):
    try:
        xgboostfit_response = requests.post(
            "http://localhost:8000/network/xgboostnetfit",
            json={
                "paramreq": {
                    "eventname": st.session_state.specific_event,
                    "specific_gene": st.session_state.gene,
                    "test_size": st.session_state._test_size
                },
                "datareq": {
                    "mi_melted_data": st.session_state.mi_melted_dict,
                    "sf_exp_df": st.session_state.exp_dict,
                    "sf_events_df": st.session_state.event_dict
                }
            }
        )
        xgboostfit_response.raise_for_status()
        xgbresponse = xgboostfit_response.json()
        st.write(xgbresponse["message"])
    except requests.RequestException as e:
        st.error(f"Error while fitting xgboostnet: {e}")
