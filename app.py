import streamlit as st
import time
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FedProx Healthcare Dashboard", 
    page_icon="⚕️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
    .stButton>button { width: 100%; font-weight: bold; }
    .terminal-box { background-color: #1e1e1e; color: #4af626; padding: 15px; border-radius: 5px; font-family: 'Courier New', Courier, monospace; height: 200px; overflow-y: auto; margin-bottom: 20px;}
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #d1d5db; }
    .dark-metric-card { background-color: #262730; color: white; padding: 20px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("Privacy-Preserving Healthcare AI")
st.caption("Powered by Federated Proximal (FedProx) Optimization")

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["FL Orchestration Hub", "Performance Matrix", "Diagnostic Portal"])

# --- SCREEN 1: FEDERATED LEARNING ORCHESTRATION HUB ---
with tab1:
    st.header("Network Visualization")
    
    # Simulating Hub and Spoke with columns
    st.markdown("<div style='text-align: center;'><h3>Global Aggregator Server (HUB)</h3></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: gray;'>Global Model Broadcast &nbsp;&nbsp; | &nbsp;&nbsp; Local Gradients Upload</div><br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.info("Clinic Node 1\n\n(Local HAM10000 Data)")
    c2.info("Clinic Node 2\n\n(Local HAM10000 Data)")
    c3.info("Clinic Node 3\n\n(Local HAM10000 Data)")
    
    st.divider()
    
    col_btn, col_term = st.columns([1, 2])
    
    with col_btn:
        st.subheader("Control Panel")
        run_sim = st.button("Run FedProx Simulation", type="primary")
        
        st.subheader("Aggregated Metrics")
        m1, m2 = st.columns(2)
        metric_acc = m1.empty()
        metric_comm = m2.empty()
        
        metric_acc.metric("Validation Accuracy", "--%")
        metric_comm.metric("Comm. Overhead", "-- MB")

    with col_term:
        st.subheader("System Logs")
        terminal_output = st.empty()
        terminal_output.markdown("<div class='terminal-box'>System Idle. Waiting for execution command...</div>", unsafe_allow_html=True)

    if run_sim:
        log_text = ""
        logs = [
            "> Initializing secure connections to 4 client nodes...",
            "> Loading HAM10000 data partitions...",
            "> Initiating Local Training (Epoch 1)...",
            "> Node 1: Local training complete. Loss: 0.45",
            "> Node 2: Local training complete. Loss: 0.42",
            "> Node 3: Local training complete. Loss: 0.48",
            "> Node 4: Local training complete. Loss: 0.41",
            "> Receiving encrypted gradients...",
            "> Executing FedProx Aggregation (μ=0.01)...",
            "> Global weights updated successfully.",
            "> Broadcasting new model to clients... Done."
        ]
        
        for log in logs:
            log_text += log + "<br>"
            terminal_output.markdown(f"<div class='terminal-box'>{log_text}</div>", unsafe_allow_html=True)
            time.sleep(0.5) # Simulating processing time
            
        metric_acc.metric("Validation Accuracy", "94.2%", "+1.2%")
        metric_comm.metric("Comm. Overhead", "14.5 MB", "-2.1 MB")


# --- SCREEN 2: PERFORMANCE MATRIX ---
with tab2:
    st.header("Comparative Analytics")
    
    # Initialize session state variables to remember which models have been run
    if 'fed_run' not in st.session_state:
        st.session_state.fed_run = False
    if 'loc_run' not in st.session_state:
        st.session_state.loc_run = False
    if 'cen_run' not in st.session_state:
        st.session_state.cen_run = False

    c1, c2, c3 = st.columns(3)
    
    # Federated
    with c1:
        st.markdown("<div class='dark-metric-card'><h3>Federated (FedProx)</h3><p>Decentralized & Secure</p></div><br>", unsafe_allow_html=True)
        if st.button("Run Federated Model"):
            with st.spinner('Simulating distributed training...'):
                time.sleep(2)
            st.session_state.fed_run = True
            
        if st.session_state.fed_run:
            st.success("Training Complete")
            st.metric("Output Accuracy", "94.2%")
            st.metric("Model Loss", "0.21")

    # Localized
    with c2:
        st.markdown("<div class='dark-metric-card'><h3>Localized Training</h3><p>Isolated Node Only</p></div><br>", unsafe_allow_html=True)
        if st.button("Run Local Model"):
            with st.spinner('Simulating local training...'):
                time.sleep(1.5)
            st.session_state.loc_run = True
            
        if st.session_state.loc_run:
            st.warning("Training Complete (Overfitted to local data)")
            st.metric("Output Accuracy", "78.5%")
            st.metric("Model Loss", "0.65")

    # Centralized
    with c3:
        st.markdown("<div class='dark-metric-card'><h3>Centralized Training</h3><p>High Privacy Risk</p></div><br>", unsafe_allow_html=True)
        if st.button("Run Centralized Model"):
            with st.spinner('Simulating centralized training...'):
                time.sleep(2.5)
            st.session_state.cen_run = True
            
        if st.session_state.cen_run:
            st.success("Training Complete")
            st.metric("Output Accuracy", "95.8%")
            st.metric("Model Loss", "0.15")

# --- SCREEN 3: DIAGNOSTIC PORTAL ---
with tab3:
    st.header("Clinical Diagnostic Interface")
    
    col_upload, col_results = st.columns(2)
    
    with col_upload:
        uploaded_file = st.file_uploader("Upload Dermoscopic Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
        classify_btn = st.button("🔍 Classify Lesion", type="primary", use_container_width=True)
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col_results:
        st.subheader("Diagnostic Results")
        if classify_btn and uploaded_file is not None:
            with st.spinner('Analyzing image through global model...'):
                time.sleep(1.5)
            
            st.info("Prediction generated via Privacy-Preserving Federated Model")
            st.markdown("### Predicted Class: **Melanocytic Nevi (Benign)**")
            
            # Simulated confidence chart
            st.write("Confidence Breakdown:")
            chart_data = pd.DataFrame(
                {
                    "Classes": ["Melanocytic Nevi", "Melanoma", "Basal Cell Carcinoma", "Benign Keratosis"],
                    "Confidence (%)": [89.5, 5.2, 3.1, 2.2]
                }
            ).set_index("Classes")
            
            st.bar_chart(chart_data, color="#4af626")
            
        elif classify_btn and uploaded_file is None:
            st.error("Please upload an image first.")
        else:
            st.write("Upload an image and click Classify to see results.")
