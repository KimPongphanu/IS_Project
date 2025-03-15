import streamlit as st

st.markdown('<div class="custom-title">MACHINE LEARNING</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Machine_Learning_Document"):
        st.switch_page("pages/Machine_Learning_Document.py")

with col2:
    if st.button("Machine_Learning_Model"):
        st.switch_page("pages/Machine_Learning_Model.py")

st.markdown("---")

st.markdown('<div class="custom-title">NEURAL NETWORK</div>', unsafe_allow_html=True)
col3, col4 = st.columns([1, 1])

with col3:
    if st.button("NeuralNetwork_Document"):
        st.switch_page("pages/NeuralNetwork_Document.py")

with col4:
    if st.button("NeuralNetwork_Model"):
        st.switch_page("pages/NeuralNetwork_Model.py")
