#user_input.py

import streamlit as st
import io
import contextlib
import base64  
from rag_agent import ask_legal_agent


st.title(" Legal RAG Agent")
st.title("honest pm iimbangalore")

user_input = st.text_input("Ask your question:")
# In app.py

if st.button("Submit"):
    if user_input:
        try:
            # We no longer need redirect_stdout here because 
            # our function now returns the logs explicitly
            answer, logs = ask_legal_agent(user_input)

            st.markdown("### Answer")
            st.success(answer)

            if logs:
                with st.expander("🔍 State Logs"):
                    st.text(logs) # Use st.text or st.code to display

        except Exception as e:
            st.error(f"Error: {e}")