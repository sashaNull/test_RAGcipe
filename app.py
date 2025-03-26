import sys
sys.modules["torch.classes"] = None

# app.py
import streamlit as st
import os
import importlib
import Full_Prompt_new  # Ensure this file is in your project folder

# Custom CSS to style the page
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
        padding: 2rem;
    }
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
    }
    .subheader {
        font-size: 1.5rem;
        color: #333333;
    }
    .icon {
        font-size: 2rem;
    }
    .footer {
        font-size: 0.8rem;
        text-align: center;
        color: #777777;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.markdown("<div class='header'>✨ RAGcipe Culinary Assistant ✨</div>", unsafe_allow_html=True)
        
    # Allow the user to enter their culinary query
    query = st.text_input("🍽️ Enter your culinary query:", "high protein tofu dish")

    if st.button("🚀 Get Culinary Response"):
        with st.spinner("⏳ Querying... Please wait."):
            response = Full_Prompt_new.query_all(query)
            if isinstance(response, dict) and "answer" in response:
                formatted_response = response["answer"]
            else:
                formatted_response = str(response)
            st.subheader("LLM Response")
            st.markdown(formatted_response, unsafe_allow_html=True)
    
    st.markdown("<div class='footer'>© 2025 RAGcipe Team - Powered by OpenAI & FairPrice Data</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()