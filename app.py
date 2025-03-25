# app.py

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
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>✨ RAGcipe Culinary Assistant ✨</div>", unsafe_allow_html=True)
    st.write("Enter your OpenAI API Key and a culinary query to receive recipe suggestions with affordable ingredient recommendations, nutritional analysis, and cost estimates. :chef:")

    # Ask for the user's OpenAI API key
    user_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
    
    if user_api_key:
        # Set the API key in the environment variable
        os.environ["OPENAI_API_KEY"] = user_api_key
        # Reload the Full_Prompt_new module to pick up the new API key
        importlib.reload(Full_Prompt_new)
        st.success("✅ API Key has been set successfully.")
        
        # Allow the user to enter their culinary query
        query = st.text_input("🍽️ Enter your culinary query:", "high protein tofu dish")
        
        if st.button("🚀 Get Culinary Response"):
            with st.spinner("⏳ Querying... Please wait."):
                response = Full_Prompt_new.query_all(query)
            st.subheader("LLM Response")
            st.markdown(response)
    else:
        st.warning("⚠️ Please enter your OpenAI API Key to continue.")
    
    st.markdown("<div class='footer'>© 2025 RAGcipe Team - Powered by OpenAI & FairPrice Data</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
