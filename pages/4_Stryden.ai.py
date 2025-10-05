import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

st.set_page_config(
    page_title="Stryden.ai Assistant",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Stryden AI - At Your Service")
st.markdown("Yo! Ask me anything about your data, machine learning concepts, or how to use this tool!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown("Enter your API key below. Your key is only used for your session.")
    
    api_key_input = st.text_input(
        "Google Gemini API Key:",
        type="password",
        help="Get your key from Google AI Studio. It is not stored after your session ends."
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("API Key entered. You can now chat with Stryden AI.")

if st.session_state.api_key:
    try:
        genai.configure(api_key=st.session_state.api_key)
    except Exception as e:
        st.error(f"Failed to configure API Key: {e}. Please check if the key is valid.")
        st.session_state.api_key = None

def create_prompt_template(df):
    """Creates a contextual prompt based on whether a dataset is loaded."""
    if df is not None:
        prompt_intro = "You are Stryden.ai, an expert data science assistant integrated into a Streamlit application. A user has uploaded a dataset and needs your guidance. Your tone should be helpful, encouraging, and clear."
        
        data_head = df.head().to_string()
        data_info_str = "Could not generate info string."
        try:
            import io
            buffer = io.StringIO()
            df.info(buf=buffer)
            data_info_str = buffer.getvalue()
        except Exception:
            pass

        data_shape = df.shape
        missing_values = df.isnull().sum().sum()

        context = f"""
        Here is the context about the user's current dataset:
        - **Dataset Shape:** {data_shape[0]} rows and {data_shape[1]} columns.
        - **Total Missing Values:** {missing_values}
        - **Data Info:** {data_info_str}
        - **Data Preview (first 5 rows):**
        {data_head}
        Based on this context, provide specific, actionable advice. Guide them on which plots to use in the 'Data Visualization' page, which columns might be good targets for the 'Model Trainer', how to handle the missing values, or answer any other questions they have about this specific dataset.
        """
        return prompt_intro + context
    else:
        prompt_intro = "You are Stryden.ai, an expert data science assistant integrated into a Streamlit application. The user has not uploaded a dataset yet. Your tone should be helpful, encouraging, and clear."
        
        context = """
        Your goal is to guide the user on how to get started. You can:
        - Explain the purpose of the application (Data Upload -> Visualization -> Model Training).
        - Suggest using one of the sample datasets (Iris, California Housing, Wine) to explore the app's features.
        - Give general advice on what makes a good dataset for machine learning (e.g., clean data, clear target variable).
        - Explain common visualization plots (like histograms for distribution, scatter plots for relationships) and why they are useful.
        - Answer general data science or machine learning questions.
        """
        return prompt_intro + context

# Chatbot
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not st.session_state.api_key:
        st.warning("Please enter your Google Gemini API key in the sidebar to get started!")
        st.stop()

    try:
        with st.spinner("🧠 Stryden is thinking..."):
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            system_prompt = create_prompt_template(st.session_state.df)
            full_prompt = system_prompt + "\n\nUser's question: " + prompt
            
            response = model.generate_content(full_prompt)
            
            try:
                response_text = response.text
            
            except ValueError:
                response_text = "I'm sorry, I cannot provide a response to that prompt. It may violate safety policies. Please try again."

        with st.chat_message("assistant"):
            st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        st.error(f"An error occurred: {e}")
        error_message = f"Sorry, I ran into an error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    if not st.session_state.api_key:
        st.info("👋 Welcome! Please add your Gemini API key in the sidebar to activate the assistant.")