# Homepage.py
import streamlit as st
import pandas as pd

# Configure the page
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state to store data across pages
if 'df' not in st.session_state:
    st.session_state.df = None

def main():
    st.title("🤖 Welcome to AutoML Studio!")
    st.markdown("### **Automate Your Machine Learning Workflow with Ease**")
    
    st.image("https://images.unsplash.com/photo-1599658880436-c61792e70672?q=80&w=2070", use_column_width=True)

    st.markdown("""
    <div style="text-align: justify;">
    Hello and welcome! I'm Stryden, your personal AI assistant designed to simplify the machine learning process. Whether you're a seasoned data scientist or just starting, our studio provides a seamless, intuitive, and powerful platform to transform your raw data into actionable insights.
    <br><br>
    Our workflow is broken down into a few simple steps:
    <ul>
        <li><b>Data Uploader:</b> Start by uploading your dataset or choose one of our sample datasets to explore.</li>
        <li><b>Data Visualization:</b> Dive deep into your data. Understand its structure, visualize relationships between variables, and clean it up by handling any missing values.</li>
        <li><b>Model Trainer:</b> Select your features and target, choose from a wide range of supervised learning models, and let the studio train them to find the best performer for your specific problem.</li>
    </ul>
    Once the training is complete, you can analyze the performance metrics and download the best-performing model as a file, ready to be deployed in your own applications.
    <br><br>
    Ready to get started? Navigate to the <b>Data Uploader</b> page using the sidebar to begin your journey!
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.header("About AutoML Studio")
        st.markdown("This tool automates the process of data analysis, model training, and performance comparison, allowing you to focus on what matters most: solving problems with data.")
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit.")

if __name__ == "__main__":
    main()