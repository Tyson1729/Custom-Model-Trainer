import streamlit as st
import pandas as pd

# Configure the page
st.set_page_config(
    page_title="Athena Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state to store data across pages
if 'df' not in st.session_state:
    st.session_state.df = None

def main():
    st.title("🤖 Welcome to Athena Studio!")
    st.markdown("### **From Messy CSV to Trained Model, We got you covered.**")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1550439062-609e1531270e?q=80&w=800", caption="Built by devs, for devs.", width=400)

    st.markdown("""
    <div style="text-align: justify;">
    Yo! Welcome to Athena Studio. We're a crew of devs who built this spot just to help you figure out your data without the usual headache. 
    <br><br>
    Tired of wrestling with notebooks just to see what's inside a CSV? We feel you. Our workflow is dead simple:
    <ul>
        <li><b>Data Uploader:</b> Drop your dataset or grab one of our samples to play around with. No judgment.</li>
        <li><b>Data Visualization:</b> Instantly see what's up with your data. Spot trends, find issues, and clean it up on the fly.</li>
        <li><b>Model Trainer:</b> Don't worry, we got your model training handled. Pick your features, choose a target, and let our custom trainer find the best algorithm for the job.</li>
    </ul>
    Once the training is done, you can predict your outcome, check the scores, and download the best-performing model as a single file, ready to drop into your own project.
    <br><br>
    Ready to build? Hit up the <b>Data Uploader</b> page in the sidebar and let's get this thing started!
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Information
    with st.sidebar:
        st.header("About Athena Studio")
        st.markdown("This tool automates the process of data analysis, model training, and performance comparison, allowing you to focus on what matters most: solving problems with data.")
        st.markdown("---")
        st.markdown("Built by Team Stryden:")
        st.markdown("Tyson")
        st.markdown("Epsilon")
        st.markdown("OdiusBull")
        st.markdown("Deadlock")

if __name__ == "__main__":
    main()