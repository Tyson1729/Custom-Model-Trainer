import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None

def load_sample_dataset(name):
    """Load sample datasets for demonstration"""
    if name == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        return df
    elif name == "Boston":
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
        df['target'] = housing.target
        return df
    elif name == "Wine":
        from sklearn.datasets import load_wine
        wine = load_wine()
        df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        return df
    return None

st.title("📁 Data Uploader")
st.markdown("Upload your own dataset or select a sample dataset to get started.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file with your dataset"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

# Sample datasets section
st.markdown("---")
st.subheader("Or try a sample dataset:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🌸 Iris Dataset (Classification)", use_container_width=True):
        st.session_state.df = load_sample_dataset("Iris")
        st.success("✅ Iris dataset loaded!")
with col2:
    if st.button("🏠 California Housing (Regression)", use_container_width=True):
        st.session_state.df = load_sample_dataset("Boston")
        st.success("✅ California Housing dataset loaded!")
with col3:
    if st.button("🍷 Wine Dataset (Classification)", use_container_width=True):
        st.session_state.df = load_sample_dataset("Wine")
        st.success("✅ Wine Quality dataset loaded!")

# Display dataset statistics if a dataset is loaded
if st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("---")
    st.subheader("Dataset Overview")

    # Display Head and Tail
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Head:**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("**Dataset Tail:**")
        st.dataframe(df.tail(), use_container_width=True)

    # Display Shape, Columns, and other info
    st.write("**Dataset Shape:**")
    st.write(f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    st.write("**Column Names and Data Types:**")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str)
    })
    st.dataframe(info_df, use_container_width=True)
    
    st.write("**Missing Values Count:**")
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Count']
    st.dataframe(missing_values[missing_values['Missing Count'] > 0], use_container_width=True)