import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔍 Data Visualization & Cleaning")

if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Please upload a dataset on the 'Data Uploader' page first.")

else:
    df = st.session_state.df

    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Column Analysis", "📈 Target Analysis", "🔗 Correlation Analysis", "🧹 Missing Value Handler"])

    with tab1:
        st.header("Column Distribution Analysis")
        
        st.subheader("Numerical Columns")
        if numeric_cols:
            num_col_select = st.selectbox("Select a numerical column to visualize its distribution:", numeric_cols)
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            sns.histplot(df[num_col_select], kde=True, ax=ax[0])
            ax[0].set_title(f'Histogram of {num_col_select}')
            sns.boxplot(x=df[num_col_select], ax=ax[1])
            ax[1].set_title(f'Box Plot of {num_col_select}')
            st.pyplot(fig)
        else:
            st.info("No numerical columns found in the dataset.")

        st.subheader("Categorical Columns")
        if categorical_cols:
            cat_col_select = st.selectbox("Select a categorical column to visualize its distribution:", categorical_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(y=df[cat_col_select], order=df[cat_col_select].value_counts().index, ax=ax)
            ax.set_title(f'Count Plot of {cat_col_select}')
            st.pyplot(fig)
        else:
            st.info("No categorical columns found in the dataset.")

    with tab2:
        st.header("Target and Feature Analysis")
        all_cols = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select your Target Variable:", all_cols)
        with col2:
            feature_col = st.selectbox("Select a Feature Variable to plot against the target:", [col for col in all_cols if col != target_col])

        if st.button("Generate Plot"):
            fig, ax = plt.subplots(figsize=(10, 6))
            target_is_numeric = target_col in numeric_cols
            feature_is_numeric = feature_col in numeric_cols
            
            if target_is_numeric and feature_is_numeric:
                sns.scatterplot(data=df, x=feature_col, y=target_col, ax=ax)
                ax.set_title(f'Scatter Plot: {feature_col} vs {target_col}')
            elif not target_is_numeric and feature_is_numeric:
                sns.boxplot(data=df, x=target_col, y=feature_col, ax=ax)
                ax.set_title(f'Box Plot: {feature_col} by {target_col}')
            elif target_is_numeric and not feature_is_numeric:
                sns.boxplot(data=df, x=feature_col, y=target_col, ax=ax)
                ax.set_title(f'Box Plot: {target_col} by {feature_col}')
            else: # both categorical
                sns.countplot(data=df, x=feature_col, hue=target_col, ax=ax)
                ax.set_title(f'Count Plot: {feature_col} with hue {target_col}')
            
            st.pyplot(fig)

    with tab3:
        st.header("Correlation and Pairwise Analysis")
        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(12, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

            st.subheader("Pair Plot")
            st.info("Pair plots can be slow for datasets with many numerical features. We recommend selecting a few key features.")
            selected_pair_cols = st.multiselect("Select columns for Pair Plot:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
            if st.button("Generate Pair Plot"):
                pair_plot = sns.pairplot(df[selected_pair_cols])
                st.pyplot(pair_plot)
        else:
            st.info("At least two numerical columns are required for correlation analysis.")
            
    with tab4:
        st.header("Handle Missing Values")
        missing_data = df.isnull().sum()
        missing_cols = missing_data[missing_data > 0].index.tolist()

        if not missing_cols:
            st.success("🎉 No missing values found in the dataset!")
        else:
            st.write("Columns with missing values:")
            st.table(missing_data[missing_data > 0])

            col_to_fill = st.selectbox("Select a column to handle:", missing_cols)
            
            if col_to_fill in numeric_cols:
                method = st.selectbox("Select imputation method:", ["Mean Imputation", "Median Imputation", "Drop Rows", "Drop Column (Not Recommended)"])
            else: # categorical
                method = st.selectbox("Select imputation method:", ["Mode Imputation", "Drop Rows", "Drop Column (Not Recommended)"])

            if st.button("Apply Changes"):
                df_copy = df.copy()
                if method == "Mean Imputation":
                    fill_value = df_copy[col_to_fill].mean()
                    df_copy[col_to_fill].fillna(fill_value, inplace=True)
                elif method == "Median Imputation":
                    fill_value = df_copy[col_to_fill].median()
                    df_copy[col_to_fill].fillna(fill_value, inplace=True)
                elif method == "Mode Imputation":
                    fill_value = df_copy[col_to_fill].mode()[0]
                    df_copy[col_to_fill].fillna(fill_value, inplace=True)
                elif method == "Drop Rows":
                    df_copy.dropna(subset=[col_to_fill], inplace=True)
                elif method == "Drop Column (Not Recommended)":
                    df_copy.drop(columns=[col_to_fill], inplace=True)
                
                st.session_state.df = df_copy
                st.success(f"✅ Action '{method}' applied to column '{col_to_fill}'. The dataset has been updated.")
                st.rerun()