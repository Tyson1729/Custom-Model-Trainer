import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(layout="wide")
st.title("⚙️ Model Trainer")

if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Please upload a dataset on the 'Data Uploader' page first.")
else:
    df = st.session_state.df

    st.header("1. Select Features and Target")
    all_cols = df.columns.tolist()

    # --- ✅ CORRECTED CODE BLOCK STARTS HERE ---

    # Validate and update session state against current dataframe columns
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = [col for col in all_cols[:-1]]
    else:
        # Filter out any selected features that are no longer in the dataframe
        st.session_state.selected_features = [f for f in st.session_state.selected_features if f in all_cols]

    if 'selected_target' not in st.session_state or st.session_state.selected_target not in all_cols:
        st.session_state.selected_target = all_cols[-1]
    
    # Ensure target is not in the default feature list
    if st.session_state.selected_target in st.session_state.selected_features:
        st.session_state.selected_features.remove(st.session_state.selected_target)

    # --- WIDGET CREATION ---
    
    target_col = st.selectbox(
        "Select Target Column (y):",
        all_cols,
        index=all_cols.index(st.session_state.selected_target)
    )

    # Update available options for features by removing the selected target
    feature_options = [col for col in all_cols if col != target_col]
    
    selected_features = st.multiselect(
        "Select Feature Columns (X):",
        feature_options, # Use filtered options
        default=st.session_state.selected_features
    )

    # Update session state with new selections
    st.session_state.selected_features = selected_features
    st.session_state.selected_target = target_col

    # --- Validation ---
    valid_selection = True
    if not selected_features:
        st.error("Please select at least one feature column.")
        valid_selection = False
    if not target_col:
        st.error("Please select a target column.")
        valid_selection = False
    if target_col in selected_features:
        st.error("Target column cannot be in the feature columns.")
        valid_selection = False

    if valid_selection:
        st.success("✅ Feature and target selections are valid.")
        
        # --- Problem Type Detection ---
        target_dtype = df[target_col].dtype
        unique_values = df[target_col].nunique()
        
        if pd.api.types.is_numeric_dtype(target_dtype) and unique_values > 20:
            problem_type = "Regression"
        else:
            problem_type = "Classification"
            
        st.info(f"**Detected Problem Type:** {problem_type}")

        # --- Model Selection ---
        st.header("2. Select Models to Train")
        
        if problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }
        else: # Classification
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }
        
        selected_models = []
        select_all = st.checkbox("Select All Models", value=True)
        if select_all:
            selected_models = list(models.keys())
        else:
            for model_name in models.keys():
                if st.checkbox(model_name, value=True):
                    selected_models.append(model_name)
        
        # --- Training Configuration ---
        st.header("3. Configure Training")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        with col2:
            cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)

        # --- Start Training ---
        if st.button("🚀 Start Model Training", type="primary", use_container_width=True):
            if not selected_models:
                st.error("Please select at least one model to train.")
            else:
                with st.spinner("Training in progress... Please wait. 🤖"):
                    # Prepare data
                    X = df[selected_features]
                    y = df[target_col]

                    # Preprocessing pipelines
                    numeric_features = X.select_dtypes(include=['number']).columns
                    categorical_features = X.select_dtypes(exclude=['number']).columns

                    numeric_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])
                    
                    categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        # Using one-hot encoding with handle_unknown to avoid errors with unseen categories
                        ('onehot', pd.get_dummies, {'drop_first': True}) 
                    ])
                    # Note: Using pd.get_dummies is a simplified approach for the pipeline.
                    # For a more robust solution, sklearn's OneHotEncoder is preferred but more complex to set up here.
                    
                    # For simplicity in this script, we'll manually apply encoding for now.
                    X = pd.get_dummies(X, columns=categorical_features.tolist(), drop_first=True)
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    scaler = StandardScaler()
                    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
                    X_test[numeric_features] = scaler.transform(X_test[numeric_features])


                    results = []
                    best_model = None
                    best_score = -np.inf if problem_type == "Classification" else -np.inf
                    best_model_name = ""

                    for name in selected_models:
                        model = models[name]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        if problem_type == "Regression":
                            score = r2_score(y_test, y_pred)
                            metric = "R2 Score"
                            if best_model is None or score > best_score:
                                best_score = score
                                best_model = model
                                best_model_name = name
                        else: # Classification
                            score = accuracy_score(y_test, y_pred)
                            metric = "Accuracy"
                            if best_model is None or score > best_score:
                                best_score = score
                                best_model = model
                                best_model_name = name
                        results.append({"Model": name, metric: f"{score:.4f}"})

                    st.session_state.results_df = pd.DataFrame(results)
                    st.session_state.best_model_obj = best_model
                    st.session_state.best_model_name = best_model_name
                    st.session_state.scaler = scaler # save scaler for download
                    st.session_state.training_cols = X.columns.tolist()
                
                st.success("✅ Training completed successfully!")

    # --- Display Results and Download ---
    if 'results_df' in st.session_state:
        st.header("4. Model Performance Results")
        st.dataframe(st.session_state.results_df, use_container_width=True)

        st.subheader("🏆 Best Performing Model")
        best_name = st.session_state.best_model_name
        st.success(f"The best model is **{best_name}**.")

        # --- Download Model ---
        st.subheader("📥 Download the Best Model")
        
        model_filename = f"{best_name.replace(' ', '_').lower()}_model.pkl"
        
        # Create a dictionary to save all necessary components
        model_pack = {
            'model': st.session_state.best_model_obj,
            'scaler': st.session_state.scaler,
            'training_columns': st.session_state.training_cols,
            'problem_type': problem_type
        }
        
        # Save the dictionary using joblib
        joblib.dump(model_pack, model_filename)

        with open(model_filename, "rb") as file:
            st.download_button(
                label=f"Download {best_name} Model",
                data=file,
                file_name=model_filename,
                mime="application/octet-stream"
            )
        
        os.remove(model_filename)