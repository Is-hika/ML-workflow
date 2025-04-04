import streamlit as st
import streamlit as st
import os
import sys

# Print Python version for debugging
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Configure page properties
st.set_page_config(
    page_title="ML Workflow Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import datetime
from sklearn.model_selection import train_test_split
import joblib

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualizer import Visualizer
from utils import get_dataset_description, get_model_description

# Set page configuration
st.set_page_config(
    page_title="ML Workflow Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'categorical_features' not in st.session_state:
    st.session_state.categorical_features = []
if 'numerical_features' not in st.session_state:
    st.session_state.numerical_features = []
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Title
st.title("🤖 Machine Learning Workflow Platform")
st.markdown("### A beginner-friendly platform to experiment with ML models")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a step in the ML workflow:",
    [
        "1. Problem Definition",
        "2. Data Upload",
        "3. Exploratory Data Analysis",
        "4. Data Preprocessing",
        "5. Feature Engineering",
        "6. Model Selection & Training",
        "7. Model Evaluation",
        "8. Make Predictions"
    ]
)

# Create directory for saving models if it doesn't exist
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Initialize data processor, model trainer and visualizer
data_processor = DataProcessor()
model_trainer = ModelTrainer()
visualizer = Visualizer()

# 1. Problem Definition
if page == "1. Problem Definition":
    st.header("1. Problem Definition")
    
    st.markdown("""
    ### What is Machine Learning?
    
    Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed. 
    It focuses on developing algorithms that can learn from and make predictions on data.
    
    ### Types of Machine Learning Problems
    
    There are several types of machine learning problems:
    
    - **Classification**: Predict a categorical label (e.g., spam or not spam)
    - **Regression**: Predict a continuous value (e.g., house price)
    - **Clustering**: Group similar data points together
    - **Dimensionality Reduction**: Reduce the number of variables in a dataset
    """)
    
    problem_type = st.radio(
        "What type of problem do you want to solve?",
        ["Classification", "Regression"]
    )
    
    if problem_type == "Classification":
        st.info("You've selected a classification problem. In this type of problem, you'll predict a categorical label.")
        st.markdown("""
        **Examples of classification problems:**
        - Email spam detection
        - Disease diagnosis
        - Customer churn prediction
        - Sentiment analysis
        """)
    else:
        st.info("You've selected a regression problem. In this type of problem, you'll predict a continuous value.")
        st.markdown("""
        **Examples of regression problems:**
        - House price prediction
        - Stock price prediction
        - Sales forecasting
        - Temperature prediction
        """)
    
    st.session_state.problem_type = problem_type
    
    st.markdown("""
    ### Next Steps
    
    Now that you've defined your problem type, you can move on to the next step: **Data Upload**.
    
    In this step, you'll upload the dataset that will be used to train your machine learning model.
    """)

# 2. Data Upload
elif page == "2. Data Upload":
    st.header("2. Data Upload")
    
    st.markdown("""
    ### Upload Your Dataset
    
    Upload a CSV file containing your data. The first row should contain column names.
    
    Tips for preparing your data:
    - Ensure your data is in CSV format
    - Make sure the first row contains column headers
    - Check for and handle any special characters
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.dataset = data
            st.session_state.categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
            st.session_state.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            st.success(f"Dataset loaded successfully! Shape: {data.shape}")
            
            st.subheader("Preview of the dataset")
            st.dataframe(data.head())
            
            st.subheader("Dataset Information")
            buffer = data_processor.get_info_buffer(data)
            st.text(buffer)
            
            st.subheader("Select Target Variable")
            st.info("The target variable is what you want to predict.")
            
            target_column = st.selectbox(
                "Select the target column (what you want to predict)",
                data.columns.tolist()
            )
            
            st.session_state.target_column = target_column
            
            # Automatically determine if it's a classification or regression problem
            if st.session_state.target_column in data.columns:
                unique_values = data[st.session_state.target_column].nunique()
                if unique_values < 10 or data[st.session_state.target_column].dtype == 'object':
                    problem_type = "Classification"
                else:
                    problem_type = "Regression"
                
                st.session_state.problem_type = problem_type
                st.info(f"Based on your target variable, this appears to be a {problem_type.lower()} problem.")
            
            st.markdown("""
            ### Next Steps
            
            Now that you've uploaded your data and selected the target variable, you can move on to 
            exploratory data analysis to better understand your dataset.
            """)
            
        except Exception as e:
            st.error(f"Error loading the dataset: {str(e)}")
    else:
        if st.button("Use Sample Dataset"):
            if 'problem_type' in st.session_state and st.session_state.problem_type == "Classification":
                # Iris dataset for classification
                from sklearn.datasets import load_iris
                iris = load_iris()
                data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
                data['target'] = iris.target
                st.session_state.target_column = 'target'
                st.session_state.problem_type = "Classification"
            else:
                # Boston housing dataset for regression
                from sklearn.datasets import fetch_california_housing
                housing = fetch_california_housing()
                data = pd.DataFrame(data=housing.data, columns=housing.feature_names)
                data['MEDV'] = housing.target
                st.session_state.target_column = 'MEDV'
                st.session_state.problem_type = "Regression"
            
            st.session_state.dataset = data
            st.session_state.categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
            st.session_state.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            st.success(f"Sample dataset loaded successfully! Shape: {data.shape}")
            
            st.subheader("Preview of the dataset")
            st.dataframe(data.head())
            
            st.subheader("Dataset Information")
            buffer = data_processor.get_info_buffer(data)
            st.text(buffer)
            
            st.info(f"Target variable '{st.session_state.target_column}' automatically selected.")
            st.info(f"This is a {st.session_state.problem_type.lower()} problem.")

# 3. Exploratory Data Analysis
elif page == "3. Exploratory Data Analysis":
    st.header("3. Exploratory Data Analysis (EDA)")
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset in the Data Upload step first.")
    else:
        data = st.session_state.dataset
        
        st.markdown("""
        Exploratory Data Analysis (EDA) is a critical step in the ML workflow. It helps you understand your data, 
        identify patterns, detect anomalies, and test assumptions. Let's explore your dataset!
        """)
        
        st.subheader("Dataset Summary")
        st.dataframe(data.describe())
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage (%)': missing_percent
        })
        
        if missing_data.sum() > 0:
            st.dataframe(missing_df[missing_df['Missing Values'] > 0])
            st.warning(f"Your dataset contains missing values. You can handle them in the Data Preprocessing step.")
        else:
            st.success("No missing values found in your dataset!")
            
        # Data distributions
        st.subheader("Data Distributions")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Distribution Plots", "Correlation Heatmap", "Scatter Plots", "Box Plots", "Target Distribution"]
        )
        
        if viz_type == "Distribution Plots":
            column = st.selectbox("Select column for distribution plot", st.session_state.numerical_features)
            fig = visualizer.plot_distribution(data, column)
            st.plotly_chart(fig)
            
        elif viz_type == "Correlation Heatmap":
            fig = visualizer.plot_correlation(data)
            st.plotly_chart(fig)
            
        elif viz_type == "Scatter Plots":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis column", st.session_state.numerical_features)
            with col2:
                y_col = st.selectbox("Select Y-axis column", [c for c in st.session_state.numerical_features if c != x_col])
            
            color_by = st.selectbox(
                "Color by (optional)", 
                ["None"] + data.columns.tolist(),
                index=0
            )
            
            if color_by == "None":
                fig = px.scatter(data, x=x_col, y=y_col)
            else:
                fig = px.scatter(data, x=x_col, y=y_col, color=color_by)
            
            st.plotly_chart(fig)
            
        elif viz_type == "Box Plots":
            column = st.selectbox("Select column for box plot", st.session_state.numerical_features)
            
            group_by = st.selectbox(
                "Group by (optional)", 
                ["None"] + st.session_state.categorical_features,
                index=0
            )
            
            if group_by == "None":
                fig = px.box(data, y=column)
            else:
                fig = px.box(data, x=group_by, y=column)
            
            st.plotly_chart(fig)
            
        elif viz_type == "Target Distribution":
            target = st.session_state.target_column
            
            if target in data.columns:
                if data[target].dtype == 'object' or data[target].nunique() < 10:
                    # Categorical target
                    fig = px.histogram(data, x=target, color=target)
                    st.plotly_chart(fig)
                    st.markdown(f"**Class distribution:**")
                    st.dataframe(data[target].value_counts().reset_index().rename(
                        columns={'index': target, target: 'Count'}))
                else:
                    # Numerical target
                    fig = visualizer.plot_distribution(data, target)
                    st.plotly_chart(fig)
                    
                    st.markdown(f"**Target statistics:**")
                    st.dataframe(data[target].describe().reset_index().rename(columns={'index': 'Statistic'}))
            else:
                st.warning(f"Target column '{target}' not found in the dataset.")
                
        st.markdown("""
        ### Insights from EDA
        
        Based on your EDA, consider the following:
        
        1. Are there any outliers that might affect your model?
        2. Do you need to handle missing values?
        3. Are there strong correlations between features?
        4. Is your target variable evenly distributed?
        
        These insights will help you in the next steps of preprocessing and feature engineering.
        """)

# 4. Data Preprocessing
elif page == "4. Data Preprocessing":
    st.header("4. Data Preprocessing")
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset in the Data Upload step first.")
    else:
        data = st.session_state.dataset.copy()
        target = st.session_state.target_column
        
        st.markdown("""
        Data preprocessing is a crucial step in preparing your data for modeling. It involves:
        
        1. Handling missing values
        2. Encoding categorical variables
        3. Scaling numerical features
        4. Splitting data into training and testing sets
        """)
        
        # Missing value treatment
        st.subheader("Handle Missing Values")
        
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if missing_cols:
            st.warning(f"Found missing values in columns: {', '.join(missing_cols)}")
            
            missing_strategy = st.radio(
                "Select a strategy to handle missing values:",
                ["Drop rows with missing values", "Fill missing values"]
            )
            
            if missing_strategy == "Drop rows with missing values":
                data = data.dropna()
                st.success(f"Dropped rows with missing values. New shape: {data.shape}")
            else:
                for col in missing_cols:
                    if data[col].dtype in ['int64', 'float64']:
                        fill_method = st.selectbox(
                            f"Choose fill method for numerical column '{col}':",
                            ["Mean", "Median", "Zero"]
                        )
                        
                        if fill_method == "Mean":
                            data[col] = data[col].fillna(data[col].mean())
                        elif fill_method == "Median":
                            data[col] = data[col].fillna(data[col].median())
                        else:
                            data[col] = data[col].fillna(0)
                    else:
                        fill_method = st.selectbox(
                            f"Choose fill method for categorical column '{col}':",
                            ["Most frequent", "A specific value"]
                        )
                        
                        if fill_method == "Most frequent":
                            data[col] = data[col].fillna(data[col].mode()[0])
                        else:
                            fill_value = st.text_input(f"Enter a value to fill '{col}':")
                            if fill_value:
                                data[col] = data[col].fillna(fill_value)
                
                st.success("Missing values have been filled.")
        else:
            st.success("No missing values found in your dataset!")
        
        # Categorical encoding
        st.subheader("Encode Categorical Variables")
        
        cat_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if target in cat_columns:
            cat_columns.remove(target)
        
        if cat_columns:
            st.info(f"Found categorical columns: {', '.join(cat_columns)}")
            
            encoding_method = st.radio(
                "Select an encoding method:",
                ["One-Hot Encoding", "Label Encoding"]
            )
            
            if encoding_method == "One-Hot Encoding":
                data = data_processor.one_hot_encode(data, cat_columns)
                st.success(f"Applied one-hot encoding. New shape: {data.shape}")
            else:
                data = data_processor.label_encode(data, cat_columns)
                st.success(f"Applied label encoding.")
        else:
            st.success("No categorical variables to encode (excluding target)!")
        
        # Feature scaling
        st.subheader("Scale Numerical Features")
        
        num_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target in num_columns:
            num_columns.remove(target)
        
        if num_columns:
            scaling_method = st.radio(
                "Select a scaling method:",
                ["StandardScaler (mean=0, std=1)", "MinMaxScaler (0 to 1)", "No scaling"]
            )
            
            if scaling_method != "No scaling":
                if scaling_method == "StandardScaler (mean=0, std=1)":
                    data, scaler = data_processor.standard_scale(data, num_columns)
                else:
                    data, scaler = data_processor.minmax_scale(data, num_columns)
                
                st.success("Numerical features have been scaled.")
                
                # Show comparison of before and after scaling for one feature
                if num_columns:
                    sample_col = num_columns[0]
                    original_data = st.session_state.dataset[sample_col]
                    scaled_data = data[sample_col]
                    
                    fig = visualizer.plot_scaling_comparison(original_data, scaled_data, sample_col)
                    st.plotly_chart(fig)
        else:
            st.info("No numerical features to scale (excluding target)!")
        
        # Train-test split
        st.subheader("Split Data into Training and Testing Sets")
        
        test_size = st.slider("Select test set size (%)", 10, 40, 20)
        random_state = st.number_input("Random state (for reproducibility)", value=42)
        
        if st.button("Split Data"):
            try:
                X = data.drop(target, axis=1)
                y = data[target]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.preprocessed_data = data
                
                st.success(f"Data split successfully! Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
                
                # Display class distribution for classification problems
                if st.session_state.problem_type == "Classification":
                    train_dist = pd.Series(y_train).value_counts().reset_index()
                    train_dist.columns = ['Class', 'Training Count']
                    
                    test_dist = pd.Series(y_test).value_counts().reset_index()
                    test_dist.columns = ['Class', 'Testing Count']
                    
                    dist_df = train_dist.merge(test_dist, on='Class', how='outer').fillna(0)
                    
                    st.dataframe(dist_df)
                    
                    # Check if classes are balanced
                    if len(dist_df) > 1:
                        max_count = dist_df['Training Count'].max()
                        min_count = dist_df['Training Count'].min()
                        
                        if min_count < max_count * 0.3:
                            st.warning("Your classes are imbalanced. Consider using class weights or resampling techniques during model training.")
            
            except Exception as e:
                st.error(f"Error splitting data: {str(e)}")
        
        st.markdown("""
        ### Next Steps
        
        Now that your data is preprocessed, you can move on to feature engineering to potentially improve model performance.
        """)

# 5. Feature Engineering
elif page == "5. Feature Engineering":
    st.header("5. Feature Engineering")
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset in the Data Upload step first.")
    elif 'preprocessed_data' not in st.session_state or st.session_state.preprocessed_data is None:
        st.warning("Please complete the Data Preprocessing step first.")
    else:
        st.markdown("""
        Feature engineering involves creating new features from existing ones to improve model performance.
        This can include:
        
        1. Creating interaction features
        2. Polynomial features
        3. Feature selection
        4. Dimensionality reduction
        """)
        
        data = st.session_state.preprocessed_data.copy()
        target = st.session_state.target_column
        
        # Feature transformation
        st.subheader("Feature Transformation")
        
        transform_options = st.multiselect(
            "Select transformation methods to apply:",
            ["Create polynomial features", "Apply log transformation", "Feature selection"]
        )
        
        num_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target in num_features:
            num_features.remove(target)
        
        if "Create polynomial features" in transform_options and num_features:
            poly_features = st.multiselect(
                "Select features for polynomial transformation:",
                num_features
            )
            
            if poly_features:
                degree = st.slider("Select polynomial degree", 2, 5, 2)
                
                if st.button("Create Polynomial Features"):
                    data = data_processor.create_polynomial_features(data, poly_features, degree)
                    st.success(f"Created polynomial features. New shape: {data.shape}")
        
        if "Apply log transformation" in transform_options and num_features:
            log_features = st.multiselect(
                "Select features for logarithmic transformation (positive values only):",
                num_features
            )
            
            if log_features and st.button("Apply Log Transformation"):
                data = data_processor.apply_log_transform(data, log_features)
                st.success("Applied logarithmic transformation.")
                
                # Show comparison for one feature
                if log_features:
                    sample_col = log_features[0]
                    original_col = sample_col
                    log_col = f"log_{sample_col}"
                    
                    fig = visualizer.plot_transformation_comparison(
                        data[original_col], data[log_col], 
                        original_name=original_col, 
                        transformed_name=log_col
                    )
                    st.plotly_chart(fig)
        
        # Feature selection
        if "Feature selection" in transform_options:
            st.subheader("Feature Selection")
            
            selection_method = st.radio(
                "Select a feature selection method:",
                ["Correlation-based selection", "Feature importance (requires model training)"]
            )
            
            if selection_method == "Correlation-based selection":
                corr_threshold = st.slider(
                    "Select correlation threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.8, 
                    step=0.05
                )
                
                if st.button("Apply Correlation-based Selection"):
                    reduced_data, removed_features = data_processor.correlation_feature_selection(
                        data, target, corr_threshold
                    )
                    
                    data = reduced_data
                    
                    if removed_features:
                        st.success(f"Removed {len(removed_features)} highly correlated features: {', '.join(removed_features)}")
                    else:
                        st.info("No highly correlated features found.")
                    
                    # Show correlation heatmap of remaining features
                    fig = visualizer.plot_correlation(data)
                    st.plotly_chart(fig)
            
            else:
                st.info("Feature importance will be calculated after model training.")
        
        # Update preprocessed data
        st.session_state.preprocessed_data = data
        
        # Show final feature set
        st.subheader("Final Feature Set")
        
        features = [col for col in data.columns if col != target]
        st.info(f"Total features: {len(features)}")
        
        if st.checkbox("Show all features"):
            st.write(features)
        
        st.markdown("""
        ### Next Steps
        
        With your engineered features, you're now ready to select and train machine learning models!
        """)

# 6. Model Selection & Training
elif page == "6. Model Selection & Training":
    st.header("6. Model Selection & Training")
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset in the Data Upload step first.")
    elif 'X_train' not in st.session_state or st.session_state.X_train is None:
        st.warning("Please complete the Data Preprocessing step first.")
    else:
        st.markdown("""
        Now it's time to select and train your machine learning models! 
        
        Different algorithms perform better on different types of data, so it's good practice to try multiple models.
        """)
        
        problem_type = st.session_state.problem_type
        
        # Model selection
        st.subheader("Select Models to Train")
        
        if problem_type == "Classification":
            model_options = [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Support Vector Machine (SVM)",
                "K-Nearest Neighbors (KNN)"
            ]
        else:  # Regression
            model_options = [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Support Vector Machine (SVM)",
                "K-Nearest Neighbors (KNN)"
            ]
        
        selected_models = st.multiselect(
            "Select models to train:",
            model_options,
            default=[model_options[0], model_options[2]]  # Default: first and third options
        )
        
        # Hyperparameter tuning
        st.subheader("Hyperparameter Tuning")
        
        tuning_method = st.radio(
            "Select hyperparameter tuning method:",
            ["Default parameters", "Manual tuning", "Automated tuning (Grid Search)"]
        )
        
        hyperparams = {}
        
        if tuning_method == "Manual tuning":
            st.info("Specify parameters for each selected model:")
            
            for model_name in selected_models:
                st.markdown(f"**{model_name} Parameters:**")
                
                if model_name in ["Decision Tree", "Random Forest"]:
                    max_depth = st.slider(
                        f"{model_name} - Maximum depth", 
                        min_value=1, 
                        max_value=20, 
                        value=5,
                        key=f"{model_name}_depth"
                    )
                    
                    min_samples_split = st.slider(
                        f"{model_name} - Minimum samples to split", 
                        min_value=2, 
                        max_value=20, 
                        value=2,
                        key=f"{model_name}_min_samples"
                    )
                    
                    hyperparams[model_name] = {
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split
                    }
                    
                    if model_name == "Random Forest":
                        n_estimators = st.slider(
                            f"{model_name} - Number of trees", 
                            min_value=10, 
                            max_value=200, 
                            value=100,
                            key=f"{model_name}_trees"
                        )
                        hyperparams[model_name]['n_estimators'] = n_estimators
                
                elif model_name in ["Logistic Regression", "Linear Regression"]:
                    regularization = st.selectbox(
                        f"{model_name} - Regularization type",
                        ["none", "l1", "l2", "elasticnet"],
                        key=f"{model_name}_reg_type"
                    )
                    
                    if regularization != "none":
                        C = st.slider(
                            f"{model_name} - Regularization strength (C)", 
                            min_value=0.1, 
                            max_value=10.0, 
                            value=1.0,
                            step=0.1,
                            key=f"{model_name}_C"
                        )
                        
                        hyperparams[model_name] = {
                            'penalty': regularization if regularization != "none" else None,
                            'C': C
                        }
                
                elif model_name == "Support Vector Machine (SVM)":
                    kernel = st.selectbox(
                        f"{model_name} - Kernel",
                        ["linear", "poly", "rbf", "sigmoid"],
                        key=f"{model_name}_kernel"
                    )
                    
                    C = st.slider(
                        f"{model_name} - Regularization (C)", 
                        min_value=0.1, 
                        max_value=10.0, 
                        value=1.0,
                        step=0.1,
                        key=f"{model_name}_C"
                    )
                    
                    hyperparams[model_name] = {
                        'kernel': kernel,
                        'C': C
                    }
                
                elif model_name == "K-Nearest Neighbors (KNN)":
                    n_neighbors = st.slider(
                        f"{model_name} - Number of neighbors", 
                        min_value=1, 
                        max_value=20, 
                        value=5,
                        key=f"{model_name}_neighbors"
                    )
                    
                    weights = st.selectbox(
                        f"{model_name} - Weight function",
                        ["uniform", "distance"],
                        key=f"{model_name}_weights"
                    )
                    
                    hyperparams[model_name] = {
                        'n_neighbors': n_neighbors,
                        'weights': weights
                    }
        
        elif tuning_method == "Automated tuning (Grid Search)":
            st.info("Automated hyperparameter tuning will use Grid Search to find optimal parameters.")
            
            for model_name in selected_models:
                hyperparams[model_name] = "grid_search"
        
        # Model training
        if st.button("Train Models"):
            if not selected_models:
                st.error("Please select at least one model to train.")
            else:
                # Show progress
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Train each model
                models = {}
                metrics = {}
                feature_importance = {}
                
                for i, model_name in enumerate(selected_models):
                    progress_text.text(f"Training {model_name}...")
                    
                    # Get model params based on tuning method
                    if tuning_method == "Default parameters":
                        model_params = None
                    elif tuning_method == "Manual tuning":
                        model_params = hyperparams.get(model_name, None)
                    else:  # Grid Search
                        model_params = "grid_search"
                    
                    # Train model
                    try:
                        model, model_metrics, importances = model_trainer.train_model(
                            st.session_state.X_train, 
                            st.session_state.y_train,
                            st.session_state.X_test,
                            st.session_state.y_test,
                            model_type=model_name,
                            params=model_params,
                            problem_type=problem_type
                        )
                        
                        models[model_name] = model
                        metrics[model_name] = model_metrics
                        
                        if importances is not None:
                            feature_importance[model_name] = importances
                        
                        # Update progress
                        progress = (i + 1) / len(selected_models)
                        progress_bar.progress(progress)
                    
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
                
                progress_text.text("Training complete!")
                progress_bar.progress(1.0)
                
                # Store models and metrics in session state
                st.session_state.models = models
                st.session_state.metrics = metrics
                st.session_state.feature_importance = feature_importance
                
                # Display metrics summary
                st.subheader("Model Performance Summary")
                
                if problem_type == "Classification":
                    metrics_df = pd.DataFrame({
                        'Model': list(metrics.keys()),
                        'Accuracy': [m['accuracy'] for m in metrics.values()],
                        'Precision': [m['precision'] for m in metrics.values()],
                        'Recall': [m['recall'] for m in metrics.values()],
                        'F1 Score': [m['f1'] for m in metrics.values()]
                    })
                    
                    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score']))
                    
                    # Plot metrics comparison
                    fig = visualizer.plot_model_comparison(metrics_df, 'classification')
                    st.plotly_chart(fig)
                    
                else:  # Regression
                    metrics_df = pd.DataFrame({
                        'Model': list(metrics.keys()),
                        'R² Score': [m['r2'] for m in metrics.values()],
                        'MAE': [m['mae'] for m in metrics.values()],
                        'MSE': [m['mse'] for m in metrics.values()],
                        'RMSE': [m['rmse'] for m in metrics.values()]
                    })
                    
                    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R² Score']).highlight_min(axis=0, subset=['MAE', 'MSE', 'RMSE']))
                    
                    # Plot metrics comparison
                    fig = visualizer.plot_model_comparison(metrics_df, 'regression')
                    st.plotly_chart(fig)
                
                # Feature importance
                if feature_importance:
                    st.subheader("Feature Importance")
                    
                    selected_model_for_importance = st.selectbox(
                        "Select model to view feature importance:",
                        list(feature_importance.keys())
                    )
                    
                    if selected_model_for_importance in feature_importance:
                        importances = feature_importance[selected_model_for_importance]
                        
                        # Plot feature importance
                        fig = visualizer.plot_feature_importance(importances)
                        st.plotly_chart(fig)
                
                # Save best model
                st.subheader("Save Best Model")
                
                if problem_type == "Classification":
                    best_model_name = metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'Model']
                else:
                    best_model_name = metrics_df.loc[metrics_df['R² Score'].idxmax(), 'Model']
                
                st.info(f"Best performing model: {best_model_name}")
                
                model_filename = st.text_input("Enter filename to save model (without extension):", f"model_{datetime.datetime.now().strftime('%Y%m%d')}")
                
                if st.button("Save Model"):
                    if model_filename:
                        try:
                            model_path = f"saved_models/{model_filename}.joblib"
                            joblib.dump(models[best_model_name], model_path)
                            st.session_state.best_model = models[best_model_name]
                            st.session_state.best_model_name = best_model_name
                            st.success(f"Model saved successfully as {model_path}")
                        except Exception as e:
                            st.error(f"Error saving model: {str(e)}")
                    else:
                        st.warning("Please enter a filename.")

# 7. Model Evaluation
elif page == "7. Model Evaluation":
    st.header("7. Model Evaluation")
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset in the Data Upload step first.")
    elif 'models' not in st.session_state or not st.session_state.models:
        st.warning("Please train models in the Model Selection & Training step first.")
    else:
        st.markdown("""
        Model evaluation is crucial to understand how well your model performs on unseen data.
        
        Let's analyze the performance of your trained models in detail.
        """)
        
        problem_type = st.session_state.problem_type
        models = st.session_state.models
        metrics = st.session_state.metrics
        
        # Select model to evaluate
        st.subheader("Select Model to Evaluate")
        
        model_to_evaluate = st.selectbox(
            "Choose a model:",
            list(models.keys())
        )
        
        if model_to_evaluate:
            model = models[model_to_evaluate]
            model_metrics = metrics[model_to_evaluate]
            
            st.subheader("Model Performance Metrics")
            
            if problem_type == "Classification":
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{model_metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{model_metrics['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{model_metrics['f1']:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                fig = visualizer.plot_confusion_matrix(
                    model_metrics['confusion_matrix'],
                    model_metrics.get('class_names', None)
                )
                st.plotly_chart(fig)
                
                # ROC Curve (if available)
                if 'roc_auc' in model_metrics and model_metrics['roc_auc'] is not None:
                    st.subheader("ROC Curve")
                    
                    fig = visualizer.plot_roc_curve(
                        model_metrics['fpr'],
                        model_metrics['tpr'],
                        model_metrics['roc_auc']
                    )
                    st.plotly_chart(fig)
                
                # Precision-Recall Curve (if available)
                if 'precision_curve' in model_metrics and model_metrics['precision_curve'] is not None:
                    st.subheader("Precision-Recall Curve")
                    
                    fig = visualizer.plot_precision_recall_curve(
                        model_metrics['recall_curve'],
                        model_metrics['precision_curve'],
                        model_metrics['average_precision']
                    )
                    st.plotly_chart(fig)
                
            else:  # Regression
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{model_metrics['r2']:.4f}")
                with col2:
                    st.metric("MAE", f"{model_metrics['mae']:.4f}")
                with col3:
                    st.metric("MSE", f"{model_metrics['mse']:.4f}")
                with col4:
                    st.metric("RMSE", f"{model_metrics['rmse']:.4f}")
                
                # Actual vs Predicted Plot
                st.subheader("Actual vs Predicted Values")
                
                y_test = st.session_state.y_test
                y_pred = model.predict(st.session_state.X_test)
                
                fig = visualizer.plot_actual_vs_predicted(y_test, y_pred)
                st.plotly_chart(fig)
                
                # Residuals Plot
                st.subheader("Residuals Analysis")
                
                fig = visualizer.plot_residuals(y_test, y_pred)
                st.plotly_chart(fig)
            
            # Model insights
            st.subheader("Model Insights")
            
            if model_to_evaluate in ["Decision Tree", "Random Forest"]:
                if problem_type == "Classification" and hasattr(model, 'classes_'):
                    st.info(f"Classes: {model.classes_}")
                
                if hasattr(model, 'n_features_in_'):
                    st.info(f"Number of features used: {model.n_features_in_}")
                
                if hasattr(model, 'feature_importances_') and 'feature_importance' in st.session_state:
                    st.markdown("**Feature Importance:**")
                    
                    importances = st.session_state.feature_importance.get(model_to_evaluate)
                    if importances is not None:
                        # Plot feature importance
                        fig = visualizer.plot_feature_importance(importances)
                        st.plotly_chart(fig)
            
            elif model_to_evaluate in ["Logistic Regression", "Linear Regression"]:
                if hasattr(model, 'coef_'):
                    st.markdown("**Model Coefficients:**")
                    
                    feature_names = st.session_state.X_train.columns
                    coefficients = model.coef_
                    
                    if len(coefficients.shape) > 1:
                        # Multiple classes for logistic regression
                        for i, class_coef in enumerate(coefficients):
                            class_name = model.classes_[i] if hasattr(model, 'classes_') else f"Class {i}"
                            st.write(f"Class: {class_name}")
                            
                            coef_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Coefficient': class_coef
                            }).sort_values('Coefficient', ascending=False)
                            
                            st.dataframe(coef_df)
                    else:
                        # Single set of coefficients
                        coef_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Coefficient': coefficients[0] if len(coefficients.shape) > 1 else coefficients
                        }).sort_values('Coefficient', ascending=False)
                        
                        st.dataframe(coef_df)
                
                if hasattr(model, 'intercept_'):
                    if isinstance(model.intercept_, (list, np.ndarray)) and len(model.intercept_) > 1:
                        st.info(f"Intercept values: {model.intercept_}")
                    else:
                        st.info(f"Intercept: {model.intercept_}")
            
            # Model explanation - SHAP values
            st.subheader("Model Explanation")
            
            st.markdown("""
            For a deeper understanding of your model's predictions, you could consider using:
            
            1. **SHAP (SHapley Additive exPlanations)**: Explains the output of any machine learning model
            2. **LIME (Local Interpretable Model-agnostic Explanations)**: Explains predictions of any classifier
            3. **Partial Dependence Plots**: Shows the dependence between the target and a set of features
            
            These techniques can help you understand which features have the biggest impact on predictions.
            """)
            
            # Cross-validation results
            if 'cv_scores' in model_metrics:
                st.subheader("Cross-Validation Results")
                
                cv_scores = model_metrics['cv_scores']
                cv_mean = model_metrics['cv_mean']
                cv_std = model_metrics['cv_std']
                
                st.info(f"Mean CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
                
                fig = visualizer.plot_cv_results(cv_scores)
                st.plotly_chart(fig)
            
            # Set as best model
            if st.button("Set as Best Model"):
                st.session_state.best_model = model
                st.session_state.best_model_name = model_to_evaluate
                st.success(f"{model_to_evaluate} set as the best model for predictions.")

# 8. Make Predictions
elif page == "8. Make Predictions":
    st.header("8. Make Predictions")
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset in the Data Upload step first.")
    elif 'models' not in st.session_state or not st.session_state.models:
        st.warning("Please train models in the Model Selection & Training step first.")
    else:
        st.markdown("""
        Now it's time to use your trained model to make predictions on new data!
        
        You can either:
        1. Make predictions on the test set
        2. Make predictions on a new dataset
        3. Enter custom values for prediction
        """)
        
        # Choose prediction method
        pred_method = st.radio(
            "Select prediction method:",
            ["Predict on test set", "Upload new dataset", "Enter custom values"]
        )
        
        # Get the model to use
        if 'best_model' in st.session_state and st.session_state.best_model is not None:
            default_model = st.session_state.best_model_name
        else:
            default_model = list(st.session_state.models.keys())[0]
        
        model_for_prediction = st.selectbox(
            "Select model for prediction:",
            list(st.session_state.models.keys()),
            index=list(st.session_state.models.keys()).index(default_model)
        )
        
        model = st.session_state.models[model_for_prediction]
        problem_type = st.session_state.problem_type
        
        # Make predictions based on method
        if pred_method == "Predict on test set":
            if st.button("Generate Predictions on Test Set"):
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Create dataframe with results
                results_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                
                # Add prediction probabilities for classification
                if problem_type == "Classification" and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    
                    # Add class probabilities
                    for i, class_name in enumerate(model.classes_):
                        results_df[f"Probability (Class {class_name})"] = y_proba[:, i]
                
                # Store predictions
                st.session_state.predictions = results_df
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results_df.head(20))
                
                # Show evaluation metrics
                if problem_type == "Classification":
                    from sklearn.metrics import accuracy_score, classification_report
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    st.metric("Accuracy on Test Set", f"{accuracy:.4f}")
                    st.table(report_df)
                    
                    # Show confusion matrix
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig = visualizer.plot_confusion_matrix(cm, model.classes_ if hasattr(model, 'classes_') else None)
                    st.plotly_chart(fig)
                    
                else:  # Regression
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                    
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    st.metric("R² Score on Test Set", f"{r2:.4f}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{mae:.4f}")
                    with col2:
                        st.metric("MSE", f"{mse:.4f}")
                    with col3:
                        st.metric("RMSE", f"{rmse:.4f}")
                    
                    # Show actual vs predicted plot
                    fig = visualizer.plot_actual_vs_predicted(y_test, y_pred)
                    st.plotly_chart(fig)
                
                # Option to download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
        
        elif pred_method == "Upload new dataset":
            st.subheader("Upload New Dataset for Prediction")
            
            st.markdown("""
            Upload a new dataset for prediction. The dataset should have the same feature columns as your training data.
            """)
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="new_data")
            
            if uploaded_file is not None:
                try:
                    # Load and preprocess new data
                    new_data = pd.read_csv(uploaded_file)
                    
                    # Check if dataset has the required columns
                    required_features = st.session_state.X_train.columns.tolist()
                    missing_features = [f for f in required_features if f not in new_data.columns]
                    
                    if missing_features:
                        st.error(f"New dataset is missing required features: {', '.join(missing_features)}")
                    else:
                        st.success("Dataset loaded successfully!")
                        st.dataframe(new_data.head())
                        
                        # Match the format of training data
                        X_new = new_data[required_features]
                        
                        # Check for target column
                        target_col = st.session_state.target_column
                        has_target = target_col in new_data.columns
                        
                        if st.button("Generate Predictions"):
                            # Make predictions
                            y_pred = model.predict(X_new)
                            
                            # Create results dataframe
                            if has_target:
                                results_df = pd.DataFrame({
                                    'Actual': new_data[target_col],
                                    'Predicted': y_pred
                                })
                            else:
                                results_df = pd.DataFrame({
                                    'Predicted': y_pred
                                })
                            
                            # Add prediction probabilities for classification
                            if problem_type == "Classification" and hasattr(model, 'predict_proba'):
                                y_proba = model.predict_proba(X_new)
                                
                                # Add class probabilities
                                for i, class_name in enumerate(model.classes_):
                                    results_df[f"Probability (Class {class_name})"] = y_proba[:, i]
                            
                            # Store predictions
                            st.session_state.predictions = results_df
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(results_df)
                            
                            # Show evaluation metrics if target column is available
                            if has_target and problem_type == "Classification":
                                from sklearn.metrics import accuracy_score, classification_report
                                
                                accuracy = accuracy_score(new_data[target_col], y_pred)
                                report = classification_report(new_data[target_col], y_pred, output_dict=True)
                                report_df = pd.DataFrame(report).transpose()
                                
                                st.metric("Accuracy on New Data", f"{accuracy:.4f}")
                                st.table(report_df)
                                
                            elif has_target and problem_type == "Regression":
                                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                                
                                r2 = r2_score(new_data[target_col], y_pred)
                                mae = mean_absolute_error(new_data[target_col], y_pred)
                                mse = mean_squared_error(new_data[target_col], y_pred)
                                rmse = np.sqrt(mse)
                                
                                st.metric("R² Score on New Data", f"{r2:.4f}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("MAE", f"{mae:.4f}")
                                with col2:
                                    st.metric("MSE", f"{mse:.4f}")
                                with col3:
                                    st.metric("RMSE", f"{rmse:.4f}")
                            
                            # Option to download predictions
                            full_results = new_data.copy()
                            full_results['Predicted'] = y_pred
                            
                            csv = full_results.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="prediction_results.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"Error processing the new dataset: {str(e)}")
        
        else:  # Custom values
            st.subheader("Enter Custom Values for Prediction")
            
            st.markdown("""
            Enter values for each feature manually to get a prediction for a single instance.
            """)
            
            # Get feature input fields
            feature_values = {}
            features = st.session_state.X_train.columns.tolist()
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(features):
                # Determine data type and range for numeric features
                if feature in st.session_state.numerical_features:
                    min_val = float(st.session_state.X_train[feature].min())
                    max_val = float(st.session_state.X_train[feature].max())
                    mean_val = float(st.session_state.X_train[feature].mean())
                    
                    with col1 if i % 2 == 0 else col2:
                        feature_values[feature] = st.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val - min_val) / 100
                        )
                else:
                    # For categorical features, show a selectbox
                    unique_values = st.session_state.X_train[feature].unique().tolist()
                    
                    with col1 if i % 2 == 0 else col2:
                        feature_values[feature] = st.selectbox(
                            f"{feature}",
                            unique_values,
                            index=0
                        )
            
            if st.button("Generate Prediction"):
                # Create a DataFrame with the input values
                input_df = pd.DataFrame([feature_values])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display prediction
                st.subheader("Prediction Result")
                
                if problem_type == "Classification":
                    st.success(f"Predicted Class: {prediction}")
                    
                    # Show probabilities if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_df)[0]
                        
                        proba_df = pd.DataFrame({
                            'Class': model.classes_,
                            'Probability': proba
                        })
                        
                        # Display probability bar chart
                        fig = px.bar(
                            proba_df, 
                            x='Class', 
                            y='Probability',
                            text='Probability',
                            text_auto='.2f',
                            title='Prediction Probabilities',
                            color='Probability'
                        )
                        st.plotly_chart(fig)
                        
                else:  # Regression
                    st.success(f"Predicted Value: {prediction:.4f}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About this Platform
This is a beginner-friendly ML model experimentation platform that guides you through
the entire machine learning workflow from data upload to model evaluation and prediction.

Created with Streamlit, Scikit-learn, Pandas, and Plotly.
""")
