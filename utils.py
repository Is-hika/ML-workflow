import pandas as pd
import numpy as np

def get_dataset_description(dataset_name):
    """
    Get description for common datasets
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        String with dataset description
    """
    descriptions = {
        "iris": """
        The Iris dataset is a classic dataset for classification problems. It contains 3 classes of 50 instances each, 
        where each class refers to a type of iris plant. Each instance has 4 features: sepal length, sepal width, 
        petal length, and petal width.
        
        Problem Type: Classification (3 classes)
        Features: 4 numerical features
        Instances: 150
        """,
        
        "boston": """
        The Boston Housing dataset contains information about different houses in Boston. The goal is to predict 
        the value of houses in Boston suburbs.
        
        Problem Type: Regression
        Features: 13 numerical features
        Instances: 506
        """,
        
        "diabetes": """
        The Diabetes dataset consists of 10 baseline variables, age, sex, body mass index, average blood pressure, 
        and six blood serum measurements for 442 diabetes patients, as well as the response of interest, a quantitative 
        measure of disease progression one year after baseline.
        
        Problem Type: Regression
        Features: 10 numerical features
        Instances: 442
        """,
        
        "wine": """
        The Wine dataset is a classic dataset for classification problems. It contains 3 classes of wine, with 
        59, 71, and 48 instances respectively, for a total of 178 instances. Each instance has 13 features 
        derived from a chemical analysis of wines grown in the same region in Italy but derived from three 
        different cultivars.
        
        Problem Type: Classification (3 classes)
        Features: 13 numerical features
        Instances: 178
        """,
        
        "breast_cancer": """
        The Breast Cancer Wisconsin dataset contains features computed from a digitized image of a fine needle 
        aspirate (FNA) of a breast mass. The goal is to classify whether the mass is benign or malignant.
        
        Problem Type: Binary Classification
        Features: 30 numerical features
        Instances: 569
        """
    }
    
    return descriptions.get(dataset_name, "No description available for this dataset.")

def get_model_description(model_name):
    """
    Get description for common machine learning models
    
    Args:
        model_name: Name of the model
        
    Returns:
        String with model description
    """
    descriptions = {
        "Logistic Regression": """
        Logistic Regression is a statistical model that uses a logistic function to model a binary dependent variable. 
        In spite of its name, it's a classification algorithm rather than a regression algorithm.
        
        Strengths:
        - Simple and interpretable
        - Works well with linearly separable classes
        - Provides probability scores
        - Less prone to overfitting in high-dimensional spaces
        
        Weaknesses:
        - Assumes linear decision boundary
        - May underperform with complex relationships
        - Requires feature scaling
        
        Hyperparameters:
        - C: Inverse of regularization strength
        - penalty: Type of regularization ('l1', 'l2', 'elasticnet', 'none')
        - solver: Algorithm for optimization
        """,
        
        "Linear Regression": """
        Linear Regression is a linear approach to modeling the relationship between a dependent variable and one or more 
        independent variables. It assumes a linear relationship between variables.
        
        Strengths:
        - Simple and interpretable
        - Fast to train
        - Works well when relationship is linear
        
        Weaknesses:
        - Assumes linear relationship
        - Sensitive to outliers
        - Can't capture complex patterns
        
        Hyperparameters:
        - fit_intercept: Whether to calculate the intercept
        - normalize: Whether to normalize features
        """,
        
        "Decision Tree": """
        Decision Tree is a non-parametric supervised learning algorithm that creates a model that predicts the value of a 
        target variable by learning simple decision rules inferred from the data features.
        
        Strengths:
        - Simple to understand and interpret
        - Requires little data preprocessing
        - Can handle numerical and categorical data
        - Can model non-linear relationships
        
        Weaknesses:
        - Can create overly complex trees that don't generalize well
        - Unstable (small variations in data can result in a completely different tree)
        - Biased toward features with more levels
        
        Hyperparameters:
        - max_depth: Maximum depth of tree
        - min_samples_split: Minimum samples required to split a node
        - min_samples_leaf: Minimum samples required at a leaf node
        """,
        
        "Random Forest": """
        Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs 
        the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
        
        Strengths:
        - Reduces overfitting compared to decision trees
        - Works well with large datasets
        - Can handle thousands of input variables
        - Provides feature importance
        
        Weaknesses:
        - Less interpretable than a single decision tree
        - Slower training time
        - May still overfit on noisy datasets
        
        Hyperparameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of trees
        - min_samples_split: Minimum samples required to split a node
        - min_samples_leaf: Minimum samples required at a leaf node
        """,
        
        "Support Vector Machine (SVM)": """
        Support Vector Machine (SVM) is a supervised learning algorithm that finds the hyperplane that best separates 
        classes in a high-dimensional space. It works by finding the hyperplane that maximizes the margin between classes.
        
        Strengths:
        - Effective in high-dimensional spaces
        - Works well when classes are separable
        - Memory efficient as it uses a subset of training points
        - Versatile through different kernel functions
        
        Weaknesses:
        - Does not directly provide probability estimates
        - Can be computationally intensive with large datasets
        - Requires feature scaling
        - Hyperparameter tuning can be challenging
        
        Hyperparameters:
        - C: Regularization parameter
        - kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        - gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        """,
        
        "K-Nearest Neighbors (KNN)": """
        K-Nearest Neighbors (KNN) is a non-parametric, lazy learning algorithm that uses a database of examples to predict 
        the classification or regression of a new data point. It works by finding the k nearest examples to the new data point 
        and taking a majority vote (for classification) or average (for regression) of their values.
        
        Strengths:
        - Simple to implement
        - No training phase (lazy learning)
        - Can naturally handle multi-class problems
        - Adaptable to different similarity metrics
        
        Weaknesses:
        - Computationally intensive during prediction
        - Requires feature scaling
        - Sensitive to irrelevant features
        - Memory-intensive as it stores all training examples
        
        Hyperparameters:
        - n_neighbors: Number of neighbors to use
        - weights: Weight function ('uniform', 'distance')
        - p: Power parameter for Minkowski metric (p=1 for Manhattan, p=2 for Euclidean)
        """
    }
    
    return descriptions.get(model_name, "No description available for this model.")