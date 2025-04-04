import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score

class ModelTrainer:
    """
    Class for training and evaluating machine learning models
    """
    
    def get_model(self, model_type, params=None, problem_type="Classification"):
        """
        Get model instance based on model type and parameters
        
        Args:
            model_type: String name of model type
            params: Dictionary of model parameters
            problem_type: 'Classification' or 'Regression'
            
        Returns:
            Model instance
        """
        # Set empty params if None
        if params is None:
            params = {}
        
        # Classification models
        if problem_type == "Classification":
            if model_type == "Logistic Regression":
                return LogisticRegression(**params, max_iter=1000, random_state=42)
            elif model_type == "Decision Tree":
                return DecisionTreeClassifier(**params, random_state=42)
            elif model_type == "Random Forest":
                return RandomForestClassifier(**params, random_state=42)
            elif model_type == "Support Vector Machine (SVM)":
                return SVC(**params, probability=True, random_state=42)
            elif model_type == "K-Nearest Neighbors (KNN)":
                return KNeighborsClassifier(**params)
            else:
                raise ValueError(f"Unsupported classification model type: {model_type}")
        
        # Regression models
        else:
            if model_type == "Linear Regression":
                return LinearRegression(**params)
            elif model_type == "Decision Tree":
                return DecisionTreeRegressor(**params, random_state=42)
            elif model_type == "Random Forest":
                return RandomForestRegressor(**params, random_state=42)
            elif model_type == "Support Vector Machine (SVM)":
                return SVR(**params)
            elif model_type == "K-Nearest Neighbors (KNN)":
                return KNeighborsRegressor(**params)
            else:
                raise ValueError(f"Unsupported regression model type: {model_type}")
    
    def get_grid_search_params(self, model_type, problem_type="Classification"):
        """
        Get grid search parameters for different model types
        
        Args:
            model_type: String name of model type
            problem_type: 'Classification' or 'Regression'
            
        Returns:
            Dictionary of parameter grid for GridSearchCV
        """
        # Classification model parameters
        if problem_type == "Classification":
            if model_type == "Logistic Regression":
                return {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                }
            elif model_type == "Decision Tree":
                return {
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type == "Random Forest":
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == "Support Vector Machine (SVM)":
                return {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                }
            elif model_type == "K-Nearest Neighbors (KNN)":
                return {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # Manhattan or Euclidean distance
                }
        
        # Regression model parameters
        else:
            if model_type == "Linear Regression":
                return {}  # Linear regression has no hyperparameters to tune
            elif model_type == "Decision Tree":
                return {
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type == "Random Forest":
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == "Support Vector Machine (SVM)":
                return {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                }
            elif model_type == "K-Nearest Neighbors (KNN)":
                return {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # Manhattan or Euclidean distance
                }
        
        return {}
    
    def evaluate_classification_model(self, model, X_test, y_test):
        """
        Evaluate a classification model and get performance metrics
        
        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Get class names if available
        if hasattr(model, 'classes_'):
            metrics['class_names'] = model.classes_
        
        # For binary classification, calculate ROC and PR curves
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
            metrics['roc_auc'] = auc(fpr, tpr)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            metrics['precision_curve'] = precision
            metrics['recall_curve'] = recall
            metrics['average_precision'] = average_precision_score(y_test, y_prob)
        else:
            metrics['roc_auc'] = None
            metrics['precision_curve'] = None
        
        return metrics
    
    def evaluate_regression_model(self, model, X_test, y_test):
        """
        Evaluate a regression model and get performance metrics
        
        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
        
        return metrics
    
    def get_feature_importance(self, model, feature_names):
        """
        Get feature importance from model if available
        
        Args:
            model: Trained model instance
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature names and importance values, or None if not available
        """
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance_values = model.feature_importances_
            return dict(zip(feature_names, importance_values))
        
        elif hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_.shape) > 1:
                # For multiclass models, take the mean of coefficients
                importance_values = np.abs(model.coef_).mean(axis=0)
            else:
                importance_values = np.abs(model.coef_)
                
            return dict(zip(feature_names, importance_values))
        
        return None
    
    def train_model(self, X_train, y_train, X_test, y_test, model_type, params=None, problem_type="Classification"):
        """
        Train a model with given data and parameters
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_type: String name of model type
            params: Dictionary of model parameters or "grid_search" for automated tuning
            problem_type: 'Classification' or 'Regression'
            
        Returns:
            Tuple of (trained model, evaluation metrics, feature importance)
        """
        # Get feature names
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # If grid search is requested
        if params == "grid_search":
            param_grid = self.get_grid_search_params(model_type, problem_type)
            
            # Get base model
            base_model = self.get_model(model_type, {}, problem_type)
            
            # Create grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy' if problem_type == 'Classification' else 'r2',
                n_jobs=-1 if model_type not in ["Support Vector Machine (SVM)"] else 1,  # Use all cores except for SVM
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model
            model = grid_search.best_estimator_
            
            # Print best parameters
            print(f"Best parameters for {model_type}: {grid_search.best_params_}")
        else:
            # Get model with provided parameters
            model = self.get_model(model_type, params, problem_type)
            
            # Train model
            model.fit(X_train, y_train)
        
        # Evaluate model
        if problem_type == "Classification":
            metrics = self.evaluate_classification_model(model, X_test, y_test)
        else:
            metrics = self.evaluate_regression_model(model, X_test, y_test)
        
        # Cross-validation
        cv_score = 'accuracy' if problem_type == 'Classification' else 'r2'
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=cv_score)
        
        metrics['cv_scores'] = cv_scores
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Get feature importance
        feature_importance = self.get_feature_importance(model, feature_names)
        
        return model, metrics, feature_importance
