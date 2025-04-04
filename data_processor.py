import pandas as pd
import numpy as np
from io import StringIO
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder

class DataProcessor:
    """
    Class for handling data preprocessing operations
    """
    
    def get_info_buffer(self, data):
        """
        Get dataset information as text
        
        Args:
            data: DataFrame to get info for
            
        Returns:
            String containing dataset info
        """
        buffer = StringIO()
        data.info(buf=buffer)
        return buffer.getvalue()
    
    def one_hot_encode(self, data, columns):
        """
        Apply one-hot encoding to categorical columns
        
        Args:
            data: DataFrame to encode
            columns: List of categorical columns to encode
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # One-hot encode each categorical column
        for col in columns:
            if col in df.columns:
                # Get one-hot encoded columns
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                
                # Drop the original column and join the encoded ones
                df = df.drop(col, axis=1)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def label_encode(self, data, columns):
        """
        Apply label encoding to categorical columns
        
        Args:
            data: DataFrame to encode
            columns: List of categorical columns to encode
            
        Returns:
            DataFrame with label encoded columns
        """
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Label encode each categorical column
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def standard_scale(self, data, columns):
        """
        Apply standard scaling to numerical columns
        
        Args:
            data: DataFrame to scale
            columns: List of numerical columns to scale
            
        Returns:
            Tuple of (scaled DataFrame, scaler object)
        """
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Scale only specified columns
        df[columns] = scaler.fit_transform(df[columns])
        
        return df, scaler
    
    def minmax_scale(self, data, columns):
        """
        Apply min-max scaling to numerical columns
        
        Args:
            data: DataFrame to scale
            columns: List of numerical columns to scale
            
        Returns:
            Tuple of (scaled DataFrame, scaler object)
        """
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Initialize scaler
        scaler = MinMaxScaler()
        
        # Scale only specified columns
        df[columns] = scaler.fit_transform(df[columns])
        
        return df, scaler
    
    def create_polynomial_features(self, data, columns, degree=2):
        """
        Create polynomial features from selected columns
        
        Args:
            data: DataFrame to process
            columns: List of columns to create polynomial features from
            degree: Polynomial degree
            
        Returns:
            DataFrame with added polynomial features
        """
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Initialize polynomial feature transformer
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Create polynomial features
        poly_features = poly.fit_transform(df[columns])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names)
        
        # Remove the original features that are duplicated in the polynomial features
        poly_df = poly_df.drop(columns, axis=1)
        
        # Concatenate with original DataFrame
        result = pd.concat([df, poly_df], axis=1)
        
        return result
    
    def apply_log_transform(self, data, columns):
        """
        Apply logarithmic transformation to selected columns
        
        Args:
            data: DataFrame to process
            columns: List of columns to apply log transform
            
        Returns:
            DataFrame with added log-transformed features
        """
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Apply log transformation (add small constant to avoid log(0))
        for col in columns:
            if col in df.columns:
                # Add small constant to handle zeros
                min_val = df[col].min()
                offset = 0 if min_val > 0 else abs(min_val) + 1e-6
                
                # Create new log-transformed column
                df[f"log_{col}"] = np.log(df[col] + offset)
        
        return df
    
    def correlation_feature_selection(self, data, target, threshold=0.8):
        """
        Remove highly correlated features
        
        Args:
            data: DataFrame to process
            target: Target column name
            threshold: Correlation threshold for removing features
            
        Returns:
            Tuple of (reduced DataFrame, list of removed features)
        """
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Calculate feature correlations
        corr_matrix = df.corr().abs()
        
        # Create a mask to ignore self-correlations and target correlations
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        if target in corr_matrix.columns:
            mask[:, corr_matrix.columns.get_loc(target)] = False
        
        # Find features with correlation greater than threshold
        features_to_drop = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if mask[i, j] and corr_matrix.iloc[i, j] > threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    
                    # If target is in the DataFrame, drop the one with lower correlation to target
                    if target in df.columns:
                        corr_i = abs(df[col_i].corr(df[target]))
                        corr_j = abs(df[col_j].corr(df[target]))
                        
                        drop_col = col_i if corr_i < corr_j else col_j
                    else:
                        # If no target, drop the second column by default
                        drop_col = col_j
                    
                    if drop_col not in features_to_drop:
                        features_to_drop.append(drop_col)
        
        # Drop the selected features
        df_reduced = df.drop(features_to_drop, axis=1)
        
        return df_reduced, features_to_drop
