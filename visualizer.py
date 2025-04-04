import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

class visualizer:
    """
    Class for creating visualizations for data analysis and model evaluation
    """
    
    def plot_distribution(self, data, column):
        """
        Plot histogram and KDE for a numerical column
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data[column],
                name="Histogram",
                opacity=0.7,
                nbinsx=30
            )
        )
        
        # Add density curve if there are enough data points
        if len(data) > 10:
            # Calculate KDE manually
            kde = gaussian_kde(data[column].dropna())
            x_range = np.linspace(data[column].min(), data[column].max(), 1000)
            y_kde = kde(x_range)
            
            # Scale the KDE to match the histogram
            hist, bin_edges = np.histogram(data[column].dropna(), bins=30)
            scaling_factor = max(hist) / max(y_kde)
            y_kde_scaled = y_kde * scaling_factor
            
            # Add KDE curve
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde_scaled,
                    mode="lines",
                    name="Density",
                    line=dict(color="red", width=2)
                ),
                secondary_y=False
            )
        
        # Add vertical line for mean and median
        fig.add_vline(
            x=data[column].mean(),
            line_dash="dash",
            line_color="green",
            annotation_text="Mean",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=data[column].median(),
            line_dash="dash",
            line_color="orange",
            annotation_text="Median",
            annotation_position="top left"
        )
        
        # Update layout
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def plot_correlation(self, data):
        """
        Plot correlation heatmap
        
        Args:
            data: DataFrame to calculate correlations
            
        Returns:
            Plotly figure
        """
        # Calculate correlation matrix
        corr = data.select_dtypes(include=['int64', 'float64']).corr()
        
        # Create heatmap
        fig = px.imshow(
            corr.values,
            x=corr.columns,
            y=corr.index,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        
        # Add correlation values as text
        for i in range(len(corr.columns)):
            for j in range(len(corr.index)):
                fig.add_annotation(
                    x=i,
                    y=j,
                    text=f"{corr.values[j, i]:.2f}",
                    showarrow=False,
                    font=dict(
                        color="black" if abs(corr.values[j, i]) < 0.7 else "white"
                    )
                )
        
        # Update layout
        fig.update_layout(
            height=600,
            width=700,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            template="plotly_white"
        )
        
        return fig
    
    def plot_scaling_comparison(self, original_data, scaled_data, feature_name):
        """
        Plot comparison of original and scaled data
        
        Args:
            original_data: Series of original values
            scaled_data: Series of scaled values
            feature_name: Name of the feature
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(rows=1, cols=2)
        
        # Original data histogram
        fig.add_trace(
            go.Histogram(
                x=original_data,
                name="Original",
                opacity=0.7,
                marker_color="blue"
            ),
            row=1, col=1
        )
        
        # Scaled data histogram
        fig.add_trace(
            go.Histogram(
                x=scaled_data,
                name="Scaled",
                opacity=0.7,
                marker_color="red"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Comparison of Original vs Scaled Data for {feature_name}",
            height=400,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Original Values", row=1, col=1)
        fig.update_xaxes(title_text="Scaled Values", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        return fig
    
    def plot_transformation_comparison(self, original_data, transformed_data, original_name, transformed_name):
        """
        Plot comparison of original and transformed data
        
        Args:
            original_data: Series of original values
            transformed_data: Series of transformed values
            original_name: Name of the original feature
            transformed_name: Name of the transformed feature
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(rows=1, cols=2)
        
        # Original data histogram
        fig.add_trace(
            go.Histogram(
                x=original_data,
                name="Original",
                opacity=0.7,
                marker_color="blue"
            ),
            row=1, col=1
        )
        
        # Transformed data histogram
        fig.add_trace(
            go.Histogram(
                x=transformed_data,
                name="Transformed",
                opacity=0.7,
                marker_color="red"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Comparison of Original vs Transformed Data",
            height=400,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text=original_name, row=1, col=1)
        fig.update_xaxes(title_text=transformed_name, row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        return fig
    
    def plot_confusion_matrix(self, conf_matrix, class_names=None):
        """
        Plot confusion matrix
        
        Args:
            conf_matrix: Confusion matrix array
            class_names: List of class names
            
        Returns:
            Plotly figure
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(conf_matrix))]
        
        # Create heatmap
        fig = px.imshow(
            conf_matrix,
            x=class_names,
            y=class_names,
            color_continuous_scale="Blues",
            zmin=0,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500,
            width=600,
            template="plotly_white"
        )
        
        return fig
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """
        Plot ROC curve
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under ROC curve
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC curve (AUC = {roc_auc:.3f})",
                line=dict(color="darkorange", width=2)
            )
        )
        
        # Add diagonal line (random classifier)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random (AUC = 0.5)",
                line=dict(color="navy", width=2, dash="dash")
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1], constrain="domain"),
            yaxis=dict(range=[0, 1], constrain="domain", scaleanchor="x", scaleratio=1),
            height=500,
            width=600,
            legend=dict(x=0.6, y=0.1),
            template="plotly_white"
        )
        
        return fig
    
    def plot_precision_recall_curve(self, recall, precision, average_precision):
        """
        Plot precision-recall curve
        
        Args:
            recall: Recall values
            precision: Precision values
            average_precision: Average precision score
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add precision-recall curve
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"Precision-Recall (AP = {average_precision:.3f})",
                line=dict(color="blue", width=2)
            )
        )
        
        # Add a horizontal line for random classifier
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0.5, 0.5],  # Assuming balanced classes
                mode="lines",
                name="Random",
                line=dict(color="red", width=2, dash="dash")
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1], constrain="domain"),
            yaxis=dict(range=[0, 1.05], constrain="domain"),
            height=500,
            width=600,
            legend=dict(x=0.6, y=0.1),
            template="plotly_white"
        )
        
        return fig
    
    def plot_model_comparison(self, metrics_df, problem_type="classification"):
        """
        Plot comparison of model metrics
        
        Args:
            metrics_df: DataFrame containing model metrics
            problem_type: 'classification' or 'regression'
            
        Returns:
            Plotly figure
        """
        if problem_type == "classification":
            metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        else:
            metric_cols = ['R² Score', 'MAE', 'MSE', 'RMSE']
        
        # Melt the DataFrame for easier plotting
        melted_df = pd.melt(
            metrics_df,
            id_vars=["Model"],
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Value"
        )
        
        # Create the figure
        fig = px.bar(
            melted_df,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            title="Model Performance Comparison",
            height=500,
            width=700,
            template="plotly_white"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Score",
            legend_title="Metric"
        )
        
        return fig
    
    def plot_feature_importance(self, importance_dict):
        """
        Plot feature importance
        
        Args:
            importance_dict: Dictionary of feature names and importance values
            
        Returns:
            Plotly figure
        """
        # Convert dictionary to DataFrame
        df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        
        # Sort by importance
        df = df.sort_values('Importance', ascending=False)
        
        # Take top 20 features if there are many
        if len(df) > 20:
            df = df.head(20)
        
        # Create the figure
        fig = px.bar(
            df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance",
            height=600,
            width=700,
            template="plotly_white"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
            yaxis=dict(autorange="reversed")  # Highest importance on top
        )
        
        return fig
    
    def plot_actual_vs_predicted(self, y_true, y_pred):
        """
        Plot actual vs predicted values for regression
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
            
        Returns:
            Plotly figure
        """
        # Create DataFrame
        df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        # Create figure
        fig = px.scatter(
            df,
            x='Actual',
            y='Predicted',
            title="Actual vs Predicted Values",
            height=500,
            width=600,
            template="plotly_white"
        )
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", width=2, dash="dash")
            )
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values"
        )
        
        return fig
    
    def plot_residuals(self, y_true, y_pred):
        """
        Plot residuals for regression
        
        Args:
            y_true: Actual target values
            y_pred: Predicted target values
            
        Returns:
            Plotly figure
        """
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create subplots
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Residuals vs Predicted", "Residual Distribution"))
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(color="blue", opacity=0.6),
                name="Residuals"
            ),
            row=1, col=1
        )
        
        # Add horizontal line at y=0
        fig.add_trace(
            go.Scatter(
                x=[min(y_pred), max(y_pred)],
                y=[0, 0],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Residual histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                marker_color="green",
                name="Residual Distribution"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Residual Analysis",
            height=400,
            width=900,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        return fig
    
    def plot_cv_results(self, cv_scores):
        """
        Plot cross-validation results
        
        Args:
            cv_scores: Array of cross-validation scores
            
        Returns:
            Plotly figure
        """
        # Create figure
        fig = go.Figure()
        
        # Add bar chart for CV scores
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(cv_scores) + 1)),
                y=cv_scores,
                marker_color="blue",
                name="CV Scores"
            )
        )
        
        # Add line for mean score
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cv_scores) + 1)),
                y=[cv_scores.mean()] * len(cv_scores),
                mode="lines",
                name=f"Mean Score: {cv_scores.mean():.3f}",
                line=dict(color="red", width=2, dash="dash")
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Cross-Validation Results",
            xaxis_title="Fold",
            yaxis_title="Score",
            height=400,
            width=600,
            template="plotly_white"
        )
        
        return fig