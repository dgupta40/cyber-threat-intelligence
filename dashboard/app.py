"""
AI-Driven Cyber Threat Intelligence Dashboard
Incorporating both TF-IDF+LR and SBERT+LightGBM model predictions
"""

import os
import sys
import logging
import json
import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_from_json

# ===============================================================================
# Constants and Config
# ===============================================================================
PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
METRICS_DIR = Path("metrics")
LOG_DIR = Path("logs")

# ===============================================================================
# Dashboard Component Functions
# ===============================================================================

def create_kpi_cards(df_filtered):
    """Create KPI metric cards for the dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Threats",
            value=len(df_filtered),
            delta=f"+{len(df_filtered) // 10}" if len(df_filtered) > 0 else "0"
        )
    
    with col2:
        if 'severity_bin' in df_filtered.columns:
            high_risk_count = len(df_filtered[df_filtered['severity_bin'].isin(['high', 'critical'])])
            st.metric(
                label="High/Critical Threats",
                value=high_risk_count,
                delta=f"{high_risk_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
            )
        else:
            st.metric(label="High/Critical Threats", value="N/A")
    
    with col3:
        if 'is_anomaly' in df_filtered.columns:
            anomaly_count = len(df_filtered[df_filtered['is_anomaly'] == True])
            st.metric(
                label="Anomalies Detected",
                value=anomaly_count,
                delta=f"{anomaly_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
            )
        else:
            st.metric(label="Anomalies Detected", value="N/A")
    
    with col4:
        # Get the most recent date safely
        date_str = "N/A"
        if 'date' in df_filtered.columns and not df_filtered.empty:
            try:
                # Try to convert to datetime first if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
                
                # Filter out NaT values
                valid_dates = df_filtered['date'].dropna()
                if not valid_dates.empty:
                    most_recent = valid_dates.max()
                    date_str = most_recent.strftime("%Y-%m-%d") if isinstance(most_recent, pd.Timestamp) else str(most_recent)
            except Exception as e:
                date_str = "Error processing dates"
                
        st.metric(
            label="Last Updated",
            value=date_str
        )

def create_threat_timeline(df_filtered):
    """Create an enhanced timeline visualization of threats."""
    st.subheader("Threat Timeline")
    
    if 'date' in df_filtered.columns and not df_filtered.empty:
        try:
            # Try to convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
            
            # Filter out invalid dates
            df_valid = df_filtered.dropna(subset=['date'])
            
            if df_valid.empty:
                st.info("No valid date information available")
                return
            
            # Group by date
            df_time = df_valid.groupby(df_valid['date'].dt.date).size().reset_index(name='count')
            
            # Calculate 7-day moving average if enough data
            if len(df_time) > 7:
                df_time['7d_avg'] = df_time['count'].rolling(window=7, min_periods=1).mean()
            
            # Create plot
            fig = px.line(
                df_time, 
                x='date', 
                y='count',
                title='Threat Count by Date',
                labels={'date': 'Date', 'count': 'Number of Threats'}
            )
            
            # Add severity category lines if available
            if 'severity_bin' in df_valid.columns:
                # Group by date and severity
                df_sev_time = df_valid.groupby([df_valid['date'].dt.date, 'severity_bin']).size().reset_index(name='count')
                
                # Create a multi-line plot by severity
                fig = px.line(
                    df_sev_time, 
                    x='date', 
                    y='count',
                    color='severity_bin',
                    title='Threats by Severity Over Time',
                    labels={'date': 'Date', 'count': 'Number of Threats', 'severity_bin': 'Severity'},
                    color_discrete_map={
                        'critical': 'red',
                        'high': 'orange',
                        'medium': 'yellow',
                        'low': 'green',
                        'unknown': 'gray'
                    }
                )
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Threats",
                legend_title="Severity" if 'severity_bin' in df_valid.columns else None,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating timeline visualization: {e}")
    else:
        st.info("Date information not available for time series visualization")

def create_severity_distribution(df_filtered):
    """Create enhanced severity distribution visualizations."""
    st.subheader("Severity Distribution Analysis")
    
    if 'severity_bin' in df_filtered.columns:
        # Create columns for side-by-side visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity level distribution
            severity_counts = df_filtered['severity_bin'].value_counts().reset_index()
            severity_counts.columns = ['severity_bin', 'count']
            
            # Define color map for severity levels
            color_map = {
                'critical': 'red',
                'high': 'orange',
                'medium': 'yellow',
                'low': 'green',
                'unknown': 'gray'
            }
            
            # Create pie chart
            fig = px.pie(
                severity_counts,
                values='count',
                names='severity_bin',
                color='severity_bin',
                color_discrete_map=color_map,
                title='Threats by Severity Level',
                hole=0.4  # Donut chart
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CVSS score distribution if available
            if 'cvssScore' in df_filtered.columns:
                # Filter out NaN values
                try:
                    cvss_df = df_filtered[pd.to_numeric(df_filtered['cvssScore'], errors='coerce').notna()]
                    cvss_df['cvssScore'] = pd.to_numeric(cvss_df['cvssScore'])
                    
                    if not cvss_df.empty:
                        fig = px.histogram(
                            cvss_df,
                            x='cvssScore',
                            nbins=20,
                            color_discrete_sequence=['#3366CC'],
                            title='Distribution of CVSS Scores',
                            labels={'cvssScore': 'CVSS Score', 'count': 'Number of Threats'}
                        )
                        
                        # Add threshold lines
                        max_height = cvss_df['cvssScore'].value_counts().max() * 1.1 if not cvss_df['cvssScore'].value_counts().empty else 10
                        
                        fig.add_shape(
                            type="line",
                            x0=9.0, y0=0, x1=9.0, y1=max_height,
                            line=dict(color="red", width=2, dash="dash")
                        )
                        
                        fig.add_shape(
                            type="line",
                            x0=7.0, y0=0, x1=7.0, y1=max_height,
                            line=dict(color="orange", width=2, dash="dash")
                        )
                        
                        fig.add_shape(
                            type="line",
                            x0=4.0, y0=0, x1=4.0, y1=max_height,
                            line=dict(color="yellow", width=2, dash="dash")
                        )
                        
                        # Add annotations
                        fig.add_annotation(
                            x=9.0, y=max_height * 0.9,
                            text="Critical Threshold", showarrow=False,
                            font=dict(color="red")
                        )
                        
                        fig.add_annotation(
                            x=7.0, y=max_height * 0.8,
                            text="High Threshold", showarrow=False,
                            font=dict(color="orange")
                        )
                        
                        fig.add_annotation(
                            x=4.0, y=max_height * 0.7,
                            text="Medium Threshold", showarrow=False,
                            font=dict(color="yellow")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid CVSS score data available")
                except Exception as e:
                    st.error(f"Error processing CVSS data: {e}")
                    st.info("CVSS score data could not be processed")
            else:
                st.info("CVSS score data not available")
        
        # Add severity trend over time
        if 'date' in df_filtered.columns and not df_filtered.empty:
            st.subheader("Severity Distribution Over Time")
            
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
                
                # Prepare data - filter out NaT values
                df_filtered_valid = df_filtered.dropna(subset=['date', 'severity_bin'])
                
                if not df_filtered_valid.empty:
                    # Group by date and severity
                    severity_categories = df_filtered_valid['severity_bin'].unique().tolist()
                    df_pivot = pd.crosstab(
                        index=df_filtered_valid['date'].dt.date,
                        columns=df_filtered_valid['severity_bin']
                    ).reset_index()
                    
                    # Ensure all severity categories exist in the DataFrame
                    for category in ['critical', 'high', 'medium', 'low', 'unknown']:
                        if category not in df_pivot.columns:
                            df_pivot[category] = 0
                    
                    # Convert to long format for plotting
                    df_pivot_long = pd.melt(
                        df_pivot,
                        id_vars=['date'],
                        value_vars=[c for c in ['critical', 'high', 'medium', 'low', 'unknown'] if c in df_pivot.columns],
                        var_name='severity',
                        value_name='count'
                    )
                    
                    # Create stacked area chart
                    fig = px.area(
                        df_pivot_long,
                        x='date',
                        y='count',
                        color='severity',
                        title='Severity Distribution Over Time',
                        labels={'date': 'Date', 'count': 'Number of Threats', 'severity': 'Severity Level'},
                        color_discrete_map={
                            'critical': 'red',
                            'high': 'orange',
                            'medium': 'yellow',
                            'low': 'green',
                            'unknown': 'gray'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough valid data for time series visualization")
            except Exception as e:
                st.error(f"Error creating severity trend visualization: {e}")
    else:
        st.info("Severity information not available")

def create_sentiment_analysis(df_filtered):
    """Create sentiment analysis visualizations."""
    st.subheader("Sentiment Analysis")
    
    if 'sentiment' in df_filtered.columns:
        try:
            # Ensure sentiment is numeric
            df_filtered['sentiment'] = pd.to_numeric(df_filtered['sentiment'], errors='coerce')
            df_valid = df_filtered.dropna(subset=['sentiment'])
            
            if df_valid.empty:
                st.info("No valid sentiment data available")
                return
                
            # Create columns for side-by-side visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution histogram
                fig = px.histogram(
                    df_valid,
                    x='sentiment',
                    nbins=20,
                    color_discrete_sequence=['#3366CC'],
                    title='Distribution of Sentiment Scores',
                    labels={'sentiment': 'Sentiment Score (-1 to 1)', 'count': 'Number of Threats'}
                )
                
                # Add zero line for reference
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=0, y1=df_valid['sentiment'].value_counts().max() * 1.1 if not df_valid['sentiment'].value_counts().empty else 10,
                    line=dict(color="black", width=2, dash="dash")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Categorize sentiment
                df_valid['sentiment_category'] = pd.cut(
                    df_valid['sentiment'],
                    bins=[-1.01, -0.5, -0.1, 0.1, 0.5, 1.01],
                    labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
                )
                
                # Sentiment category counts
                sentiment_counts = df_valid['sentiment_category'].value_counts().reset_index()
                sentiment_counts.columns = ['sentiment_category', 'count']
                
                # Define color map for sentiment categories
                color_map = {
                    'Very Negative': 'red',
                    'Negative': 'orange',
                    'Neutral': 'gray',
                    'Positive': 'lightgreen',
                    'Very Positive': 'green'
                }
                
                # Create pie chart
                fig = px.pie(
                    sentiment_counts,
                    values='count',
                    names='sentiment_category',
                    color='sentiment_category',
                    color_discrete_map=color_map,
                    title='Sentiment Distribution',
                    hole=0.4  # Donut chart
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add relation between sentiment and severity if severity data exists
            if 'severity_bin' in df_valid.columns:
                st.subheader("Sentiment vs. Severity")
                
                # Calculate average sentiment by severity
                sentiment_by_severity = df_valid.groupby('severity_bin')['sentiment'].mean().reset_index()
                
                # Color map
                color_map = {
                    'critical': 'red',
                    'high': 'orange',
                    'medium': 'yellow',
                    'low': 'green',
                    'unknown': 'gray'
                }
                
                # Create bar chart
                fig = px.bar(
                    sentiment_by_severity,
                    x='severity_bin',
                    y='sentiment',
                    color='severity_bin',
                    color_discrete_map=color_map,
                    title='Average Sentiment by Severity Level',
                    labels={'severity_bin': 'Severity Level', 'sentiment': 'Average Sentiment Score'}
                )
                
                # Add zero line for reference
                fig.add_shape(
                    type="line",
                    x0=-0.5, y0=0, x1=len(sentiment_by_severity)-0.5, y1=0,
                    line=dict(color="black", width=2, dash="dash")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add boxplot for sentiment distribution by severity
                fig = px.box(
                    df_valid,
                    x='severity_bin',
                    y='sentiment',
                    color='severity_bin',
                    color_discrete_map=color_map,
                    title='Sentiment Distribution by Severity Level',
                    labels={'severity_bin': 'Severity Level', 'sentiment': 'Sentiment Score'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing sentiment data: {e}")
    else:
        st.info("Sentiment information not available")

def create_model_performance_view(df_filtered):
    """Create visualization for the severity classification model performance."""
    st.header("Model Performance Analysis")
    
    # Check if we have prediction data
    has_predictions = all(col in df_filtered.columns for col in ['severity_pred', 'severity_prob_max'])
    
    if not has_predictions:
        st.info("No model predictions available. Run the severity classification model first.")
        return
    
    # Create tabs for different model comparisons
    model_tab1, model_tab2, model_tab3 = st.tabs([
        "Model Metrics", "Prediction Analysis", "Model Comparison"
    ])
    
    # Tab 1: Model Metrics from classification reports
    with model_tab1:
        st.subheader("Classification Report Metrics")
        
        # Find the latest metrics for both models
        try:
            lr_files = list(METRICS_DIR.glob("lr_report_*.json"))
            lgbm_files = list(METRICS_DIR.glob("lgbm_report_*.json"))
            
            col1, col2 = st.columns(2)
            
            # Display LogisticRegression metrics if available
            with col1:
                st.markdown("### TF-IDF + LogisticRegression")
                if lr_files:
                    latest_lr = max(lr_files)
                    with open(latest_lr, 'r') as f:
                        lr_metrics = json.load(f)
                    
                    # Format the metrics into a DataFrame
                    lr_df = pd.DataFrame({
                        'Precision': [lr_metrics[c]['precision'] for c in ['critical', 'high', 'medium', 'low']],
                        'Recall': [lr_metrics[c]['recall'] for c in ['critical', 'high', 'medium', 'low']],
                        'F1-Score': [lr_metrics[c]['f1-score'] for c in ['critical', 'high', 'medium', 'low']],
                        'Support': [lr_metrics[c]['support'] for c in ['critical', 'high', 'medium', 'low']]
                    }, index=['Critical', 'High', 'Medium', 'Low'])
                    
                    # Display metrics
                    st.dataframe(lr_df.style.format("{:.2f}", subset=['Precision', 'Recall', 'F1-Score']))
                    
                    # Display overall accuracy
                    if 'accuracy' in lr_metrics:
                        st.metric("Overall Accuracy", f"{lr_metrics['accuracy']:.2f}")
                    
                    # Display weighted averages
                    if 'weighted avg' in lr_metrics:
                        st.metric("Weighted F1-Score", f"{lr_metrics['weighted avg']['f1-score']:.2f}")
                else:
                    st.info("No LogisticRegression metrics available")
            
            # Display LightGBM metrics if available
            with col2:
                st.markdown("### SBERT + LightGBM")
                if lgbm_files:
                    latest_lgbm = max(lgbm_files)
                    with open(latest_lgbm, 'r') as f:
                        lgbm_metrics = json.load(f)
                    
                    # Format the metrics into a DataFrame
                    lgbm_df = pd.DataFrame({
                        'Precision': [lgbm_metrics[c]['precision'] for c in ['critical', 'high', 'medium', 'low']],
                        'Recall': [lgbm_metrics[c]['recall'] for c in ['critical', 'high', 'medium', 'low']],
                        'F1-Score': [lgbm_metrics[c]['f1-score'] for c in ['critical', 'high', 'medium', 'low']],
                        'Support': [lgbm_metrics[c]['support'] for c in ['critical', 'high', 'medium', 'low']]
                    }, index=['Critical', 'High', 'Medium', 'Low'])
                    
                    # Display metrics
                    st.dataframe(lgbm_df.style.format("{:.2f}", subset=['Precision', 'Recall', 'F1-Score']))
                    
                    # Display overall accuracy
                    if 'accuracy' in lgbm_metrics:
                        st.metric("Overall Accuracy", f"{lgbm_metrics['accuracy']:.2f}")
                    
                    # Display weighted averages
                    if 'weighted avg' in lgbm_metrics:
                        st.metric("Weighted F1-Score", f"{lgbm_metrics['weighted avg']['f1-score']:.2f}")
                else:
                    st.info("No LightGBM metrics available")
            
            # Compare metrics if both models are available
            if lr_files and lgbm_files:
                st.subheader("Model Comparison")
                
                # Create a bar chart to compare F1-scores
                comparison_data = []
                
                # Process LogisticRegression metrics
                for severity in ['critical', 'high', 'medium', 'low']:
                    comparison_data.append({
                        'model': 'LogisticRegression (TF-IDF)',
                        'severity': severity.capitalize(),
                        'metric': 'F1-Score',
                        'value': lr_metrics[severity]['f1-score']
                    })
                
                # Process LightGBM metrics
                for severity in ['critical', 'high', 'medium', 'low']:
                    comparison_data.append({
                        'model': 'LightGBM (SBERT)',
                        'severity': severity.capitalize(),
                        'metric': 'F1-Score',
                        'value': lgbm_metrics[severity]['f1-score']
                    })
                
                # Create comparison DataFrame
                comparison_df = pd.DataFrame(comparison_data)
                
                # Plot the comparison
                fig = px.bar(
                    comparison_df,
                    x='severity',
                    y='value',
                    color='model',
                    barmode='group',
                    title='F1-Score Comparison by Severity Level',
                    labels={'severity': 'Severity Level', 'value': 'F1-Score', 'model': 'Model'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Compare overall accuracy
                if 'accuracy' in lr_metrics and 'accuracy' in lgbm_metrics:
                    accuracy_data = [
                        {'model': 'LogisticRegression (TF-IDF)', 'accuracy': lr_metrics['accuracy']},
                        {'model': 'LightGBM (SBERT)', 'accuracy': lgbm_metrics['accuracy']}
                    ]
                    
                    accuracy_df = pd.DataFrame(accuracy_data)
                    
                    fig = px.bar(
                        accuracy_df,
                        x='model',
                        y='accuracy',
                        title='Overall Accuracy Comparison',
                        labels={'model': 'Model', 'accuracy': 'Accuracy'},
                        color='model'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error loading model metrics: {e}")
    
    # Tab 2: Prediction Analysis
    with model_tab2:
        st.subheader("Prediction Analysis")
        
        try:
            # Confusion Matrix
            st.markdown("### Actual vs. Predicted Severity")
            
            # Create a crosstab of actual vs. predicted
            confusion = pd.crosstab(
                df_filtered['severity_bin'], 
                df_filtered['severity_pred'],
                rownames=['Actual'],
                colnames=['Predicted']
            )
            
            # Visualize as heatmap
            fig = px.imshow(
                confusion,
                text_auto=True,
                labels=dict(x="Predicted Severity", y="Actual Severity", color="Count"),
                title="Confusion Matrix: Actual vs. Predicted Severity",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction Confidence
            st.markdown("### Prediction Confidence Analysis")
            
            # Confidence distribution
            fig = px.histogram(
                df_filtered,
                x='severity_prob_max',
                color='severity_pred',
                nbins=20,
                title='Distribution of Prediction Confidence Scores',
                labels={'severity_prob_max': 'Confidence Score', 'count': 'Number of Predictions'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence by Severity
            fig = px.box(
                df_filtered,
                x='severity_bin',
                y='severity_prob_max',
                color='severity_bin',
                title='Prediction Confidence by Severity Level',
                labels={'severity_bin': 'Severity Level', 'severity_prob_max': 'Confidence Score'},
                color_discrete_map={
                    'critical': 'red',
                    'high': 'orange',
                    'medium': 'yellow',
                    'low': 'green',
                    'unknown': 'gray'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Misclassification Analysis
            st.markdown("### Misclassification Analysis")
            
            # Create a dataframe with misclassified items
            df_misclassified = df_filtered[df_filtered['severity_bin'] != df_filtered['severity_pred']].copy()
            
            if not df_misclassified.empty:
                # Count misclassifications by severity
                misclass_counts = pd.crosstab(
                    df_misclassified['severity_bin'],
                    df_misclassified['severity_pred'],
                    rownames=['Actual'],
                    colnames=['Predicted']
                )
                
                # Visualize misclassifications
                fig = px.imshow(
                    misclass_counts,
                    text_auto=True,
                    labels=dict(x="Predicted as", y="Actually", color="Count"),
                    title="Misclassification Patterns",
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate error rate by severity
                error_rates = []
                for severity in df_filtered['severity_bin'].unique():
                    if severity == 'unknown':
                        continue
                    severity_count = len(df_filtered[df_filtered['severity_bin'] == severity])
                    misclass_count = len(df_misclassified[df_misclassified['severity_bin'] == severity])
                    if severity_count > 0:
                        error_rates.append({
                            'severity': severity,
                            'error_rate': misclass_count / severity_count * 100
                        })
                
                if error_rates:
                    error_df = pd.DataFrame(error_rates)
                    
                    fig = px.bar(
                        error_df,
                        x='severity',
                        y='error_rate',
                        color='severity',
                        title='Error Rate by Severity Level',
                        labels={'severity': 'Severity Level', 'error_rate': 'Error Rate (%)'},
                        color_discrete_map={
                            'critical': 'red',
                            'high': 'orange',
                            'medium': 'yellow',
                            'low': 'green'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No misclassifications found in the filtered data")
                
        except Exception as e:
            st.error(f"Error analyzing predictions: {e}")
    
    # Tab 3: Model Comparison
    with model_tab3:
        st.subheader("Model Architecture Comparison")
        
        # Create comparison table
        comparison_data = {
            'Feature': ['Feature Extraction', 'Feature Type', 'Model', 'Training Approach', 'Advantages', 'Disadvantages'],
            'TF-IDF + LogisticRegression': [
                'TF-IDF Vectorization',
                'Sparse bag-of-words representation',
                'Logistic Regression',
                'Standard training with class weights',
                'Fast training and inference, Interpretable model',
                'Ignores word order and context, Limited semantic understanding'
            ],
            'SBERT + LightGBM': [
                'SBERT Embeddings',
                'Dense contextual embeddings',
                'LightGBM Classifier',
                'Oversampling with RandomOverSampler',
                'Captures semantic meaning, Better for nuanced text',
                'More complex model, Requires pre-trained embeddings'
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Show model selection
        st.markdown("### Model Selection Criteria")
        st.markdown("""
        When choosing between these models, consider:
        
        1. **Performance**: Compare the F1-scores and accuracy metrics
        2. **Confidence**: Check the confidence score distributions
        3. **Error patterns**: Analyze which model makes fewer critical misclassifications
        4. **Resource requirements**: TF-IDF+LR is lighter weight, SBERT+LightGBM is more resource-intensive
        5. **Data characteristics**: SBERT may perform better with nuanced text and complex semantics
        """)

def create_word_cloud(df_filtered):
    """Create an enhanced word cloud visualization."""
    st.subheader("Threat Intelligence Word Cloud")
    
    text_column = None
    for col in ['clean_text', 'content']:
        if col in df_filtered.columns:
            text_column = col
            break
            
    if text_column is None:
        st.info("Text content not available for word cloud generation")
        return
    
    # Combine all text
    all_text = ' '.join(df_filtered[text_column].fillna('').astype(str))
    
    # Generate word cloud
    if all_text and len(all_text.strip()) > 10:  # Ensure there's some meaningful text
        try:
            # Set up columns for word cloud and settings
            col1, col2 = st.columns([3, 1])
            
            with col2:
                # Word cloud settings
                max_words = st.slider("Max Words", min_value=50, max_value=200, value=100, step=10)
                colormap = st.selectbox("Color Map", options=["viridis", "plasma", "inferno", "magma", "cividis"])
                
                # Severity-specific word cloud
                if 'severity_bin' in df_filtered.columns:
                    severity_options = ["All"] + sorted(df_filtered['severity_bin'].dropna().unique().tolist())
                    selected_severity = st.selectbox(
                        "Severity Filter", 
                        options=severity_options
                    )
            
            with col1:
                # Filter by severity if selected
                if 'severity_bin' in df_filtered.columns and selected_severity != "All":
                    severity_text = ' '.join(
                        df_filtered[df_filtered['severity_bin'] == selected_severity][text_column].fillna('').astype(str)
                    )
                    text_to_use = severity_text if severity_text and len(severity_text.strip()) > 10 else all_text
                    title = f"Word Cloud for {selected_severity.capitalize()} Severity Threats"
                else:
                    text_to_use = all_text
                    title = "Word Cloud for All Threats"
                
                # Generate and display word cloud
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    max_words=max_words,
                    colormap=colormap,
                    contour_width=1,
                    contour_color='steelblue'
                ).generate(text_to_use)
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(title)
                ax.axis('off')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating word cloud: {e}")
            st.info("Could not generate word cloud. Please check your data.")
    else:
        st.info("No meaningful text content available for word cloud generation")

def create_anomaly_analysis(df_filtered):
    """Create enhanced anomaly detection visualizations."""
    st.subheader("Anomaly Detection Analysis")
    
    if 'is_anomaly' in df_filtered.columns:
        # Basic anomaly stats
        anomaly_count = len(df_filtered[df_filtered['is_anomaly'] == True])
        anomaly_percentage = (anomaly_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Anomalies Detected",
                value=anomaly_count,
                delta=f"{anomaly_percentage:.1f}% of total"
            )
        
        with col2:
            if 'is_consensus_anomaly' in df_filtered.columns:
                consensus_count = len(df_filtered[df_filtered['is_consensus_anomaly'] == True])
                consensus_percentage = (consensus_count / anomaly_count * 100) if anomaly_count > 0 else 0
                
                st.metric(
                    label="Consensus Anomalies",
                    value=consensus_count,
                    delta=f"{consensus_percentage:.1f}% of anomalies"
                )
        
        # Anomaly detection methods
        if 'anomaly_detection' in df_filtered.columns and df_filtered['anomaly_detection'].notna().any():
            # Extract methods used
            methods = []
            for item in df_filtered[df_filtered['is_anomaly'] == True]['anomaly_detection']:
                if isinstance(item, dict) and 'methods' in item:
                    methods.extend(item['methods'])
            
            if methods:
                method_counts = pd.Series(methods).value_counts().reset_index()
                method_counts.columns = ['method', 'count']
                
                # Create bar chart
                fig = px.bar(
                    method_counts,
                    x='method',
                    y='count',
                    title='Anomalies by Detection Method',
                    labels={'method': 'Method', 'count': 'Number of Anomalies'},
                    color='count',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Anomalies over time
        if 'date' in df_filtered.columns and not df_filtered.empty:
            st.subheader("Anomalies Over Time")
            
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
                    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
                
                # Group by date for all threats
                df_total = df_filtered.groupby(df_filtered['date'].dt.date).size().reset_index(name='total')
                
                # Group by date for anomalies
                df_anomaly = df_filtered[df_filtered['is_anomaly'] == True].groupby(
                    df_filtered[df_filtered['is_anomaly'] == True]['date'].dt.date
                ).size().reset_index(name='anomalies')
                
                # Merge the two
                df_combined = pd.merge(df_total, df_anomaly, on='date', how='left').fillna(0)
                
                # Calculate anomaly percentage
                df_combined['anomaly_percentage'] = (df_combined['anomalies'] / df_combined['total']) * 100
                
                # Create dual-axis plot
                fig = go.Figure()
                
                # Add total threats
                fig.add_trace(
                    go.Scatter(
                        x=df_combined['date'],
                        y=df_combined['total'],
                        name='Total Threats',
                        mode='lines',
                        line=dict(color='blue', width=2)
                    )
                )
                
                # Add anomalies on secondary y-axis
                fig.add_trace(
                    go.Scatter(
                        x=df_combined['date'],
                        y=df_combined['anomalies'],
                        name='Anomalies',
                        mode='lines+markers',
                        line=dict(color='red', width=2),
                        marker=dict(size=8)
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title='Threats and Anomalies Over Time',
                    xaxis_title='Date',
                    yaxis_title='Count',
                    hovermode='x unified',
                    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add anomaly percentage chart
                fig = px.line(
                    df_combined,
                    x='date',
                    y='anomaly_percentage',
                    title='Anomaly Percentage Over Time',
                    labels={'date': 'Date', 'anomaly_percentage': 'Anomaly Percentage (%)'},
                    line_shape='linear'
                )
                
                # Add threshold line at 10%
                fig.add_shape(
                    type="line",
                    x0=df_combined['date'].min(), y0=10,
                    x1=df_combined['date'].max(), y1=10,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Add annotation
                if len(df_combined) > 1:
                    fig.add_annotation(
                        x=df_combined['date'].iloc[len(df_combined)//2], y=11,
                        text="Alert Threshold",
                        showarrow=False,
                        font=dict(color="red")
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating anomaly time series: {e}")
        
        # Anomalies by severity
        if 'severity_bin' in df_filtered.columns:
            st.subheader("Anomalies by Severity Level")
            
            try:
                # Create crosstab of anomalies by severity
                anomaly_severity = pd.crosstab(
                    df_filtered['severity_bin'],
                    df_filtered['is_anomaly'],
                    rownames=['Severity'],
                    colnames=['Is Anomaly'],
                    normalize='index'
                ) * 100  # Convert to percentage
                
                # Create bar chart
                fig = px.bar(
                    anomaly_severity.reset_index(),
                    x='Severity',
                    y=True,  # Is Anomaly = True
                    title='Percentage of Anomalies by Severity Level',
                    labels={'Severity': 'Severity Level', True: 'Percentage of Anomalies'},
                    color='Severity',
                    color_discrete_map={
                        'critical': 'red',
                        'high': 'orange',
                        'medium': 'yellow',
                        'low': 'green',
                        'unknown': 'gray'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating anomaly by severity chart: {e}")
    else:
        st.info("Anomaly detection information not available")

def create_enhanced_details_view(search_results):
    """Create an enhanced details view for threats."""
    st.subheader("Detailed Threat Analysis")
    
    if search_results.empty:
        st.info("No results match your search criteria.")
        return
    
    # Display results as cards
    for i, (_, item) in enumerate(search_results.iterrows()):
        # Create an expander for each item
        title = item.get('title', f'Item {i+1}')
        if pd.isna(title) or title == '':
            title = f'Item {i+1}'
            
        with st.expander(f"{i+1}. {title}", expanded=False):
            # Create columns for metadata and content
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Metadata")
                
                # Display basic info
                source = item.get('source', 'Unknown')
                if pd.notna(source):
                    st.markdown(f"**Source:** {source}")
                
                # Handle date with proper error handling
                if 'date' in item and pd.notna(item['date']):
                    try:
                        # Try to format as date if it's a datetime object
                        if isinstance(item['date'], pd.Timestamp):
                            date_str = item['date'].strftime("%Y-%m-%d")
                        else:
                            date_str = str(item['date'])
                        st.markdown(f"**Date:** {date_str}")
                    except:
                        # Fall back to string representation
                        st.markdown(f"**Date:** {str(item['date'])}")
                
                # Display severity information
                if 'severity_bin' in item and pd.notna(item['severity_bin']):
                    severity_color = {
                        'critical': 'red',
                        'high': 'orange',
                        'medium': 'yellow',
                        'low': 'green'
                    }.get(item['severity_bin'], 'gray')
                    
                    st.markdown(f"**Severity Level:** <span style='color:{severity_color};font-weight:bold'>{str(item['severity_bin']).capitalize()}</span>", unsafe_allow_html=True)
                    
                    if 'cvssScore' in item and pd.notna(item['cvssScore']):
                        try:
                            cvss_score = float(item['cvssScore'])
                            st.markdown(f"**CVSS Score:** {cvss_score:.1f}")
                        except:
                            st.markdown(f"**CVSS Score:** {item['cvssScore']}")
                
                # Display model predictions
                if 'severity_pred' in item and pd.notna(item['severity_pred']):
                    pred_color = {
                        'critical': 'red',
                        'high': 'orange',
                        'medium': 'yellow',
                        'low': 'green'
                    }.get(item['severity_pred'], 'gray')
                    
                    is_correct = item['severity_bin'] == item['severity_pred']
                    prediction_style = "" if is_correct else "text-decoration: line-through;"
                    
                    st.markdown(f"**Predicted Severity:** <span style='color:{pred_color};font-weight:bold;{prediction_style}'>{str(item['severity_pred']).capitalize()}</span>", unsafe_allow_html=True)
                    
                    if 'severity_prob_max' in item and pd.notna(item['severity_prob_max']):
                        try:
                            prob = float(item['severity_prob_max'])
                            st.markdown(f"**Prediction Confidence:** {prob:.2f}")
                        except:
                            st.markdown(f"**Prediction Confidence:** {item['severity_prob_max']}")
                
                # Display sentiment information
                if 'sentiment' in item and pd.notna(item['sentiment']):
                    try:
                        sentiment_value = float(item['sentiment'])
                        sentiment_color = 'red' if sentiment_value < -0.3 else 'orange' if sentiment_value < 0 else 'green' if sentiment_value > 0.3 else 'blue'
                        sentiment_label = 'Negative' if sentiment_value < -0.1 else 'Positive' if sentiment_value > 0.1 else 'Neutral'
                        
                        st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};font-weight:bold'>{sentiment_label} ({sentiment_value:.2f})</span>", unsafe_allow_html=True)
                    except:
                        st.markdown(f"**Sentiment:** {item['sentiment']}")
                
                # Display anomaly information
                if 'is_anomaly' in item and item['is_anomaly']:
                    st.markdown("**Anomaly Status:** <span style='color:purple;font-weight:bold'>Anomaly Detected</span>", unsafe_allow_html=True)
                    
                    if 'is_consensus_anomaly' in item and item['is_consensus_anomaly']:
                        st.markdown("**Consensus:** Multiple detection methods")
                
                # Display additional metadata
                meta_data = {k: v for k, v in item.items() if k.startswith('metadata_') and pd.notna(v)}
                if meta_data:
                    st.markdown("### Additional Information")
                    for key, value in meta_data.items():
                        clean_key = key.replace('metadata_', '').replace('_', ' ').title()
                        st.markdown(f"**{clean_key}:** {value}")
            
            with col2:
                st.markdown("### Content")
                
                # Display content with proper error handling
                content_field = None
                for field in ['clean_text', 'content']:
                    if field in item and pd.notna(item[field]):
                        content_field = field
                        break
                
                if content_field:
                    try:
                        content = str(item[content_field])
                        st.markdown(content, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error displaying content: {e}")
                        st.markdown("Content could not be displayed properly.")
                else:
                    st.markdown("No content available")
                
                # Display related threats if available
                if 'related_threats' in item and isinstance(item['related_threats'], list) and item['related_threats']:
                    st.markdown("### Related Threats")
                    for related in item['related_threats']:
                        st.markdown(f"- {related}")

# ===============================================================================
# Data Loading Functions
# ===============================================================================

def load_latest_parquet():
    """
    Load the latest master parquet file, preferring prediction-enhanced files.
    
    Returns:
        tuple: (dataframe, success_flag)
    """
    try:
        # Find prediction-enhanced parquet files first
        pred_parquet_files = list(PROC_DIR.glob("master_plus_pred_*.parquet"))
        
        if pred_parquet_files:
            # Use the prediction-enhanced file if available
            latest_file = max(pred_parquet_files)
            st.info(f"Loading model-enhanced data from: {latest_file.name}")
        else:
            # Fall back to regular master files
            master_files = list(PROC_DIR.glob("master_*.parquet"))
            if not master_files:
                return pd.DataFrame(), False
            latest_file = max(master_files)
            st.info(f"Loading data from: {latest_file.name} (no model predictions available)")
        
        # Load the parquet file
        df = pd.read_parquet(latest_file)
        
        # Basic data validation and cleaning
        if df.empty:
            st.warning("The parquet file is empty.")
            return df, False
            
        # Convert dates to datetime where possible
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Ensure severity_bin is a string
        if 'severity_bin' in df.columns:
            df['severity_bin'] = df['severity_bin'].astype(str)
        
        return df, True
        
    except Exception as e:
        st.error(f"Error loading parquet file: {str(e)}")
        return pd.DataFrame(), False

def load_anomaly_data():
    """
    Load anomaly detection data if available.
    
    Returns:
        dict: Anomaly data
    """
    try:
        anomaly_path = PROC_DIR / "anomaly_detected_dataset.json"
        if os.path.exists(anomaly_path):
            return load_from_json(anomaly_path)
        return {}
    except Exception as e:
        st.warning(f"Could not load anomaly data: {e}")
        return {}

def enrich_with_anomaly_data(df, anomaly_data):
    """
    Enrich the dataframe with anomaly detection data.
    
    Args:
        df (pandas.DataFrame): Main dataframe
        anomaly_data (dict): Anomaly detection data
        
    Returns:
        pandas.DataFrame: Enriched dataframe
    """
    if not anomaly_data:
        return df
    
    try:
        # Extract anomaly information for each item
        anomaly_records = []
        
        for item in anomaly_data:
            record = {
                'id': item.get('id', ''),
                'is_anomaly': False,
                'is_consensus_anomaly': False,
                'anomaly_methods': [],
                'anomaly_scores': {}
            }
            
            if 'anomaly_detection' in item:
                anomaly = item['anomaly_detection']
                record['is_anomaly'] = anomaly.get('is_anomaly', False)
                record['is_consensus_anomaly'] = anomaly.get('consensus', {}).get('is_consensus_anomaly', False)
                record['anomaly_methods'] = anomaly.get('methods', [])
                record['anomaly_scores'] = anomaly.get('scores', {})
                record['anomaly_detection'] = anomaly
            
            anomaly_records.append(record)
        
        # Convert to DataFrame
        anomaly_df = pd.DataFrame(anomaly_records)
        
        # Merge with main DataFrame if possible
        if 'id' in df.columns and not anomaly_df.empty:
            df = pd.merge(df, anomaly_df, on='id', how='left')
        
        return df
    except Exception as e:
        st.warning(f"Error enriching data with anomalies: {e}")
        return df

# ===============================================================================
# Main Dashboard Function
# ===============================================================================

def run_dashboard():
    """Main function to run the dashboard."""
    st.set_page_config(
        page_title="Cyber Threat Intelligence Dashboard",
        page_icon="🛡️",
        layout="wide"
    )
    
    # App title and description
    st.title("AI-Driven Cyber Threat Intelligence Dashboard")
    st.markdown("""
    This dashboard provides real-time insights into cyber threat intelligence collected from multiple sources.
    It combines TF-IDF+LogisticRegression and SBERT+LightGBM model predictions for severity classification.
    """)
    
    # Load data
    df, success = load_latest_parquet()
    
    if not success:
        st.error("Failed to load data. Please check the data pipeline.")
        st.stop()
    
    # Load anomaly data and enrich dataframe
    anomaly_data = load_anomaly_data()
    if anomaly_data:
        df = enrich_with_anomaly_data(df, anomaly_data)
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Date filter - with error handling
    if 'date' in df.columns:
        try:
            # Get valid dates only
            valid_dates = pd.to_datetime(df['date'], errors='coerce').dropna()
            
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                
                # Create date filter
                date_range = st.sidebar.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    # Filter the DataFrame, handling potential type issues
                    mask = (pd.to_datetime(df['date'], errors='coerce').dt.date >= start_date) & \
                           (pd.to_datetime(df['date'], errors='coerce').dt.date <= end_date)
                    df_filtered = df[mask]
                else:
                    df_filtered = df
            else:
                df_filtered = df
                st.sidebar.info("No valid dates available for filtering")
        except Exception as e:
            st.sidebar.warning(f"Error with date filtering: {e}")
            df_filtered = df
    else:
        df_filtered = df
        st.sidebar.info("Date filtering not available")
    
    # Source filter
    if 'source' in df_filtered.columns:
        # Get non-null sources
        sources = df_filtered['source'].dropna().unique().tolist()
        if sources:
            selected_sources = st.sidebar.multiselect(
                "Sources",
                options=sources,
                default=sources
            )
            if selected_sources:
                df_filtered = df_filtered[df_filtered['source'].isin(selected_sources)]
    
    # Severity filter
    if 'severity_bin' in df_filtered.columns:
        severity_levels = df_filtered['severity_bin'].dropna().unique().tolist()
        if severity_levels:
            selected_severity_levels = st.sidebar.multiselect(
                "Severity Levels",
                options=severity_levels,
                default=[]
            )
            if selected_severity_levels:
                df_filtered = df_filtered[df_filtered['severity_bin'].isin(selected_severity_levels)]
    
    # Model prediction filter
    if 'severity_pred' in df_filtered.columns:
        pred_levels = df_filtered['severity_pred'].dropna().unique().tolist()
        if pred_levels:
            selected_pred_levels = st.sidebar.multiselect(
                "Predicted Severity",
                options=pred_levels,
                default=[]
            )
            if selected_pred_levels:
                df_filtered = df_filtered[df_filtered['severity_pred'].isin(selected_pred_levels)]
        
        # Misclassification filter
        show_misclassified = st.sidebar.checkbox("Show Misclassified Only", value=False)
        if show_misclassified:
            df_filtered = df_filtered[df_filtered['severity_bin'] != df_filtered['severity_pred']]
    
    # Anomaly filter
    if 'is_anomaly' in df_filtered.columns:
        show_anomalies = st.sidebar.checkbox("Show Anomalies Only", value=False)
        if show_anomalies:
            df_filtered = df_filtered[df_filtered['is_anomaly'] == True]
    
    # Sentiment filter
    if 'sentiment' in df_filtered.columns:
        try:
            # Convert to numeric first
            df_filtered['sentiment'] = pd.to_numeric(df_filtered['sentiment'], errors='coerce')
            
            # Get min/max from valid values
            valid_sentiments = df_filtered['sentiment'].dropna()
            if not valid_sentiments.empty:
                min_sentiment = max(-1.0, valid_sentiments.min())
                max_sentiment = min(1.0, valid_sentiments.max())
                
                sentiment_range = st.sidebar.slider(
                    "Sentiment Score Range",
                    min_value=float(min_sentiment),
                    max_value=float(max_sentiment),
                    value=(float(min_sentiment), float(max_sentiment)),
                    step=0.1
                )
                
                df_filtered = df_filtered[
                    (df_filtered['sentiment'] >= sentiment_range[0]) & 
                    (df_filtered['sentiment'] <= sentiment_range[1])
                ]
        except Exception as e:
            st.sidebar.warning(f"Error with sentiment filtering: {e}")
    
    # Display data summary safely
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Summary")
    
    # Safe date range calculation
    date_range_text = "N/A"
    if 'date' in df.columns:
        try:
            valid_dates = pd.to_datetime(df['date'], errors='coerce').dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().strftime("%Y-%m-%d")
                max_date = valid_dates.max().strftime("%Y-%m-%d")
                date_range_text = f"{min_date} to {max_date}"
        except Exception as e:
            date_range_text = "Error processing dates"
    
    # Safe source count
    source_count = "N/A"
    if 'source' in df.columns:
        try:
            source_count = len(df['source'].dropna().unique())
        except:
            pass
    
    # Model information
    model_info = ""
    if 'severity_pred' in df.columns:
        model_info = "Model predictions available"
    
    st.sidebar.info(f"""
    Total Items: {len(df)}
    Filtered Items: {len(df_filtered)}
    Sources: {source_count}
    Date Range: {date_range_text}
    {model_info}
    """)
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Severity Analysis", "Model Performance", 
        "Sentiment Analysis", "Anomaly Detection", "Detailed View"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Threat Intelligence Overview")
        
        # Use enhanced KPI cards
        create_kpi_cards(df_filtered)
        
        # Use enhanced timeline visualization
        create_threat_timeline(df_filtered)
        
        # Source distribution
        if 'source' in df_filtered.columns:
            st.subheader("Threat Distribution by Source")
            # Get source counts
            source_counts = df_filtered['source'].value_counts().reset_index()
            source_counts.columns = ['source', 'count']
            
            if not source_counts.empty:
                fig = px.pie(
                    source_counts, 
                    names='source',
                    values='count',
                    title='Threats by Source',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No source information available")
        
        # Use enhanced word cloud
        create_word_cloud(df_filtered)
    
    # Tab 2: Severity Analysis
    with tab2:
        st.header("Severity Analysis")
        
        # Use enhanced severity distribution analysis
        create_severity_distribution(df_filtered)
    
    # Tab 3: Model Performance
    with tab3:
        # Use enhanced model performance view
        create_model_performance_view(df_filtered)
    
    # Tab 4: Sentiment Analysis
    with tab4:
        st.header("Sentiment Analysis")
        
        # Use sentiment analysis visualizations
        create_sentiment_analysis(df_filtered)
    
    # Tab 5: Anomaly Detection
    with tab5:
        st.header("Anomaly Detection")
        
        if 'is_anomaly' in df_filtered.columns:
            # Use enhanced anomaly analysis
            create_anomaly_analysis(df_filtered)
        else:
            st.info("Anomaly detection data not available. Run the anomaly detection component of your pipeline first.")
    
    # Tab 6: Detailed View
    with tab6:
        st.header("Detailed Threat View")
        
        # Search functionality
        st.subheader("Search Threats")
        search_term = st.text_input("Search in title or content")
        
        search_results = df_filtered.copy()
        
        if search_term:
            try:
                # Search in title and content
                search_fields = []
                for field in ['title', 'content', 'clean_text']:
                    if field in df_filtered.columns:
                        search_fields.append(field)
                
                if search_fields:
                    search_results = pd.DataFrame()
                    for field in search_fields:
                        # Handle potential None values safely
                        field_results = df_filtered[
                            df_filtered[field].astype(str).str.contains(search_term, case=False, na=False)
                        ]
                        search_results = pd.concat([search_results, field_results])
                    
                    # Remove duplicates
                    if not search_results.empty and 'id' in search_results.columns:
                        search_results = search_results.drop_duplicates(subset=['id'])
            except Exception as e:
                st.warning(f"Error with search: {e}")
                search_results = df_filtered
        
        # Sorting options
        sort_options = ['None']
        if 'date' in search_results.columns:
            sort_options.extend(['Date (Newest First)', 'Date (Oldest First)'])
        if 'cvssScore' in search_results.columns:
            sort_options.append('CVSS Score (Highest First)')
        if 'sentiment' in search_results.columns:
            sort_options.extend(['Sentiment (Most Negative First)', 'Sentiment (Most Positive First)'])
        if 'severity_bin' in search_results.columns:
            sort_options.append('Severity')
        if 'severity_prob_max' in search_results.columns:
            sort_options.append('Prediction Confidence')
        if 'source' in search_results.columns:
            sort_options.append('Source')
        
        sort_by = st.selectbox("Sort by", options=sort_options)
        
        # Apply sorting
        try:
            if sort_by == 'Date (Newest First)':
                search_results['date'] = pd.to_datetime(search_results['date'], errors='coerce')
                search_results = search_results.sort_values('date', ascending=False, na_position='last')
            elif sort_by == 'Date (Oldest First)':
                search_results['date'] = pd.to_datetime(search_results['date'], errors='coerce')
                search_results = search_results.sort_values('date', ascending=True, na_position='last')
            elif sort_by == 'CVSS Score (Highest First)':
                search_results['cvssScore'] = pd.to_numeric(search_results['cvssScore'], errors='coerce')
                search_results = search_results.sort_values('cvssScore', ascending=False, na_position='last')
            elif sort_by == 'Sentiment (Most Negative First)':
                search_results['sentiment'] = pd.to_numeric(search_results['sentiment'], errors='coerce')
                search_results = search_results.sort_values('sentiment', ascending=True, na_position='last')
            elif sort_by == 'Sentiment (Most Positive First)':
                search_results['sentiment'] = pd.to_numeric(search_results['sentiment'], errors='coerce')
                search_results = search_results.sort_values('sentiment', ascending=False, na_position='last')
            elif sort_by == 'Prediction Confidence':
                search_results['severity_prob_max'] = pd.to_numeric(search_results['severity_prob_max'], errors='coerce')
                search_results = search_results.sort_values('severity_prob_max', ascending=False, na_position='last')
            elif sort_by == 'Severity':
                # Create a severity order for sorting
                severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'unknown': 4}
                search_results['severity_order'] = search_results['severity_bin'].map(severity_order)
                search_results = search_results.sort_values('severity_order', na_position='last')
            elif sort_by == 'Source':
                search_results = search_results.sort_values('source', na_position='last')
        except Exception as e:
            st.warning(f"Error sorting results: {e}")
        
        # Use enhanced details view
        create_enhanced_details_view(search_results)

if __name__ == "__main__":
    run_dashboard()