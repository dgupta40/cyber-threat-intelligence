"""
Interactive dashboard for cyber threat intelligence visualization.
Built with Streamlit.
"""

import os
import sys
import logging
import json
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
from sklearn.decomposition import PCA

# Import enhanced dashboard components
from dashboard.enhanced_streamlit_dashboard import (
    create_kpi_cards,
    create_threat_timeline,
    create_risk_distribution,
    create_category_analysis,
    create_anomaly_analysis,
    create_word_cloud,
    create_enhanced_details_view
)

# Ensure parent directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_from_json
from classification.train_model import ThreatClassifier
from sentiment_analysis.assess_risk import RiskAssessor


def load_data():
    """
    Load the processed data for the dashboard.
    
    Returns:
        tuple: (data, success_flag)
    """
    # Try to load most processed data first, then fall back to less processed versions
    data_paths = [
        'data/processed/anomaly_detected_dataset.json',
        'data/processed/risk_assessed_dataset.json',
        'data/processed/combined_dataset.json'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            data = load_from_json(path)
            if data:
                return data, True
    
    return [], False


def load_summaries():
    """
    Load summary data.
    
    Returns:
        dict: Summary data
    """
    summaries = {}
    
    # Risk summary
    risk_path = 'data/processed/risk_summary.json'
    if os.path.exists(risk_path):
        summaries['risk'] = load_from_json(risk_path)
    
    # Anomaly summary
    anomaly_path = 'data/processed/anomaly_summary.json'
    if os.path.exists(anomaly_path):
        summaries['anomaly'] = load_from_json(anomaly_path)
    
    return summaries


def create_dataframe(data):
    """
    Convert JSON data to pandas DataFrame.
    
    Args:
        data (list): JSON data
        
    Returns:
        pandas.DataFrame: DataFrame
    """
    # Extract relevant fields and flatten structure
    records = []
    for item in data:
        record = {
            'id': item.get('id', ''),
            'source': item.get('source', ''),
            'title': item.get('title', ''),
            'content': item.get('content', ''),
            'date': item.get('date', ''),
        }
        
        # Add category if available
        if 'category' in item:
            record['category'] = item['category']
        
        # Add risk assessment if available
        if 'risk_assessment' in item:
            risk = item['risk_assessment'].get('combined', {})
            record['risk_level'] = risk.get('risk_level', 'Unknown')
            record['risk_score'] = risk.get('risk_score', 0.0)
        
        # Add anomaly detection if available
        if 'anomaly_detection' in item:
            anomaly = item['anomaly_detection']
            record['is_anomaly'] = anomaly.get('is_anomaly', False)
            record['is_consensus_anomaly'] = anomaly.get('consensus', {}).get('is_consensus_anomaly', False)
            # Store the complete anomaly_detection object for detailed analysis
            record['anomaly_detection'] = anomaly
        
        # Add metadata
        if 'metadata' in item:
            for key, value in item['metadata'].items():
                record[f'metadata_{key}'] = value
        
        records.append(record)
    
    return pd.DataFrame(records)


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
    """)
    
    # Load data
    data, success = load_data()
    summaries = load_summaries()
    
    if not success:
        st.error("Failed to load data. Please run the data pipeline first.")
        return
    
    # Convert to DataFrame
    df = create_dataframe(data)
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Date filter
    if 'date' in df.columns and not df['date'].empty:
        try:
            # Convert dates
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min()
            max_date = df['date'].max()
            
            # Create date filter
            date_range = st.sidebar.date_input(
                "Date range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
            else:
                df_filtered = df
        except Exception as e:
            st.sidebar.warning(f"Error processing dates: {str(e)}")
            df_filtered = df
    else:
        df_filtered = df
    
    # Source filter
    if 'source' in df_filtered.columns and not df_filtered['source'].empty:
        sources = df_filtered['source'].unique().tolist()
        selected_sources = st.sidebar.multiselect(
            "Sources",
            options=sources,
            default=sources
        )
        if selected_sources:
            df_filtered = df_filtered[df_filtered['source'].isin(selected_sources)]
    
    # Category filter
    if 'category' in df_filtered.columns and not df_filtered['category'].empty:
        categories = df_filtered['category'].unique().tolist()
        selected_categories = st.sidebar.multiselect(
            "Categories",
            options=categories,
            default=[]
        )
        if selected_categories:
            df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]
    
    # Risk level filter
    if 'risk_level' in df_filtered.columns and not df_filtered['risk_level'].empty:
        risk_levels = df_filtered['risk_level'].unique().tolist()
        selected_risk_levels = st.sidebar.multiselect(
            "Risk Levels",
            options=risk_levels,
            default=[]
        )
        if selected_risk_levels:
            df_filtered = df_filtered[df_filtered['risk_level'].isin(selected_risk_levels)]
    
    # Anomaly filter
    if 'is_anomaly' in df_filtered.columns:
        show_anomalies = st.sidebar.checkbox("Show Anomalies Only", value=False)
        if show_anomalies:
            df_filtered = df_filtered[df_filtered['is_anomaly'] == True]
    
    # Display data summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Summary")
    st.sidebar.info(f"""
    Total Items: {len(df)}
    Filtered Items: {len(df_filtered)}
    Sources: {len(df['source'].unique())}
    Date Range: {df['date'].min().date() if 'date' in df.columns and not df['date'].empty else 'N/A'} to {df['date'].max().date() if 'date' in df.columns and not df['date'].empty else 'N/A'}
    """)
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Threat Categories", "Risk Assessment", "Anomaly Detection", "Detailed View"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Threat Intelligence Overview")
        
        # Use enhanced KPI cards
        create_kpi_cards(df_filtered)
        
        # Use enhanced timeline visualization
        create_threat_timeline(df_filtered)
        
        # Source distribution (keep existing pie chart)
        st.subheader("Threat Distribution by Source")
        fig = px.pie(
            df_filtered, 
            names='source',
            title='Threats by Source',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Use enhanced word cloud
        create_word_cloud(df_filtered)
    
    # Tab 2: Threat Categories
    with tab2:
        st.header("Threat Categories Analysis")
        
        # Use enhanced category analysis
        create_category_analysis(df_filtered)
    
    # Tab 3: Risk Assessment
    with tab3:
        st.header("Risk Assessment")
        
        # Use enhanced risk distribution analysis
        create_risk_distribution(df_filtered)
        
        # Add risk summary metrics if available
        if 'risk' in summaries:
            st.subheader("Risk Summary Statistics")
            
            risk_summary = summaries['risk']
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Average Risk Score",
                    value=f"{risk_summary.get('average_risk_score', 0):.2f}"
                )
            
            with col2:
                high_risk_percent = risk_summary.get('risk_percentages', {}).get('High', 0)
                st.metric(
                    label="High Risk Percentage",
                    value=f"{high_risk_percent:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="Total Assessed Items",
                    value=risk_summary.get('total_items', 0)
                )
            
            # Show risk by source
            if 'by_source' in risk_summary:
                st.subheader("Risk Distribution by Source")
                
                # Extract source risks
                source_risks = risk_summary.get('by_source', {})
                
                # Prepare data
                source_risk_data = []
                for source, risks in source_risks.items():
                    for risk_level, count in risks.items():
                        source_risk_data.append({
                            'source': source,
                            'risk_level': risk_level,
                            'count': count
                        })
                
                if source_risk_data:
                    source_risk_df = pd.DataFrame(source_risk_data)
                    
                    # Define color map for risk levels
                    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green', 'Unknown': 'gray'}
                    
                    # Create plot
                    fig = px.bar(
                        source_risk_df,
                        x='source',
                        y='count',
                        color='risk_level',
                        color_discrete_map=color_map,
                        title='Risk Distribution by Source',
                        labels={'source': 'Source', 'count': 'Number of Threats', 'risk_level': 'Risk Level'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Anomaly Detection
    with tab4:
        st.header("Anomaly Detection")
        
        # Use enhanced anomaly analysis
        create_anomaly_analysis(df_filtered)
        
        # Add additional temporal anomaly information if available
        if 'anomaly' in summaries and 'temporal_anomalies' in summaries['anomaly']:
            st.subheader("Temporal Anomalies")
            
            temporal = summaries['anomaly']['temporal_anomalies']
            spikes = temporal.get('spikes', [])
            clusters = temporal.get('clusters', [])
            
            if spikes:
                st.markdown("#### Activity Spikes")
                
                # Create dataframe
                spike_df = pd.DataFrame(spikes)
                spike_df['date'] = pd.to_datetime(spike_df['date'])
                
                # Create plot
                fig = px.bar(
                    spike_df,
                    x='date',
                    y='count',
                    hover_data=['z_score'],
                    title='Detected Activity Spikes',
                    labels={'date': 'Date', 'count': 'Threat Count', 'z_score': 'Z-Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if clusters:
                st.markdown("#### Activity Clusters")
                
                # Create dataframe
                cluster_data = []
                for cluster in clusters:
                    cluster_data.append({
                        'start_date': pd.to_datetime(cluster['start_date']),
                        'end_date': pd.to_datetime(cluster['end_date']),
                        'days': cluster['days'],
                        'total_count': cluster['total_count'],
                        'average_count': cluster['average_count']
                    })
                
                cluster_df = pd.DataFrame(cluster_data)
                
                # Display as table
                st.dataframe(cluster_df, use_container_width=True)
    
    # Tab 5: Detailed View
    with tab5:
        st.header("Detailed Threat View")
        
        # Search functionality
        st.subheader("Search Threats")
        search_term = st.text_input("Search in title or content")
        
        if search_term:
            # Search in title and content
            if 'title' in df_filtered.columns and 'content' in df_filtered.columns:
                search_results = df_filtered[
                    df_filtered['title'].str.contains(search_term, case=False, na=False) | 
                    df_filtered['content'].str.contains(search_term, case=False, na=False)
                ]
            else:
                search_results = df_filtered
        else:
            search_results = df_filtered
        
        # Sorting options
        sort_options = []
        if 'date' in search_results.columns:
            sort_options.append('Date (Newest First)')
            sort_options.append('Date (Oldest First)')
        if 'risk_score' in search_results.columns:
            sort_options.append('Risk Score (Highest First)')
        if 'category' in search_results.columns:
            sort_options.append('Category')
        if 'source' in search_results.columns:
            sort_options.append('Source')
        
        sort_by = st.selectbox("Sort by", options=sort_options) if sort_options else None
        
        # Apply sorting
        if sort_by:
            if sort_by == 'Date (Newest First)':
                search_results = search_results.sort_values('date', ascending=False)
            elif sort_by == 'Date (Oldest First)':
                search_results = search_results.sort_values('date', ascending=True)
            elif sort_by == 'Risk Score (Highest First)':
                search_results = search_results.sort_values('risk_score', ascending=False)
            elif sort_by == 'Category':
                search_results = search_results.sort_values('category')
            elif sort_by == 'Source':
                search_results = search_results.sort_values('source')
        
        # Use enhanced details view
        create_enhanced_details_view(search_results)


if __name__ == "__main__":
    # Run the dashboard
    run_dashboard()