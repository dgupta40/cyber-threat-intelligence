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
        
        # Create columns for KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Threats",
                value=len(df_filtered)
            )
        
        with col2:
            if 'risk_level' in df_filtered.columns:
                high_risk_count = len(df_filtered[df_filtered['risk_level'] == 'High'])
                st.metric(
                    label="High Risk Threats",
                    value=high_risk_count,
                    delta=f"{high_risk_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
                )
            else:
                st.metric(label="High Risk Threats", value="N/A")
        
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
            if 'date' in df_filtered.columns and not df_filtered['date'].empty:
                last_updated = df_filtered['date'].max()
                st.metric(
                    label="Last Updated",
                    value=last_updated.strftime("%Y-%m-%d")
                )
            else:
                st.metric(label="Last Updated", value="N/A")
        
        # Threats over time
        st.subheader("Threats Over Time")
        if 'date' in df_filtered.columns and not df_filtered['date'].empty:
            # Group by date
            df_time = df_filtered.groupby(df_filtered['date'].dt.date).size().reset_index(name='count')
            
            # Create plot
            fig = px.line(
                df_time, 
                x='date', 
                y='count',
                title='Threat Count by Date',
                labels={'date': 'Date', 'count': 'Number of Threats'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add source breakdown if multiple sources
            if len(df_filtered['source'].unique()) > 1:
                st.subheader("Threats by Source Over Time")
                # Group by date and source
                df_source_time = df_filtered.groupby([df_filtered['date'].dt.date, 'source']).size().reset_index(name='count')
                
                # Create plot
                fig = px.line(
                    df_source_time, 
                    x='date', 
                    y='count',
                    color='source',
                    title='Threat Count by Source and Date',
                    labels={'date': 'Date', 'count': 'Number of Threats', 'source': 'Source'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date information not available for time series visualization")
        
        # Source distribution
        st.subheader("Threat Distribution by Source")
        fig = px.pie(
            df_filtered, 
            names='source',
            title='Threats by Source',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate word cloud
        st.subheader("Common Terms in Threat Intelligence")
        if 'content' in df_filtered.columns:
            # Combine all text
            all_text = ' '.join(df_filtered['content'].fillna(''))
            
            # Generate word cloud
            if all_text:
                try:
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=100,
                        contour_width=3
                    ).generate(all_text)
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
            else:
                st.info("No text content available for word cloud")
        else:
            st.info("Text content not available for word cloud generation")
        
    # Tab 2: Threat Categories
    with tab2:
        st.header("Threat Categories Analysis")
        
        if 'category' in df_filtered.columns and not df_filtered['category'].empty:
            # Category distribution
            st.subheader("Threat Category Distribution")
            
            # Create category counts
            category_counts = df_filtered['category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            
            # Bar chart
            fig = px.bar(
                category_counts,
                x='category',
                y='count',
                title='Threats by Category',
                labels={'category': 'Category', 'count': 'Number of Threats'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Category by source
            if len(df_filtered['source'].unique()) > 1:
                st.subheader("Categories by Source")
                
                # Create cross-tabulation
                category_source = pd.crosstab(df_filtered['category'], df_filtered['source'])
                
                # Create stacked bar chart
                fig = px.bar(
                    category_source,
                    title='Categories by Source',
                    labels={'value': 'Number of Threats', 'index': 'Category'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Category over time
            if 'date' in df_filtered.columns and not df_filtered['date'].empty:
                st.subheader("Categories Over Time")
                
                # Group by date and category
                df_cat_time = df_filtered.groupby([df_filtered['date'].dt.date, 'category']).size().reset_index(name='count')
                
                # Create plot
                fig = px.line(
                    df_cat_time, 
                    x='date', 
                    y='count',
                    color='category',
                    title='Threat Categories Over Time',
                    labels={'date': 'Date', 'count': 'Number of Threats', 'category': 'Category'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Category and risk level
            if 'risk_level' in df_filtered.columns:
                st.subheader("Risk Level by Category")
                
                # Create cross-tabulation
                category_risk = pd.crosstab(df_filtered['category'], df_filtered['risk_level'])
                
                # Create stacked bar chart
                fig = px.bar(
                    category_risk,
                    title='Risk Level by Category',
                    labels={'value': 'Number of Threats', 'index': 'Category'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Extract top terms for each category
            st.subheader("Top Terms by Category")
            
            # Get categories
            categories = df_filtered['category'].unique()
            
            # Create columns for multiple word clouds
            cols = st.columns(min(3, len(categories)))
            
            # Generate word cloud for each category
            for i, category in enumerate(categories):
                col_idx = i % len(cols)
                
                with cols[col_idx]:
                    st.markdown(f"#### {category}")
                    
                    # Get text for this category
                    category_text = ' '.join(df_filtered[df_filtered['category'] == category]['content'].fillna(''))
                    
                    if category_text:
                        try:
                            wordcloud = WordCloud(
                                width=400, 
                                height=200, 
                                background_color='white',
                                max_words=50,
                                contour_width=3
                            ).generate(category_text)
                            
                            # Display word cloud
                            fig, ax = plt.subplots(figsize=(5, 3))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error generating word cloud for {category}: {str(e)}")
                    else:
                        st.info(f"No text content available for {category}")
        else:
            st.info("Category information not available")
    
    # Tab 3: Risk Assessment
    with tab3:
        st.header("Risk Assessment")
        
        if 'risk_level' in df_filtered.columns and 'risk_score' in df_filtered.columns:
            # Risk level distribution
            st.subheader("Risk Level Distribution")
            
            # Create risk level counts
            risk_counts = df_filtered['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['risk_level', 'count']
            
            # Define color map for risk levels
            color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green', 'Unknown': 'gray'}
            
            # Bar chart
            fig = px.bar(
                risk_counts,
                x='risk_level',
                y='count',
                color='risk_level',
                color_discrete_map=color_map,
                title='Threats by Risk Level',
                labels={'risk_level': 'Risk Level', 'count': 'Number of Threats'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk score distribution
            st.subheader("Risk Score Distribution")
            
            # Histogram
            fig = px.histogram(
                df_filtered,
                x='risk_score',
                nbins=20,
                title='Distribution of Risk Scores',
                labels={'risk_score': 'Risk Score', 'count': 'Number of Threats'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk by source
            if len(df_filtered['source'].unique()) > 1:
                st.subheader("Risk Level by Source")
                
                # Create cross-tabulation
                risk_source = pd.crosstab(df_filtered['risk_level'], df_filtered['source'])
                
                # Create stacked bar chart
                fig = px.bar(
                    risk_source,
                    title='Risk Level by Source',
                    labels={'value': 'Number of Threats', 'index': 'Risk Level'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk over time
            if 'date' in df_filtered.columns and not df_filtered['date'].empty:
                st.subheader("Risk Levels Over Time")
                
                # Group by date and risk level
                df_risk_time = df_filtered.groupby([df_filtered['date'].dt.date, 'risk_level']).size().reset_index(name='count')
                
                # Create plot
                fig = px.line(
                    df_risk_time, 
                    x='date', 
                    y='count',
                    color='risk_level',
                    color_discrete_map=color_map,
                    title='Risk Levels Over Time',
                    labels={'date': 'Date', 'count': 'Number of Threats', 'risk_level': 'Risk Level'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Average risk score over time
                st.subheader("Average Risk Score Over Time")
                
                # Group by date
                df_risk_avg = df_filtered.groupby(df_filtered['date'].dt.date)['risk_score'].mean().reset_index()
                
                # Create plot
                fig = px.line(
                    df_risk_avg, 
                    x='date', 
                    y='risk_score',
                    title='Average Risk Score Over Time',
                    labels={'date': 'Date', 'risk_score': 'Average Risk Score'}
                )
                
                # Add threshold lines
                fig.add_shape(
                    type="line",
                    x0=df_risk_avg['date'].min(),
                    y0=0.7,
                    x1=df_risk_avg['date'].max(),
                    y1=0.7,
                    line=dict(color="red", width=2, dash="dash"),
                    name="High Risk Threshold"
                )
                
                fig.add_shape(
                    type="line",
                    x0=df_risk_avg['date'].min(),
                    y0=0.3,
                    x1=df_risk_avg['date'].max(),
                    y1=0.3,
                    line=dict(color="orange", width=2, dash="dash"),
                    name="Medium Risk Threshold"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk summary
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
                st.subheader("Risk Distribution by Source")
                
                # Extract source risks
                source_risks = risk_summary.get('by_source', {})
                if source_risks:
                    # Prepare data
                    source_risk_data = []
                    for source, risks in source_risks.items():
                        for risk_level, count in risks.items():
                            source_risk_data.append({
                                'source': source,
                                'risk_level': risk_level,
                                'count': count
                            })
                    
                    source_risk_df = pd.DataFrame(source_risk_data)
                    
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
                else:
                    st.info("Source risk distribution not available")
        else:
            st.info("Risk assessment information not available")
    
    # Tab 4: Anomaly Detection
    with tab4:
        st.header("Anomaly Detection")
        
        if 'is_anomaly' in df_filtered.columns:
            # Basic anomaly count
            anomaly_count = len(df_filtered[df_filtered['is_anomaly'] == True])
            consensus_count = len(df_filtered[df_filtered['is_consensus_anomaly'] == True]) if 'is_consensus_anomaly' in df_filtered.columns else 0
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Anomalies",
                    value=anomaly_count,
                    delta=f"{anomaly_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
                )
            
            with col2:
                st.metric(
                    label="Consensus Anomalies",
                    value=consensus_count,
                    delta=f"{consensus_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
                )
            
            with col3:
                if 'anomaly' in summaries:
                    method_counts = summaries['anomaly'].get('method_counts', {})
                    st.metric(
                        label="Detection Methods",
                        value=len(method_counts)
                    )
                else:
                    st.metric(label="Detection Methods", value="N/A")
            
            # Anomaly detection methods
            if 'anomaly' in summaries and 'method_counts' in summaries['anomaly']:
                st.subheader("Anomaly Detection Methods")
                
                # Extract method counts
                method_counts = summaries['anomaly']['method_counts']
                
                # Prepare data
                method_data = [{'method': method, 'count': count} for method, count in method_counts.items()]
                method_df = pd.DataFrame(method_data)
                
                # Create plot
                fig = px.bar(
                    method_df,
                    x='method',
                    y='count',
                    title='Anomalies by Detection Method',
                    labels={'method': 'Method', 'count': 'Number of Anomalies'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Temporal anomalies
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
            
            # Anomalies by category
            if 'category' in df_filtered.columns:
                st.subheader("Anomalies by Category")
                
                # Group by category and anomaly status
                anomaly_by_category = df_filtered.groupby(['category', 'is_anomaly']).size().reset_index(name='count')
                
                # Get only anomalies
                anomaly_by_category = anomaly_by_category[anomaly_by_category['is_anomaly'] == True]
                
                if not anomaly_by_category.empty:
                    # Create plot
                    fig = px.bar(
                        anomaly_by_category,
                        x='category',
                        y='count',
                        title='Anomalies by Category',
                        labels={'category': 'Category', 'count': 'Number of Anomalies'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No anomalies found in the current filter")
            
            # Anomalies by risk level
            if 'risk_level' in df_filtered.columns:
                st.subheader("Anomalies by Risk Level")
                
                # Group by risk level and anomaly status
                anomaly_by_risk = df_filtered.groupby(['risk_level', 'is_anomaly']).size().reset_index(name='count')
                
                # Get only anomalies
                anomaly_by_risk = anomaly_by_risk[anomaly_by_risk['is_anomaly'] == True]
                
                if not anomaly_by_risk.empty:
                    # Create plot
                    fig = px.bar(
                        anomaly_by_risk,
                        x='risk_level',
                        y='count',
                        color='risk_level',
                        color_discrete_map=color_map,
                        title='Anomalies by Risk Level',
                        labels={'risk_level': 'Risk Level', 'count': 'Number of Anomalies'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No anomalies found in the current filter")
            
            # Anomalies over time
            if 'date' in df_filtered.columns and not df_filtered['date'].empty:
                st.subheader("Anomalies Over Time")
                
                # Group by date and anomaly status
                df_anomaly_time = df_filtered.groupby([df_filtered['date'].dt.date, 'is_anomaly']).size().reset_index(name='count')
                
                # Get only anomalies
                df_anomaly_time = df_anomaly_time[df_anomaly_time['is_anomaly'] == True]
                
                if not df_anomaly_time.empty:
                    # Create plot
                    fig = px.line(
                        df_anomaly_time, 
                        x='date', 
                        y='count',
                        title='Anomalies Over Time',
                        labels={'date': 'Date', 'count': 'Number of Anomalies'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No anomalies found in the selected time range")
        else:
            st.info("Anomaly detection information not available")
    
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
        
        # Display results
        st.subheader(f"Results ({len(search_results)} threats)")
        
        # Pagination
        items_per_page = 10
        num_pages = (len(search_results) - 1) // items_per_page + 1
        
        page = st.number_input(
            f"Page (1-{num_pages})",
            min_value=1,
            max_value=max(1, num_pages),
            value=1
        )
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(search_results))
        
        page_results = search_results.iloc[start_idx:end_idx]
        
        # Show results as expandable items
        for i, (_, item) in enumerate(page_results.iterrows()):
            # Create title with metadata
            title = item.get('title', 'No Title')
            source = item.get('source', 'Unknown Source')
            date = item.get('date', '')
            date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
            
            # Add risk and anomaly indicators
            risk_level = item.get('risk_level', '')
            is_anomaly = item.get('is_anomaly', False)
            
            # Set color based on risk level
            title_color = 'red' if risk_level == 'High' else 'orange' if risk_level == 'Medium' else 'green' if risk_level == 'Low' else 'black'
            
            # Create title
            title_html = f"<span style='color:{title_color}'>{title}</span>"
            if is_anomaly:
                title_html += " <span style='color:purple'>[ANOMALY]</span>"
            
            title_html += f" <small>({source}, {date_str})</small>"
            
            # Create expandable section
            with st.expander(f"{i+1+start_idx}. {title_html}", expanded=False):
                if 'category' in item:
                    st.markdown(f"**Category:** {item['category']}")
                
                if 'risk_level' in item:
                    st.markdown(f"**Risk Level:** {item['risk_level']} (Score: {item.get('risk_score', 0):.2f})")
                
                if 'is_anomaly' in item and item['is_anomaly']:
                    st.markdown("**Anomaly Detection:** This threat was identified as an anomaly")
                    if 'is_consensus_anomaly' in item and item['is_consensus_anomaly']:
                        st.markdown("  - Detected by multiple methods (consensus anomaly)")
                
                st.markdown("**Content:**")
                st.markdown(item.get('content', 'No content available'), unsafe_allow_html=False)
                
                # Show metadata
                st.markdown("**Additional Information:**")
                meta_data = {k: v for k, v in item.items() if k.startswith('metadata_') and pd.notna(v)}
                
                if meta_data:
                    for key, value in meta_data.items():
                        clean_key = key.replace('metadata_', '').replace('_', ' ').title()
                        st.markdown(f"- **{clean_key}:** {value}")
                else:
                    st.markdown("No additional metadata available")
        
        # Show pagination controls
        st.write(f"Page {page} of {num_pages}")


if __name__ == "__main__":
    # Run the dashboard
    run_dashboard()