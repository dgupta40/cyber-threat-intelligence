"""
Enhanced visualization components for the Streamlit dashboard.
These functions can be imported into your existing dashboard/app.py file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def create_kpi_cards(df_filtered):
    """Create KPI metric cards for the dashboard."""
    # Create columns for KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Threats",
            value=len(df_filtered),
            delta=f"+{len(df_filtered) // 10}" if len(df_filtered) > 0 else "0"  # Example delta
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
                value=last_updated.strftime("%Y-%m-%d") if isinstance(last_updated, pd.Timestamp) else str(last_updated)
            )
        else:
            st.metric(label="Last Updated", value="N/A")

def create_threat_timeline(df_filtered):
    """Create an enhanced timeline visualization of threats."""
    st.subheader("Threat Timeline")
    
    if 'date' in df_filtered.columns and not df_filtered['date'].empty:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_filtered['date']):
            try:
                df_filtered['date'] = pd.to_datetime(df_filtered['date'])
            except Exception as e:
                st.warning(f"Could not convert dates: {str(e)}")
                return
        
        # Group by date
        df_time = df_filtered.groupby(df_filtered['date'].dt.date).size().reset_index(name='count')
        
        # Calculate 7-day moving average
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
        
        # Add moving average if available
        if 'category' in df_filtered.columns:
            # Group by date and category
            df_cat_time = df_filtered.groupby([df_filtered['date'].dt.date, 'category']).size().reset_index(name='count')
            
            # Create a multi-line plot by category
            fig = px.line(
                df_cat_time, 
                x='date', 
                y='count',
                color='category',
                title='Threats by Category Over Time',
                labels={'date': 'Date', 'count': 'Number of Threats', 'category': 'Category'}
            )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Threats",
            legend_title="Category" if 'category' in df_filtered.columns else None,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Date information not available for time series visualization")

def create_risk_distribution(df_filtered):
    """Create enhanced risk distribution visualizations."""
    st.subheader("Risk Distribution Analysis")
    
    if 'risk_level' in df_filtered.columns and 'risk_score' in df_filtered.columns:
        # Create columns for side-by-side visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution
            risk_counts = df_filtered['risk_level'].value_counts().reset_index()
            risk_counts.columns = ['risk_level', 'count']
            
            # Define color map for risk levels
            color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green', 'Unknown': 'gray'}
            
            # Create pie chart
            fig = px.pie(
                risk_counts,
                values='count',
                names='risk_level',
                color='risk_level',
                color_discrete_map=color_map,
                title='Threats by Risk Level',
                hole=0.4  # Donut chart
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk score distribution histogram
            fig = px.histogram(
                df_filtered,
                x='risk_score',
                nbins=20,
                color_discrete_sequence=['#3366CC'],
                title='Distribution of Risk Scores',
                labels={'risk_score': 'Risk Score', 'count': 'Number of Threats'}
            )
            
            # Add threshold lines
            fig.add_shape(
                type="line",
                x0=0.7, y0=0, x1=0.7, y1=df_filtered['risk_score'].value_counts().max() * 1.1,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.add_shape(
                type="line",
                x0=0.3, y0=0, x1=0.3, y1=df_filtered['risk_score'].value_counts().max() * 1.1,
                line=dict(color="orange", width=2, dash="dash")
            )
            
            # Add annotations
            fig.add_annotation(
                x=0.7, y=df_filtered['risk_score'].value_counts().max() * 0.9,
                text="High Risk Threshold", showarrow=False,
                font=dict(color="red")
            )
            
            fig.add_annotation(
                x=0.3, y=df_filtered['risk_score'].value_counts().max() * 0.8,
                text="Medium Risk Threshold", showarrow=False,
                font=dict(color="orange")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add risk trend over time
        if 'date' in df_filtered.columns and not df_filtered['date'].empty:
            st.subheader("Risk Score Trends")
            
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
                x0=df_risk_avg['date'].min(), y0=0.7,
                x1=df_risk_avg['date'].max(), y1=0.7,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.add_shape(
                type="line",
                x0=df_risk_avg['date'].min(), y0=0.3,
                x1=df_risk_avg['date'].max(), y1=0.3,
                line=dict(color="orange", width=2, dash="dash")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Risk assessment information not available")

def create_category_analysis(df_filtered):
    """Create enhanced category analysis visualizations."""
    st.subheader("Threat Category Analysis")
    
    if 'category' in df_filtered.columns and not df_filtered['category'].empty:
        # Create category counts
        category_counts = df_filtered['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        
        # Create columns for side-by-side visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                category_counts,
                x='category',
                y='count',
                title='Threats by Category',
                labels={'category': 'Category', 'count': 'Number of Threats'},
                color='category'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(
                category_counts,
                values='count',
                names='category',
                title='Category Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category and risk level relationship
        if 'risk_level' in df_filtered.columns:
            st.subheader("Risk Level by Category")
            
            # Create cross-tabulation
            category_risk = pd.crosstab(df_filtered['category'], df_filtered['risk_level'])
            
            # Convert to percentage for better visualization
            category_risk_pct = category_risk.div(category_risk.sum(axis=1), axis=0) * 100
            
            # Create stacked bar chart
            fig = px.bar(
                category_risk_pct,
                title='Risk Level by Category (%)',
                labels={'value': 'Percentage', 'index': 'Category'},
                barmode='stack',
                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a heatmap visualization
            st.subheader("Category-Risk Heatmap")
            
            # Create heatmap
            fig = px.imshow(
                category_risk,
                text_auto=True,
                aspect="auto",
                title='Number of Threats by Category and Risk Level',
                labels=dict(x="Risk Level", y="Category", color="Count"),
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Category information not available")

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
        if 'date' in df_filtered.columns and not df_filtered['date'].empty:
            st.subheader("Anomalies Over Time")
            
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
            fig.add_annotation(
                x=df_combined['date'].iloc[len(df_combined)//2], y=11,
                text="Alert Threshold",
                showarrow=False,
                font=dict(color="red")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Anomaly detection information not available")

def create_word_cloud(df_filtered):
    """Create an enhanced word cloud visualization."""
    st.subheader("Threat Intelligence Word Cloud")
    
    if 'content' in df_filtered.columns:
        # Combine all text
        all_text = ' '.join(df_filtered['content'].fillna('').astype(str))
        
        # Generate word cloud
        if all_text:
            try:
                # Set up columns for word cloud and settings
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    # Word cloud settings
                    max_words = st.slider("Max Words", min_value=50, max_value=200, value=100, step=10)
                    colormap = st.selectbox("Color Map", options=["viridis", "plasma", "inferno", "magma", "cividis"])
                    
                    # Category-specific word cloud
                    if 'category' in df_filtered.columns:
                        selected_category = st.selectbox(
                            "Category Filter", 
                            options=["All"] + sorted(df_filtered['category'].unique().tolist())
                        )
                
                with col1:
                    # Filter by category if selected
                    if 'category' in df_filtered.columns and selected_category != "All":
                        category_text = ' '.join(
                            df_filtered[df_filtered['category'] == selected_category]['content'].fillna('').astype(str)
                        )
                        text_to_use = category_text if category_text else all_text
                        title = f"Word Cloud for {selected_category} Threats"
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
                st.error(f"Error generating word cloud: {str(e)}")
        else:
            st.info("No text content available for word cloud")
    else:
        st.info("Text content not available for word cloud generation")

def create_enhanced_details_view(search_results):
    """Create an enhanced details view for threats."""
    st.subheader("Detailed Threat Analysis")
    
    # Display results as cards
    for i, (_, item) in enumerate(search_results.iterrows()):
        # Create an expander for each item
        with st.expander(f"{i+1}. {item.get('title', 'No Title')}", expanded=False):
            # Create columns for metadata and content
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Metadata")
                
                # Display basic info
                st.markdown(f"**Source:** {item.get('source', 'Unknown')}")
                
                if 'date' in item and pd.notna(item['date']):
                    date_str = item['date'].strftime("%Y-%m-%d") if isinstance(item['date'], pd.Timestamp) else str(item['date'])
                    st.markdown(f"**Date:** {date_str}")
                
                if 'category' in item and pd.notna(item['category']):
                    st.markdown(f"**Category:** {item['category']}")
                
                # Display risk information
                if 'risk_level' in item and pd.notna(item['risk_level']):
                    risk_color = "red" if item['risk_level'] == 'High' else "orange" if item['risk_level'] == 'Medium' else "green"
                    st.markdown(f"**Risk Level:** <span style='color:{risk_color};font-weight:bold'>{item['risk_level']}</span>", unsafe_allow_html=True)
                    
                    if 'risk_score' in item and pd.notna(item['risk_score']):
                        st.markdown(f"**Risk Score:** {item['risk_score']:.2f}")
                
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
                
                # Display content with highlighting
                if 'content' in item and pd.notna(item['content']):
                    content = item['content']
                    # Highlight key terms
                    if 'category' in item and pd.notna(item['category']):
                        category = item['category']
                        content = content.replace(category, f"<span style='background-color:yellow'>{category}</span>")
                    
                    st.markdown(content, unsafe_allow_html=True)
                else:
                    st.markdown("No content available")
                
                # Display related threats if available
                if 'related_threats' in item and isinstance(item['related_threats'], list) and item['related_threats']:
                    st.markdown("### Related Threats")
                    for related in item['related_threats']:
                        st.markdown(f"- {related}")