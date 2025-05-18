"""
dashboard_enhancements.py - Add threat analysis visualizations to dashboard

This file provides components to enhance your existing dashboard with
visualizations for threat categorization, urgency assessment, and emerging threats.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_threat_category_visualization(df_filtered):
    """Create visualizations for threat categories."""
    st.subheader("Threat Categories Analysis")
    
    if 'primary_category' not in df_filtered.columns:
        st.info("Threat categorization data not available. Run the threat categorization component first.")
        return
    
    # Create columns for side-by-side visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution pie chart
        category_counts = df_filtered['primary_category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        
        fig = px.pie(
            category_counts,
            values='count',
            names='category',
            title='Distribution of Threat Categories',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Categories by severity
        if 'severity_bin' in df_filtered.columns:
            # Create a crosstab of category vs severity
            category_severity = pd.crosstab(
                df_filtered['primary_category'], 
                df_filtered['severity_bin']
            )
            
            # Create a stacked bar chart
            fig = px.bar(
                category_severity, 
                barmode='stack',
                title='Threat Categories by Severity',
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
            st.info("Severity data not available for cross-analysis.")

def create_urgency_assessment_visualization(df_filtered):
    """Create visualizations for urgency assessment."""
    st.subheader("Urgency Assessment Analysis")
    
    if 'urgency_level' not in df_filtered.columns:
        st.info("Urgency assessment data not available. Run the urgency assessment component first.")
        return
    
    # Create columns for KPI cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_urgency_count = len(df_filtered[df_filtered['urgency_level'] == 'High'])
        st.metric(
            label="High Urgency Threats",
            value=high_urgency_count,
            delta=f"{high_urgency_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
        )
    
    with col2:
        medium_urgency_count = len(df_filtered[df_filtered['urgency_level'] == 'Medium'])
        st.metric(
            label="Medium Urgency Threats",
            value=medium_urgency_count,
            delta=f"{medium_urgency_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
        )
    
    with col3:
        # Critical AND high urgency - the most important threats
        if 'severity_bin' in df_filtered.columns:
            critical_high_count = len(df_filtered[
                (df_filtered['severity_bin'] == 'critical') & 
                (df_filtered['urgency_level'] == 'High')
            ])
            st.metric(
                label="Critical & High Urgency",
                value=critical_high_count,
                delta=f"{critical_high_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
            )
        else:
            st.metric(label="Critical & High Urgency", value="N/A")
    
    # Create columns for side-by-side visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Urgency level distribution
        urgency_counts = df_filtered['urgency_level'].value_counts().reset_index()
        urgency_counts.columns = ['urgency_level', 'count']
        
        # Map for consistent ordering
        level_order = {'High': 0, 'Medium': 1, 'Low': 2}
        urgency_counts['order'] = urgency_counts['urgency_level'].map(level_order)
        urgency_counts = urgency_counts.sort_values('order')
        
        # Define color map for urgency levels
        color_map = {
            'High': 'red',
            'Medium': 'orange',
            'Low': 'green'
        }
        
        # Create bar chart
        fig = px.bar(
            urgency_counts,
            x='urgency_level',
            y='count',
            color='urgency_level',
            color_discrete_map=color_map,
            title='Distribution of Urgency Levels',
            labels={'urgency_level': 'Urgency Level', 'count': 'Number of Threats'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Urgency vs. Severity matrix
        if 'severity_bin' in df_filtered.columns:
            # Create a crosstab of urgency vs severity
            urgency_severity = pd.crosstab(
                df_filtered['urgency_level'], 
                df_filtered['severity_bin'],
                normalize='columns'  # Normalize by column (severity)
            ) * 100  # Convert to percentage
            
            # Create the heatmap
            fig = px.imshow(
                urgency_severity,
                text_auto='.1f',
                labels=dict(x="Severity Level", y="Urgency Level", color="Percentage"),
                title="Urgency vs. Severity Matrix",
                color_continuous_scale='RdYlGn_r'  # Red (high) to green (low)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Severity data not available for cross-analysis.")

def create_emerging_threats_visualization(df_filtered):
    """Create visualizations for emerging threats detection."""
    st.subheader("Emerging Threats Analysis")
    
    if 'emerging_threat' not in df_filtered.columns:
        st.info("Emerging threats data not available. Run the emerging threats detection component first.")
        return
    
    try:
        # Extract emerging threat flag
        df_filtered['is_emerging'] = df_filtered['emerging_threat'].apply(
            lambda x: x.get('is_emerging', False) if isinstance(x, dict) else False
        )
        
        # Extract confidence score
        df_filtered['emerging_confidence'] = df_filtered['emerging_threat'].apply(
            lambda x: x.get('confidence', 0.0) if isinstance(x, dict) else 0.0
        )
        
        # Extract detection methods
        df_filtered['detection_methods'] = df_filtered['emerging_threat'].apply(
            lambda x: x.get('detection_methods', []) if isinstance(x, dict) else []
        )
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            emerging_count = df_filtered['is_emerging'].sum()
            st.metric(
                label="Emerging Threats",
                value=emerging_count,
                delta=f"{emerging_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%"
            )
        
        with col2:
            # High confidence emerging threats (>0.7)
            high_confidence = (df_filtered['is_emerging'] & (df_filtered['emerging_confidence'] > 0.7)).sum()
            st.metric(
                label="High Confidence Emerging",
                value=high_confidence,
                delta=f"{high_confidence/emerging_count*100:.1f}%" if emerging_count > 0 else "0%"
            )
        
        with col3:
            # Critical emerging threats
            if 'severity_bin' in df_filtered.columns:
                critical_emerging = (df_filtered['is_emerging'] & (df_filtered['severity_bin'] == 'critical')).sum()
                st.metric(
                    label="Critical & Emerging",
                    value=critical_emerging,
                    delta=f"{critical_emerging/emerging_count*100:.1f}%" if emerging_count > 0 else "0%"
                )
            else:
                st.metric(label="Critical & Emerging", value="N/A")
        
        # Create columns for side-by-side visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Detection methods distribution
            if df_filtered['detection_methods'].any():
                # Flatten the list of detection methods
                all_methods = []
                for methods in df_filtered['detection_methods']:
                    all_methods.extend(methods)
                
                # Count occurrences of each method
                method_counts = pd.Series(all_methods).value_counts().reset_index()
                method_counts.columns = ['method', 'count']
                
                # Map method names to more readable labels
                method_map = {
                    'zeroday_candidate': 'Zero-day Candidate',
                    'term_spike': 'Term Spike',
                    'statistical_anomaly': 'Statistical Anomaly',
                    'emerging_cluster': 'Emerging Cluster'
                }
                method_counts['method'] = method_counts['method'].map(lambda x: method_map.get(x, x))
                
                # Create bar chart
                fig = px.bar(
                    method_counts,
                    x='method',
                    y='count',
                    color='method',
                    title='Detection Methods for Emerging Threats',
                    labels={'method': 'Detection Method', 'count': 'Number of Threats'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detection methods data available.")
        
        with col2:
            # Confidence score distribution
            if df_filtered['is_emerging'].any():
                # Filter to just emerging threats
                emerging_df = df_filtered[df_filtered['is_emerging']]
                
                # Create histogram
                fig = px.histogram(
                    emerging_df,
                    x='emerging_confidence',
                    nbins=10,
                    color_discrete_sequence=['purple'],
                    title='Emerging Threats Confidence Distribution',
                    labels={'emerging_confidence': 'Confidence Score', 'count': 'Number of Threats'}
                )
                
                # Add vertical threshold lines
                fig.add_shape(
                    type="line",
                    x0=0.7, y0=0, x1=0.7, y1=emerging_df['emerging_confidence'].value_counts().max() * 1.1,
                    line=dict(color="red", width=2, dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=0.4, y0=0, x1=0.4, y1=emerging_df['emerging_confidence'].value_counts().max() * 1.1,
                    line=dict(color="orange", width=2, dash="dash")
                )
                
                # Add annotations
                fig.add_annotation(
                    x=0.7, y=emerging_df['emerging_confidence'].value_counts().max() * 0.8,
                    text="High Confidence", showarrow=False,
                    font=dict(color="red")
                )
                
                fig.add_annotation(
                    x=0.4, y=emerging_df['emerging_confidence'].value_counts().max() * 0.6,
                    text="Medium Confidence", showarrow=False,
                    font=dict(color="orange")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emerging threats data available.")
        
        # Emerging threats by category
        if 'primary_category' in df_filtered.columns and df_filtered['is_emerging'].any():
            st.subheader("Emerging Threats by Category")
            
            # Create crosstab
            category_emerging = pd.crosstab(
                df_filtered['primary_category'],
                df_filtered['is_emerging'],
                normalize='index'
            ) * 100  # Convert to percentage
            
            # Make sure True column exists
            if True not in category_emerging.columns:
                category_emerging[True] = 0
            
            # Sort by percentage emerging
            category_emerging = category_emerging.sort_values(True, ascending=False)
            
            # Create bar chart of percentage emerging by category
            fig = px.bar(
                category_emerging.reset_index(),
                x='primary_category',
                y=True,  # True column = percent emerging
                color=True,
                color_continuous_scale='Viridis',
                title='Percentage of Emerging Threats by Category',
                labels={'primary_category': 'Threat Category', True: 'Percentage Emerging'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing emerging threats data: {str(e)}")

# Add these components to your dashboard
def add_to_dashboard():
    """
    Add these components to your Streamlit dashboard.
    
    Example usage in your dashboard/app.py:
    
    import dashboard_enhancements as de
    
    # In your app
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Threat Categories", "Urgency Assessment", "Emerging Threats"
    ])
    
    with tab2:
        st.header("Threat Categories")
        de.create_threat_category_visualization(df_filtered)
    
    with tab3:
        st.header("Urgency Assessment")
        de.create_urgency_assessment_visualization(df_filtered)
    
    with tab4:
        st.header("Emerging Threats")
        de.create_emerging_threats_visualization(df_filtered)
    """
    pass

if __name__ == "__main__":
    st.title("Dashboard Enhancement Demo")
    st.warning("This is just a demo of the visualizations. Add these components to your main dashboard.")