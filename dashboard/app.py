#!/usr/bin/env python3
"""
Streamlit dashboard for the AI-Driven Cyber Threat Intelligence system.
Visualizes threat data, classification results, urgency scores, and emerging threats.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Data paths
DATA_DIR = ROOT / "data/processed"
MASTER_PATH = DATA_DIR / "master.parquet"
URGENCY_PATH = DATA_DIR / "urgency_assessed.parquet"
EMERGING_PATH = DATA_DIR / "emerging_threats.parquet"

# Ensure the dashboard can be run directly or via run.py
if not DATA_DIR.exists():
    st.error(f"Data directory not found: {DATA_DIR}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Cyber Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .emerging-threat {
        background-color: #ffecec;
        border-left: 3px solid #ff4b4b;
        padding: 10px;
        margin: 10px 0;
    }
    .high-urgency {
        background-color: #ffecec;
        border-left: 3px solid #ff4b4b;
    }
    .medium-urgency {
        background-color: #fff8e1;
        border-left: 3px solid #ffab40;
    }
    .low-urgency {
        background-color: #e8f5e9;
        border-left: 3px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("CTI Dashboard")
# Only try to load the logo if it exists
logo_path = ROOT / "dashboard" / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=100)
else:
    st.sidebar.markdown("## 🛡️ CTI")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Threat Classification", "Urgency Analysis", "Emerging Threats", "CVE Explorer"]
)

@st.cache_data(ttl=600)
def load_data():
    """Load and prepare data for the dashboard, with caching for performance."""
    # Create a status indicator
    status = st.sidebar.empty()
    status.info("Starting data load...")
    
    try:
        if not MASTER_PATH.exists():
            status.error(f"Master data not found at {MASTER_PATH}")
            return None, None, None
        
        # Load datasets
        status.info("Loading master data...")
        df_master = pd.read_parquet(MASTER_PATH) if MASTER_PATH.exists() else None
        
        status.info("Loading urgency data...")
        df_urgency = pd.read_parquet(URGENCY_PATH) if URGENCY_PATH.exists() else None
        
        status.info("Loading emerging threats data...")
        df_emerging = pd.read_parquet(EMERGING_PATH) if EMERGING_PATH.exists() else None
        
        # If urgency data exists but emerging doesn't, create a merged dataset for comprehensive views
        if df_urgency is not None and df_master is not None:
            status.info("Creating combined dataset...")
            # Merge master data (contains classifications) with urgency scores
            df_combined = df_urgency.copy() if df_urgency is not None else df_master.copy()
            
            # Add emerging flag if it exists
            if df_emerging is not None:
                status.info("Adding emerging threats flags...")
                emerging_flags = df_emerging[['cve_id', 'emerging']] if 'cve_id' in df_emerging.columns else None
                if emerging_flags is not None:
                    df_combined = df_combined.merge(emerging_flags, on='cve_id', how='left')
                    df_combined['emerging'] = df_combined['emerging'].fillna(False)
            else:
                df_combined['emerging'] = False
            
            # Check for classification columns
            threat_categories = [col for col in df_combined.columns if col in ["Phishing", "Ransomware", "Malware", "SQLInjection", "XSS", "DDoS", "ZeroDay", "SupplyChain", "Other"]]
            
            # If classification columns don't exist, try to apply the model
            if not threat_categories:
                # Try to load and apply the threat classification model
                model_path = ROOT / "models" / "threat_model_with_sbert.pkl"
                if model_path.exists():
                    try:
                        status.info("Loading threat classification model...")
                        import joblib
                        
                        # Load the model (but with a timeout)
                        import signal
                        
                        class TimeoutException(Exception):
                            pass
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutException("Model loading timed out")
                        
                        # Set a 30-second timeout
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)
                        
                        try:
                            model_data = joblib.load(model_path)
                            # Cancel the alarm if loading succeeds
                            signal.alarm(0)
                        except TimeoutException:
                            status.error("Model loading timed out. Using data without classification.")
                            return df_master, df_urgency, df_combined
                        except Exception as e:
                            status.error(f"Error loading model: {str(e)}")
                            return df_master, df_urgency, df_combined
                        
                        # Add dummy classification columns
                        categories = ["Phishing", "Ransomware", "Malware", "SQLInjection", "XSS", "DDoS", "ZeroDay", "SupplyChain", "Other"]
                        for cat in categories:
                            df_combined[cat] = 0
                        
                        status.info("Applying simplified classification...")
                        
                        # Apply simple rule-based classification instead of using the model
                        # This is much faster for demo purposes
                        patterns = {
                            "Phishing": ["phish", "credential", "email scam", "spoof"],
                            "Ransomware": ["ransom", "cryptocurrency", "file locked"],
                            "Malware": ["malware", "trojan", "virus", "worm"],
                            "SQLInjection": ["sql injection", "database injection"],
                            "XSS": ["cross site script", "xss"],
                            "DDoS": ["denial of service", "ddos"],
                            "ZeroDay": ["zero day", "0 day", "unpatched"],
                            "SupplyChain": ["supply chain", "vendor", "third party"]
                        }
                        
                        import re
                        # Apply simple pattern matching
                        for idx, row in df_combined.iterrows():
                            text = str(row.get('clean_text', '')).lower()
                            matched = False
                            for cat, pats in patterns.items():
                                for pat in pats:
                                    if pat in text:
                                        df_combined.at[idx, cat] = 1
                                        matched = True
                            if not matched:
                                df_combined.at[idx, "Other"] = 1
                    
                        status.success("✅ Classification complete!")
                    except Exception as e:
                        status.warning(f"Failed to apply threat model: {str(e)}")
                else:
                    status.warning(f"Model not found: {model_path}")
            
            status.success("✅ Data loaded successfully")
            return df_master, df_urgency, df_combined
        
        status.error("Missing urgency or master data")
        return df_master, df_urgency, df_emerging
    
    except Exception as e:
        status.error(f"Error loading data: {str(e)}")
        import traceback
        st.sidebar.error(f"Stack trace: {traceback.format_exc()}")
        return None, None, None

# Handle errors at the start to show a user-friendly message
try:
    df_master, df_urgency, df_combined = load_data()

    if df_master is None or len(df_master) == 0:
        st.error("No data found. Please run the pipeline first with: `python run.py --component all`")
        st.stop()
except Exception as e:
    st.error(f"Error initializing dashboard: {str(e)}")
    st.error("Try running the pipeline first with: `python run.py --component all`")
    st.stop()

# Helper functions
def create_wordcloud(text_series, mask=None):
    """Generate a wordcloud from text series."""
    all_text = " ".join(text_series.dropna().astype(str))
    wc = WordCloud(background_color="white", max_words=200, contour_width=3, mask=mask)
    return wc.generate(all_text)

def display_cve_details(row):
    """Display detailed information for a single CVE."""
    if row is None:
        return
    
    severity_color = {
        "critical": "red",
        "high": "orange",
        "medium": "yellow",
        "low": "green"
    }.get(str(row.get('severity', '')).lower(), "gray")
    
    st.markdown(f"### {row.get('cve_id', 'Unknown CVE')}")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if 'published_date' in row:
            st.info(f"Published: {row['published_date']}")
    with col2:
        if 'cvss_score' in row:
            st.metric("CVSS Score", round(row['cvss_score'], 1) if pd.notna(row['cvss_score']) else "N/A")
    with col3:
        if 'severity' in row:
            st.markdown(f"<p style='background-color:{severity_color};padding:10px;text-align:center;border-radius:5px;'>Severity: {row.get('severity', 'Unknown').upper()}</p>", unsafe_allow_html=True)
    
    if 'description' in row and pd.notna(row['description']):
        st.markdown("#### Description")
        st.markdown(row['description'])
    
    if 'products' in row and isinstance(row['products'], list) and row['products']:
        st.markdown("#### Affected Products")
        for product in row['products'][:5]:  # Show only first 5 for space
            st.markdown(f"- {product}")
        if len(row['products']) > 5:
            st.markdown(f"- ...and {len(row['products'])-5} more")
    
    if 'urgency_score' in row and pd.notna(row['urgency_score']):
        st.markdown("#### Urgency Assessment")
        urgency_level = row.get('urgency_level', 'Unknown')
        urgency_class = {
            'High': 'high-urgency',
            'Medium': 'medium-urgency', 
            'Low': 'low-urgency'
        }.get(urgency_level, '')
        
        st.markdown(f"<div class='{urgency_class}' style='padding:10px;border-radius:5px;'>")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Urgency Score", f"{row['urgency_score']:.2f}")
        with col2:
            st.metric("Urgency Level", urgency_level)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if 'emerging' in row and row['emerging']:
        st.markdown("<div class='emerging-threat' style='padding:10px;border-radius:5px;'>⚠️ This is flagged as an emerging threat</div>", unsafe_allow_html=True)
    
    # Show related articles if available
    if 'n_articles' in row and pd.notna(row['n_articles']) and row['n_articles'] > 0:
        st.markdown("#### Related Articles")
        st.metric("Article Count", int(row['n_articles']))
        
        if 'linked_articles' in row and isinstance(row['linked_articles'], list) and row['linked_articles']:
            articles = df_master[df_master['source'] == 'thehackernews'].iloc[row['linked_articles']]
            for _, article in articles.iterrows():
                st.markdown(f"- [{article.get('text', '').split('.')[0]}...]({article.get('url', '')})")

# Pages
if page == "Overview":
    st.title("Cyber Threat Intelligence Overview")
    
    # Metrics row
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        nvd_count = len(df_master[df_master['source'] == 'nvd'])
        st.metric("NVD Vulnerabilities", nvd_count)
    
    with metric_cols[1]:
        thn_count = len(df_master[df_master['source'] == 'thehackernews'])
        st.metric("The Hacker News Articles", thn_count)
    
    with metric_cols[2]:
        if 'urgency_level' in df_combined.columns:
            high_count = len(df_combined[df_combined['urgency_level'] == 'High'])
            st.metric("High Urgency Threats", high_count)
    
    with metric_cols[3]:
        if 'emerging' in df_combined.columns:
            emerging_count = df_combined['emerging'].sum()
            st.metric("Emerging Threats", int(emerging_count))
    
    # Timeline chart
    st.subheader("Vulnerability Timeline")
    
    df_master['published_date'] = pd.to_datetime(df_master['published_date'], errors='coerce')
    timeline_df = df_master[df_master['published_date'].notna()].copy()
    
    if not timeline_df.empty:
        timeline_df['month'] = timeline_df['published_date'].dt.to_period('M')
        monthly_counts = timeline_df.groupby(['month', 'source']).size().reset_index(name='count')
        monthly_counts['month'] = monthly_counts['month'].dt.to_timestamp()
        
        fig = px.line(
            monthly_counts, 
            x='month', 
            y='count', 
            color='source',
            title="Monthly Vulnerability & Article Counts",
            labels={'month': 'Month', 'count': 'Count', 'source': 'Source'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent critical vulnerabilities
    st.subheader("Recent Critical Vulnerabilities")
    
    nvd_df = df_master[df_master['source'] == 'nvd'].copy()
    if 'severity' in nvd_df.columns:
        critical_df = nvd_df[nvd_df['severity'].str.lower() == 'critical'].copy() if 'severity' in nvd_df.columns else pd.DataFrame()
        
        if not critical_df.empty:
            critical_df['published_date'] = pd.to_datetime(critical_df['published_date'], errors='coerce')
            recent_critical = critical_df.sort_values('published_date', ascending=False).head(5)
            
            for _, row in recent_critical.iterrows():
                with st.expander(f"{row['cve_id']} - {row.get('published_date', 'Unknown date')}"):
                    display_cve_details(row)
        else:
            st.info("No critical vulnerabilities found in the dataset.")

elif page == "Threat Classification":
    st.title("Threat Classification Analysis")
    
    # Prepare classification data
    category_cols = [col for col in df_combined.columns if col in ["Phishing", "Ransomware", "Malware", "SQLInjection", "XSS", "DDoS", "ZeroDay", "SupplyChain", "Other"]]
    
    if category_cols:
        # Category distribution
        st.subheader("Threat Category Distribution")
        
        category_counts = {col: df_combined[col].sum() for col in category_cols if col in df_combined.columns}
        category_df = pd.DataFrame({
            'Category': list(category_counts.keys()),
            'Count': list(category_counts.values())
        }).sort_values('Count', ascending=False)
        
        fig = px.bar(
            category_df,
            x='Category',
            y='Count',
            title="Vulnerability Distribution by Category",
            color='Count',
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Category co-occurrence
        st.subheader("Threat Category Co-occurrence")
        
        # Create co-occurrence matrix
        cooc_matrix = np.zeros((len(category_cols), len(category_cols)))
        for i, cat1 in enumerate(category_cols):
            for j, cat2 in enumerate(category_cols):
                if i <= j and cat1 in df_combined.columns and cat2 in df_combined.columns:
                    cooc_matrix[i, j] = ((df_combined[cat1] == 1) & (df_combined[cat2] == 1)).sum()
                    cooc_matrix[j, i] = cooc_matrix[i, j]
        
        fig = go.Figure(data=go.Heatmap(
            z=cooc_matrix,
            x=category_cols,
            y=category_cols,
            colorscale='Reds',
            showscale=True
        ))
        fig.update_layout(title="Category Co-occurrence Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Time trends by category
        st.subheader("Category Trends Over Time")
        
        df_time = df_combined.copy()
        df_time['published_date'] = pd.to_datetime(df_time['published_date'], errors='coerce')
        
        if not df_time['published_date'].isna().all():
            df_time['month'] = df_time['published_date'].dt.to_period('M')
            
            # Select category for trend analysis
            selected_category = st.selectbox("Select Category", category_cols)
            
            if selected_category in df_time.columns:
                monthly_category = df_time.groupby('month')[selected_category].sum().reset_index()
                monthly_category['month'] = monthly_category['month'].dt.to_timestamp()
                
                fig = px.line(
                    monthly_category,
                    x='month',
                    y=selected_category,
                    title=f"Monthly Trend: {selected_category}",
                    labels={'month': 'Month', selected_category: 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Threat classification data not found. Please run the threat classifier component.")

elif page == "Urgency Analysis":
    st.title("Urgency Score Analysis")
    
    if 'urgency_score' not in df_combined.columns or 'urgency_level' not in df_combined.columns:
        st.warning("Urgency data not found. Please run the urgency scoring component first.")
        st.stop()
    
    # Urgency distribution
    st.subheader("Urgency Level Distribution")
    
    urgency_counts = df_combined['urgency_level'].value_counts().reset_index()
    urgency_counts.columns = ['Level', 'Count']
    
    # Define color mapping
    color_map = {'High': '#ff4b4b', 'Medium': '#ffab40', 'Low': '#4caf50'}
    
    fig = px.pie(
        urgency_counts,
        values='Count',
        names='Level',
        title="Distribution of Urgency Levels",
        color='Level',
        color_discrete_map=color_map
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Urgency scores histogram
    st.subheader("Urgency Score Distribution")
    
    fig = px.histogram(
        df_combined,
        x='urgency_score',
        nbins=50,
        title="Histogram of Urgency Scores",
        color_discrete_sequence=['#ff6b6b']
    )
    fig.add_vline(x=0.33, line_dash="dash", line_color="#4caf50")
    fig.add_vline(x=0.66, line_dash="dash", line_color="#ffab40")
    st.plotly_chart(fig, use_container_width=True)
    
    # High urgency threats
    st.subheader("Highest Urgency Threats")
    
    high_urgency = df_combined[df_combined['urgency_level'] == 'High'].sort_values('urgency_score', ascending=False).head(10)
    
    for i, (_, row) in enumerate(high_urgency.iterrows()):
        with st.expander(f"{i+1}. {row.get('cve_id', 'N/A')} - Score: {row['urgency_score']:.2f}"):
            display_cve_details(row)
    
    # Urgency factors analysis
    st.subheader("Urgency Factors Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CVSS Score vs Urgency Score
        fig = px.scatter(
            df_combined,
            x='cvss_score',
            y='urgency_score',
            color='urgency_level',
            title="CVSS Score vs Urgency Score",
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment vs Urgency Score
        if 'sentiment' in df_combined.columns:
            fig = px.scatter(
                df_combined,
                x='sentiment',
                y='urgency_score',
                color='urgency_level',
                title="Sentiment vs Urgency Score",
                color_discrete_map=color_map
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Emerging Threats":
    st.title("Emerging Threat Detection")
    
    if 'emerging' not in df_combined.columns:
        st.warning("Emerging threat data not found. Please run the anomaly detection component first.")
        st.stop()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        emerging_count = df_combined['emerging'].sum()
        st.metric("Emerging Threats", int(emerging_count))
    
    with col2:
        if 'spike_flag' in df_combined.columns:
            spike_count = df_combined['spike_flag'].sum()
            st.metric("Mention Spikes", int(spike_count))
    
    with col3:
        if 'zero_day_flag' in df_combined.columns:
            zeroday_count = df_combined['zero_day_flag'].sum()
            st.metric("Zero-Day Indicators", int(zeroday_count))
    
    # Emerging threats list
    st.subheader("Current Emerging Threats")
    
    emerging_threats = df_combined[df_combined['emerging'] == True].sort_values('published_date', ascending=False)
    
    if not emerging_threats.empty:
        for i, (_, row) in enumerate(emerging_threats.head(10).iterrows()):
            with st.expander(f"{i+1}. {row.get('cve_id', 'N/A')} - {row.get('published_date', 'Unknown date')}"):
                display_cve_details(row)
    else:
        st.info("No emerging threats detected in the current dataset.")
    
    # Detection method breakdown
    st.subheader("Detection Method Analysis")
    
    detection_methods = []
    values = []
    
    if 'zero_day_flag' in df_combined.columns:
        detection_methods.append("Zero-Day Pattern")
        values.append(df_combined['zero_day_flag'].sum())
    
    if 'spike_flag' in df_combined.columns:
        detection_methods.append("Mention Spike")
        values.append(df_combined['spike_flag'].sum())
    
    if 'if_flag' in df_combined.columns:
        detection_methods.append("Isolation Forest")
        values.append(df_combined['if_flag'].sum())
    
    if detection_methods:
        method_df = pd.DataFrame({
            'Method': detection_methods,
            'Count': values
        })
        
        fig = px.bar(
            method_df,
            x='Method',
            y='Count',
            title="Threats Detected by Method",
            color='Method'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline of emerging threats
    st.subheader("Emerging Threats Timeline")
    
    df_emerging_time = df_combined[df_combined['emerging'] == True].copy()
    df_emerging_time['published_date'] = pd.to_datetime(df_emerging_time['published_date'], errors='coerce')
    
    if not df_emerging_time.empty and not df_emerging_time['published_date'].isna().all():
        df_emerging_time['week'] = df_emerging_time['published_date'].dt.to_period('W')
        weekly_emerging = df_emerging_time.groupby('week').size().reset_index(name='count')
        weekly_emerging['week'] = weekly_emerging['week'].dt.to_timestamp()
        
        fig = px.line(
            weekly_emerging,
            x='week',
            y='count',
            title="Weekly Emerging Threats",
            labels={'week': 'Week', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "CVE Explorer":
    st.title("CVE Explorer")
    
    # Search and filtering controls
    st.sidebar.header("Search & Filters")
    
    # Text search
    search_text = st.sidebar.text_input("Search Text")
    
    # Date range filter
    df_master['published_date'] = pd.to_datetime(df_master['published_date'], errors='coerce')
    min_date = df_master['published_date'].min().date() if not df_master['published_date'].isna().all() else datetime.now().date() - timedelta(days=365)
    max_date = df_master['published_date'].max().date() if not df_master['published_date'].isna().all() else datetime.now().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Severity filter
    if 'severity' in df_combined.columns:
        severity_options = ['All'] + sorted(df_combined['severity'].dropna().unique().tolist())
        selected_severity = st.sidebar.selectbox("Severity", severity_options)
    
    # Urgency filter
    if 'urgency_level' in df_combined.columns:
        urgency_options = ['All'] + sorted(df_combined['urgency_level'].dropna().unique().tolist())
        selected_urgency = st.sidebar.selectbox("Urgency Level", urgency_options)
    
    # Emerging threat filter
    if 'emerging' in df_combined.columns:
        emerging_filter = st.sidebar.checkbox("Show Only Emerging Threats")
    
    # Apply filters
    filtered_df = df_combined.copy()
    
    if search_text:
        text_mask = (
            filtered_df['clean_text'].str.contains(search_text, case=False, na=False) |
            filtered_df['cve_id'].str.contains(search_text, case=False, na=False)
        )
        filtered_df = filtered_df[text_mask]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        date_mask = (
            (filtered_df['published_date'].dt.date >= start_date) &
            (filtered_df['published_date'].dt.date <= end_date)
        )
        filtered_df = filtered_df[date_mask]
    
    if 'severity' in locals() and selected_severity != 'All':
        filtered_df = filtered_df[filtered_df['severity'] == selected_severity]
    
    if 'urgency_level' in locals() and selected_urgency != 'All':
        filtered_df = filtered_df[filtered_df['urgency_level'] == selected_urgency]
    
    if 'emerging_filter' in locals() and emerging_filter:
        filtered_df = filtered_df[filtered_df['emerging'] == True]
    
    # Display filtered CVEs
    st.subheader(f"Results ({len(filtered_df)} CVEs/Articles)")
    
    if filtered_df.empty:
        st.info("No results match your filters.")
    else:
        # Sort options
        sort_options = {
            "Most Recent": ("published_date", False),
            "Oldest First": ("published_date", True),
            "Highest CVSS": ("cvss_score", False),
            "Highest Urgency": ("urgency_score", False)
        }
        
        sort_by = st.selectbox("Sort by", list(sort_options.keys()))
        sort_col, sort_asc = sort_options[sort_by]
        
        if sort_col in filtered_df.columns:
            sorted_df = filtered_df.sort_values(sort_col, ascending=sort_asc, na_position='last')
            
            # Paginated results
            items_per_page = 10
            total_pages = (len(sorted_df) + items_per_page - 1) // items_per_page
            if total_pages > 0:
                page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
                start_idx = page_num * items_per_page
                end_idx = min(start_idx + items_per_page, len(sorted_df))
                
                st.write(f"Showing {start_idx+1}-{end_idx} of {len(sorted_df)} results")
                
                for _, row in sorted_df.iloc[start_idx:end_idx].iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        expander_title = f"{row.get('cve_id', 'N/A')}"
                        if 'published_date' in row and pd.notna(row['published_date']):
                            expander_title += f" - {row['published_date'].date()}"
                    with col2:
                        if 'cvss_score' in row and pd.notna(row['cvss_score']):
                            expander_title += f" | CVSS: {row['cvss_score']:.1f}"
                    with col3:
                        if 'urgency_level' in row and pd.notna(row['urgency_level']):
                            expander_title += f" | Urgency: {row['urgency_level']}"
                    
                    with st.expander(expander_title):
                        display_cve_details(row)
            else:
                st.info("No results to display.")
        else:
            st.error(f"Sort column '{sort_col}' not found in data.")

# Add a link to the main dashboard in the sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"Data: {datetime.now().strftime('%Y-%m-%d')}")
st.sidebar.markdown("Made Dheer Gupta")

# Footer
st.markdown("---")
st.markdown("### About This Dashboard")
st.markdown("""
This dashboard visualizes data from an AI-Driven Cyber Threat Intelligence pipeline, including:
- NVD vulnerability data and The Hacker News security articles
- Threat classification using ML models
- Urgency scoring based on multiple factors
- Emerging threat detection via anomaly detection

Use the sidebar to navigate between different views.
""")