#!/usr/bin/env python3
"""
Improved Streamlit dashboard for the AI-Driven Cyber Threat Intelligence system.
Enhanced with better error handling, performance optimizations, and cleaner UI.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from database import load_table
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Page configuration
st.set_page_config(
    page_title="Cyber Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Alert styling */
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .alert-high {
        background-color: #ff8800;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .alert-medium {
        background-color: #ffbb33;
        color: #333;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .alert-low {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Data paths
DB_PATH = ROOT / "data/processed/cti.db"
MODEL_DIR = ROOT / "models"

# Initialize session state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_master = None
    st.session_state.df_urgency = None
    st.session_state.df_emerging = None

# Sidebar
with st.sidebar:
    st.title("🛡️ CTI Dashboard")
    st.markdown("---")

    # Navigation
    page = st.selectbox(
        "Select Page",
        [
            "📊 Overview",
            "🎯 Threat Analysis",
            "⚡ Urgency Monitor",
            "🚨 Emerging Threats",
            "🔍 CVE Explorer",
            "📈 Trends & Insights",
        ],
    )

    st.markdown("---")

    # Data refresh button
    if st.button("🔄 Refresh Data"):
        st.session_state.data_loaded = False
        st.experimental_rerun()

    # Data status
    if st.session_state.data_loaded:
        st.success("✅ Data loaded successfully")
    else:
        st.info("📥 Loading data...")


# Helper functions
@st.cache_resource
def load_threat_model():
    """Load the threat classification model once."""
    model_path = MODEL_DIR / "threat_model_with_sbert.pkl"
    if model_path.exists():
        try:
            import joblib

            model_data = joblib.load(model_path)
            return model_data
        except Exception as e:
            st.warning(f"Could not load threat model: {e}")
    return None


@st.cache_data(ttl=300)
def apply_threat_classification(df, _model_data):
    """Apply threat classification to dataframe using the trained model."""
    # leading underscore tells Streamlit not to try hashing this param
    if _model_data is None:
        return df

    try:
        import numpy as np
        from scipy.sparse import hstack, csr_matrix

        clf = _model_data["model"]
        tfidf = _model_data["tfidf"]
        mlb = _model_data["mlb"]
        num_cols = _model_data.get("num_cols", ["sentiment", "cvss_score"])

        # — prepare text features —
        X_txt = tfidf.transform(df["clean_text"].fillna(""))

        # — numeric features as sparse —
        X_num_arr = df[num_cols].fillna(0).values
        X_num = csr_matrix(X_num_arr)

        # — ensure total feature-count matches clf.n_features_in_ —
        n_samples, n_txt = X_txt.shape
        n_num = X_num.shape[1]
        expected_total = clf.n_features_in_
        expected_txt = expected_total - n_num

        if n_txt < expected_txt:
            # pad missing text columns with zeros
            pad = csr_matrix((n_samples, expected_txt - n_txt))
            X_txt = hstack([X_txt, pad])
        elif n_txt > expected_txt:
            # truncate any extra
            X_txt = X_txt[:, :expected_txt]

        # — stack text + numeric —
        X = hstack([X_txt, X_num])

        # — run prediction —
        y_prob = clf.predict_proba(X)

        # — attach back to DataFrame —
        out = df.copy()
        for i, cat in enumerate(mlb.classes_):
            out[cat] = y_prob[:, i]

        return out

    except ValueError as ve:
        st.warning(f"Could not apply classification model: {ve}")
        return df
    except Exception as e:
        st.warning(f"Error in classification: {e}")
        return df


@st.cache_data(ttl=300)
def load_data():
    """Load and prepare data with robust error handling."""
    try:
        if not DB_PATH.exists():
            return (
                None,
                None,
                None,
                "Database not found. Please run the pipeline first.",
            )

        df_master = load_table("master", DB_PATH)
        try:
            df_urgency = load_table("urgency_assessed", DB_PATH)
        except Exception:
            df_urgency = df_master.copy()

        try:
            df_emerging = load_table("emerging_threats", DB_PATH)
        except Exception:
            df_emerging = df_urgency.copy()

        # Add missing columns with defaults
        if "urgency_score" not in df_urgency.columns:
            df_urgency["urgency_score"] = 0.5
            df_urgency["urgency_level"] = "Medium"

        if "emerging" not in df_emerging.columns:
            df_emerging["emerging"] = False

        # Ensure datetime columns
        for df in [df_master, df_urgency, df_emerging]:
            if "published_date" in df.columns:
                df["published_date"] = pd.to_datetime(
                    df["published_date"], errors="coerce"
                )

        # Load and apply threat classification model
        model_data = load_threat_model()
        if model_data is not None:
            df_emerging = apply_threat_classification(df_emerging, model_data)

        return df_master, df_urgency, df_emerging, None

    except Exception as e:
        return None, None, None, f"Error loading data: {str(e)}"


def get_severity_color(severity):
    """Get color for severity level."""
    severity_map = {
        "critical": "#ff4444",
        "high": "#ff8800",
        "medium": "#ffbb33",
        "low": "#00C851",
        "unknown": "#6c757d",
    }
    return severity_map.get(str(severity).lower(), "#6c757d")


def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card."""
    delta_html = ""
    if delta is not None:
        color = "green" if delta_color == "normal" else "red"
        arrow = "↑" if delta > 0 else "↓"
        delta_html = f'<p style="color: {color}; margin: 0;">{arrow} {abs(delta)}%</p>'

    return f"""
    <div class="metric-card">
        <h4 style="margin: 0; color: #666;">{title}</h4>
        <h2 style="margin: 0.5rem 0; color: #333;">{value:,}</h2>
        {delta_html}
    </div>
    """


def plot_severity_distribution(df):
    """Create severity distribution chart."""
    if "severity" not in df.columns:
        return None

    severity_counts = df["severity"].value_counts()
    colors = [get_severity_color(sev) for sev in severity_counts.index]

    fig = go.Figure(
        data=[
            go.Bar(
                x=severity_counts.index,
                y=severity_counts.values,
                marker_color=colors,
                text=severity_counts.values,
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="CVE Severity Distribution",
        xaxis_title="Severity Level",
        yaxis_title="Count",
        height=400,
        showlegend=False,
    )

    return fig


def plot_timeline(df):
    """Create timeline visualization."""
    if "published_date" not in df.columns or df["published_date"].isna().all():
        return None

    # Group by source and date
    df_timeline = df.copy()
    df_timeline["date"] = df_timeline["published_date"].dt.date

    timeline_data = (
        df_timeline.groupby(["date", "source"]).size().reset_index(name="count")
    )

    fig = px.line(
        timeline_data,
        x="date",
        y="count",
        color="source",
        title="Vulnerability & Article Timeline",
        labels={"date": "Date", "count": "Count", "source": "Source"},
        line_shape="linear",
    )

    fig.update_layout(height=400, hovermode="x unified")

    return fig


# Load data
if not st.session_state.data_loaded:
    df_master, df_urgency, df_emerging, error = load_data()

    if error:
        st.error(error)
        st.stop()
    else:
        st.session_state.df_master = df_master
        st.session_state.df_urgency = df_urgency
        st.session_state.df_emerging = df_emerging
        st.session_state.data_loaded = True

# Access data from session state
df_master = st.session_state.df_master
df_urgency = st.session_state.df_urgency
df_emerging = st.session_state.df_emerging

# Pages
if page == "📊 Overview":
    # Header
    st.markdown(
        """
    <div class="dashboard-header">
        <h1 style="margin: 0;">Cyber Threat Intelligence Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0;">Real-time monitoring and analysis of cybersecurity threats</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_cves = len(df_master[df_master["source"] == "nvd"])
        st.markdown(
            create_metric_card("Total CVEs", total_cves), unsafe_allow_html=True
        )

    with col2:
        total_articles = len(df_master[df_master["source"] == "thehackernews"])
        st.markdown(
            create_metric_card("Security Articles", total_articles),
            unsafe_allow_html=True,
        )

    with col3:
        if "urgency_level" in df_urgency.columns:
            high_urgency = len(df_urgency[df_urgency["urgency_level"] == "High"])
            st.markdown(
                create_metric_card(
                    "High Urgency", high_urgency, delta=12, delta_color="inverse"
                ),
                unsafe_allow_html=True,
            )

    with col4:
        if "emerging" in df_emerging.columns:
            emerging_count = df_emerging["emerging"].sum()
            st.markdown(
                create_metric_card(
                    "Emerging Threats",
                    int(emerging_count),
                    delta=25,
                    delta_color="inverse",
                ),
                unsafe_allow_html=True,
            )

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        fig = plot_severity_distribution(df_master[df_master["source"] == "nvd"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = plot_timeline(df_master)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Recent critical vulnerabilities
    st.markdown("### 🚨 Recent Critical Vulnerabilities")

    nvd_df = df_master[df_master["source"] == "nvd"].copy()
    if "severity" in nvd_df.columns:
        critical_cves = (
            nvd_df[nvd_df["severity"].str.lower() == "critical"]
            .sort_values("published_date", ascending=False)
            .head(5)
        )

        if not critical_cves.empty:
            for _, row in critical_cves.iterrows():
                severity_class = f"alert-{row.get('severity', 'unknown').lower()}"
                st.markdown(
                    f"""
                <div class="{severity_class}">
                    <strong>{row.get('cve_id', 'Unknown')}</strong> - {row.get('published_date', 'Unknown date').strftime('%Y-%m-%d') if pd.notna(row.get('published_date')) else 'Unknown date'}
                    <br>CVSS: {row.get('cvss_score', 'N/A')} | {row.get('description', 'No description available')[:200]}...
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No critical vulnerabilities found in the current dataset.")

elif page == "🎯 Threat Analysis":
    st.title("Threat Classification Analysis")

    # Check if classification data exists or if model was applied
    threat_categories = [
        "Phishing",
        "Ransomware",
        "Malware",
        "SQLInjection",
        "XSS",
        "DDoS",
        "ZeroDay",
        "SupplyChain",
        "Other",
    ]
    available_categories = [
        cat for cat in threat_categories if cat in df_emerging.columns
    ]

    if not available_categories:
        # Try to load and apply the model
        model_data = load_threat_model()
        if model_data is not None:
            with st.spinner("Applying threat classification model..."):
                df_emerging = apply_threat_classification(df_emerging, model_data)
                available_categories = [
                    cat for cat in threat_categories if cat in df_emerging.columns
                ]

                # Update session state with classified data
                st.session_state.df_emerging = df_emerging

    if not available_categories:
        st.warning(
            "Threat classification model not found or could not be applied. Showing pattern-based analysis."
        )

        # Simple pattern-based analysis
        st.markdown("### Threat Keywords Analysis")

        patterns = {
            "Phishing": ["phish", "credential", "spoof"],
            "Ransomware": ["ransom", "encrypt", "lock"],
            "Malware": ["malware", "trojan", "virus"],
            "SQL Injection": ["sql", "injection", "database"],
            "XSS": ["xss", "script", "cross-site"],
            "DDoS": ["ddos", "denial", "service"],
        }

        threat_counts = {}
        for threat, keywords in patterns.items():
            pattern = "|".join(keywords)
            count = (
                df_master["clean_text"]
                .str.contains(pattern, case=False, na=False)
                .sum()
            )
            threat_counts[threat] = count

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(threat_counts.keys()),
                    y=list(threat_counts.values()),
                    marker_color="#667eea",
                )
            ]
        )

        fig.update_layout(
            title="Threat Keyword Frequency",
            xaxis_title="Threat Type",
            yaxis_title="Mentions",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        # Display classification results
        st.success("✅ Threat classification model applied successfully!")
        st.markdown("### Threat Category Distribution")

        category_counts = {cat: df_emerging[cat].sum() for cat in available_categories}

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(category_counts.keys()),
                    y=list(category_counts.values()),
                    marker_color=[
                        "#ff6b6b",
                        "#4ecdc4",
                        "#45b7d1",
                        "#f7b731",
                        "#5f27cd",
                        "#00d2d3",
                        "#ff9ff3",
                        "#54a0ff",
                        "#48dbfb",
                    ],
                )
            ]
        )

        fig.update_layout(
            title="Vulnerabilities by Category",
            xaxis_title="Category",
            yaxis_title="Count",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Category trends over time
        st.markdown("### Category Trends Over Time")

        selected_category = st.selectbox("Select Category", available_categories)

        if selected_category:
            # 1) Filter for any rows tagged with this category
            df_trend = df_emerging[df_emerging[selected_category] == 1].copy()

            # 2) If there are none, show an info message instead of a blank chart
            if df_trend.empty:
                st.info(f"No data to display for **{selected_category}** over time.")
            else:
                # 3) Build monthly counts
                df_trend["month"] = df_trend["published_date"].dt.to_period("M")
                monthly_trend = (
                    df_trend.groupby("month").size().reset_index(name="count")
                )
                # back to timestamps for Plotly
                monthly_trend["month"] = monthly_trend["month"].dt.to_timestamp()

                # 4) Plot
                fig = px.line(
                    monthly_trend,
                    x="month",
                    y="count",
                    title=f"Monthly Trend: {selected_category}",
                    labels={"month": "Month", "count": "Count"},
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "⚡ Urgency Monitor":
    st.title("Urgency Score Analysis")

    if "urgency_score" not in df_urgency.columns:
        st.error("Urgency data not available. Please run the urgency scoring module.")
        st.stop()

    # Urgency overview
    col1, col2 = st.columns(2)

    with col1:
        # Urgency distribution pie chart
        urgency_dist = df_urgency["urgency_level"].value_counts()

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=urgency_dist.index,
                    values=urgency_dist.values,
                    hole=0.4,
                    marker_colors=["#00C851", "#ffbb33", "#ff4444"],
                )
            ]
        )

        fig.update_layout(title="Urgency Level Distribution", height=400)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Urgency score histogram
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=df_urgency["urgency_score"], nbinsx=30, marker_color="#667eea"
                )
            ]
        )

        fig.update_layout(
            title="Urgency Score Distribution",
            xaxis_title="Urgency Score",
            yaxis_title="Frequency",
            height=400,
        )

        # Add threshold lines
        fig.add_vline(
            x=0.33, line_dash="dash", line_color="green", annotation_text="Low"
        )
        fig.add_vline(
            x=0.66, line_dash="dash", line_color="orange", annotation_text="Medium"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Top urgent threats
    st.markdown("### 🔥 Top Urgent Threats")

    top_threats = df_urgency.nlargest(10, "urgency_score")

    for _, row in top_threats.iterrows():
        urgency_level = row.get("urgency_level", "Unknown")
        urgency_class = f"alert-{urgency_level.lower()}"

        st.markdown(
            f"""
        <div class="{urgency_class}">
            <strong>{row.get('cve_id', 'Unknown')}</strong> - Score: {row.get('urgency_score', 0):.2f}
            <br>Severity: {row.get('severity', 'Unknown')} | CVSS: {row.get('cvss_score', 'N/A')}
            <br>{row.get('description', 'No description available')[:150]}...
        </div>
        """,
            unsafe_allow_html=True,
        )

elif page == "🚨 Emerging Threats":
    st.title("Emerging Threat Detection")

    if "emerging" not in df_emerging.columns:
        st.warning(
            "Emerging threat detection not available. Showing recent high-severity vulnerabilities instead."
        )

        # Show recent high-severity as proxy
        recent_high = (
            df_master[
                (df_master["source"] == "nvd")
                & (df_master["severity"].isin(["critical", "high"]))
            ]
            .sort_values("published_date", ascending=False)
            .head(10)
        )

        for _, row in recent_high.iterrows():
            st.markdown(
                f"""
            <div class="alert-high">
                <strong>{row.get('cve_id', 'Unknown')}</strong> - {row.get('published_date', 'Unknown date').strftime('%Y-%m-%d') if pd.notna(row.get('published_date')) else 'Unknown date'}
                <br>Severity: {row.get('severity', 'Unknown')} | CVSS: {row.get('cvss_score', 'N/A')}
            </div>
            """,
                unsafe_allow_html=True,
            )

    else:
        # Emerging threats metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_emerging = df_emerging["emerging"].sum()
            st.metric("Total Emerging", int(total_emerging))

        with col2:
            if "zero_day_flag" in df_emerging.columns:
                zero_day = df_emerging["zero_day_flag"].sum()
                st.metric("Zero-Day Indicators", int(zero_day))

        with col3:
            if "spike_flag" in df_emerging.columns:
                spikes = df_emerging["spike_flag"].sum()
                st.metric("Mention Spikes", int(spikes))

        with col4:
            if "if_flag" in df_emerging.columns:
                anomalies = df_emerging["if_flag"].sum()
                st.metric("Anomalies Detected", int(anomalies))

        # Emerging threats list
        st.markdown("### 🚨 Current Emerging Threats")

        emerging_list = (
            df_emerging[df_emerging["emerging"] == True]
            .sort_values("published_date", ascending=False)
            .head(10)
        )

        if not emerging_list.empty:
            for _, row in emerging_list.iterrows():
                st.markdown(
                    f"""
                <div class="alert-critical">
                    <strong>⚠️ EMERGING: {row.get('cve_id', 'Unknown')}</strong> - {row.get('published_date', 'Unknown date').strftime('%Y-%m-%d') if pd.notna(row.get('published_date')) else 'Unknown date'}
                    <br>Detection: {', '.join([flag.replace('_flag', '') for flag in ['zero_day_flag', 'spike_flag', 'if_flag'] if flag in row and row[flag]])}
                    <br>{row.get('description', 'No description available')[:200]}...
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No emerging threats detected in the current time window.")

elif page == "🔍 CVE Explorer":
    st.title("CVE Explorer")

    # Search and filters in columns
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input(
            "🔍 Search CVEs", placeholder="Enter CVE ID, keyword, or description..."
        )

    with col2:
        severity_filter = st.selectbox(
            "Severity",
            (
                ["All"] + sorted(df_master["severity"].dropna().unique().tolist())
                if "severity" in df_master.columns
                else ["All"]
            ),
        )

    with col3:
        source_filter = st.selectbox(
            "Source", ["All"] + sorted(df_master["source"].unique().tolist())
        )

    # Date range
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start_date = st.date_input(
            "Start Date", value=datetime.now() - timedelta(days=30)
        )
    with date_col2:
        end_date = st.date_input("End Date", value=datetime.now())

    # Apply filters
    filtered_df = df_master.copy()

    if search_query:
        search_mask = (
            filtered_df["cve_id"].str.contains(search_query, case=False, na=False)
            | filtered_df["description"].str.contains(
                search_query, case=False, na=False
            )
            | filtered_df["clean_text"].str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[search_mask]

    if severity_filter != "All":
        filtered_df = filtered_df[filtered_df["severity"] == severity_filter]

    if source_filter != "All":
        filtered_df = filtered_df[filtered_df["source"] == source_filter]

    # Date filter
    filtered_df = filtered_df[
        (filtered_df["published_date"].dt.date >= start_date)
        & (filtered_df["published_date"].dt.date <= end_date)
    ]

    # Results
    st.markdown(f"### Found {len(filtered_df)} results")

    if not filtered_df.empty:
        # Sort options
        sort_by = st.selectbox(
            "Sort by", ["Most Recent", "Highest CVSS", "Highest Urgency"]
        )

        if sort_by == "Most Recent":
            filtered_df = filtered_df.sort_values("published_date", ascending=False)
        elif sort_by == "Highest CVSS" and "cvss_score" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("cvss_score", ascending=False)
        elif sort_by == "Highest Urgency" and "urgency_score" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("urgency_score", ascending=False)

        # Display results
        for _, row in filtered_df.head(20).iterrows():
            with st.expander(
                f"{row.get('cve_id', 'Unknown')} - {row.get('published_date', 'Unknown').strftime('%Y-%m-%d') if pd.notna(row.get('published_date')) else 'Unknown'}"
            ):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**Source:** {row.get('source', 'Unknown')}")
                    if "severity" in row:
                        st.markdown(f"**Severity:** {row.get('severity', 'Unknown')}")

                with col2:
                    if "cvss_score" in row:
                        st.metric(
                            "CVSS Score",
                            (
                                f"{row.get('cvss_score', 0):.1f}"
                                if pd.notna(row.get("cvss_score"))
                                else "N/A"
                            ),
                        )

                with col3:
                    if "urgency_score" in row:
                        st.metric(
                            "Urgency",
                            (
                                f"{row.get('urgency_score', 0):.2f}"
                                if pd.notna(row.get("urgency_score"))
                                else "N/A"
                            ),
                        )

                st.markdown("---")
                st.markdown(row.get("description", "No description available"))

                if (
                    "products" in row
                    and isinstance(row["products"], list)
                    and row["products"]
                ):
                    st.markdown("**Affected Products:**")
                    for product in row["products"][:5]:
                        st.markdown(f"- {product}")
    else:
        st.info("No results found matching your criteria.")

elif page == "📈 Trends & Insights":
    st.title("Trends & Insights")

    # Time period selector
    period = st.selectbox(
        "Select Time Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
    )

    # Calculate date range
    if period == "Last 7 Days":
        date_filter = datetime.now() - timedelta(days=7)
    elif period == "Last 30 Days":
        date_filter = datetime.now() - timedelta(days=30)
    elif period == "Last 90 Days":
        date_filter = datetime.now() - timedelta(days=90)
    else:
        date_filter = df_master["published_date"].min()

    # Filter data
    trend_df = df_master[df_master["published_date"] >= date_filter].copy()

    # Trend visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Severity trends
        st.markdown("### Severity Trends")
        if "severity" in trend_df.columns:
            severity_time = (
                trend_df.groupby(
                    [pd.Grouper(key="published_date", freq="W"), "severity"]
                )
                .size()
                .reset_index(name="count")
            )

            fig = px.line(
                severity_time,
                x="published_date",
                y="count",
                color="severity",
                title="Weekly Severity Trends",
                color_discrete_map={
                    "critical": "#ff4444",
                    "high": "#ff8800",
                    "medium": "#ffbb33",
                    "low": "#00C851",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Source activity
        st.markdown("### Source Activity")
        source_time = (
            trend_df.groupby([pd.Grouper(key="published_date", freq="D"), "source"])
            .size()
            .reset_index(name="count")
        )

        fig = px.area(
            source_time,
            x="published_date",
            y="count",
            color="source",
            title="Daily Activity by Source",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Key insights
    st.markdown("### 📊 Key Insights")

    insights = []

    # Calculate insights
    if "severity" in trend_df.columns:
        critical_pct = (trend_df["severity"].str.lower() == "critical").mean() * 100
        insights.append(
            f"🔴 {critical_pct:.1f}% of recent vulnerabilities are critical severity"
        )

    if "urgency_score" in trend_df.columns:
        avg_urgency = trend_df["urgency_score"].mean()
        insights.append(f"⚡ Average urgency score: {avg_urgency:.2f}")

    if "emerging" in trend_df.columns:
        emerging_pct = trend_df["emerging"].mean() * 100
        insights.append(
            f"🚨 {emerging_pct:.1f}% of recent threats are flagged as emerging"
        )

    # Most common products
    if "products" in trend_df.columns:
        all_products = []
        for products in trend_df["products"].dropna():
            if isinstance(products, list):
                all_products.extend(products)

        if all_products:
            from collections import Counter

            product_counts = Counter(all_products).most_common(3)
            top_products = ", ".join(
                [f"{prod[0]} ({prod[1]})" for prod in product_counts]
            )
            insights.append(f"💻 Most affected products: {top_products}")

    # Display insights
    for insight in insights:
        st.info(insight)

    # Word cloud of recent threats
    st.markdown("### 🌐 Threat Landscape Word Cloud")

    if "clean_text" in trend_df.columns:
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            # Combine all text
            all_text = " ".join(trend_df["clean_text"].dropna().astype(str))

            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="viridis",
                max_words=100,
            ).generate(all_text)

            # Display
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        except ImportError:
            st.info(
                "Word cloud visualization requires the 'wordcloud' package. Install with: pip install wordcloud"
            )

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666;">
    <p>Cyber Threat Intelligence Dashboard | Last Updated: {}</p>
    <p>Made with ❤️ by Dheer Gupta</p>
</div>
""".format(
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ),
    unsafe_allow_html=True,
)
