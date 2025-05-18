"""
anomaly_detection.py - Detect emerging cyber threats

This module implements anomaly detection techniques to identify emerging threats,
following methods described by Pang et al. (2021) and Shin et al. (2020).
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class EmergingThreatDetector:
    """
    Detect emerging cyber threats using anomaly detection techniques
    as recommended by Pang et al. (2021).
    """
    
    def __init__(self, data_dir='data/processed', model_dir='models'):
        """Initialize the emerging threat detector."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.output_dir = data_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Define patterns for zero-day detection (based on Shin et al., 2020)
        self.zeroday_patterns = [
            r'zero.?day',
            r'0.?day',
            r'previously.?unknown',
            r'unpatched',
            r'no.?patch.?available',
            r'undisclosed.?vulnerability'
        ]
        
        # Initialize models
        self.isolation_forest = None
        self.dbscan = None
    
    def detect_zeroday_candidates(self, data):
        """
        Detect potential zero-day vulnerabilities based on text analysis.
        Following methodology from Shin et al. (2020).
        
        Args:
            data (list): List of threat data items
            
        Returns:
            list: List of potential zero-day candidates
        """
        print("Detecting potential zero-day candidates...")
        
        candidates = []
        
        for i, item in enumerate(data):
            # Get content
            content = item.get('clean_text', '') or item.get('content', '') or item.get('description', '')
            
            if not content or not isinstance(content, str):
                continue
            
            # Check for zero-day patterns
            zeroday_match = False
            for pattern in self.zeroday_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    zeroday_match = True
                    break
            
            # Check if no CVE assigned yet
            no_cve = True
            if 'cve_id' in item and item['cve_id']:
                no_cve = False
            elif 'mentioned_cves' in item and item['mentioned_cves']:
                if isinstance(item['mentioned_cves'], list) and item['mentioned_cves']:
                    no_cve = False
                elif isinstance(item['mentioned_cves'], str) and item['mentioned_cves'].strip():
                    no_cve = False
            
            # Check recency (if available)
            is_recent = False
            for date_field in ['published_date', 'date']:
                if date_field in item and item[date_field]:
                    try:
                        pub_date = datetime.fromisoformat(str(item[date_field]).replace('Z', '+00:00'))
                        days_ago = (datetime.now() - pub_date).days
                        is_recent = days_ago <= 30  # Within the last month
                        break
                    except (ValueError, TypeError):
                        try:
                            # Try alternative format
                            pub_date = datetime.strptime(str(item[date_field])[:10], '%Y-%m-%d')
                            days_ago = (datetime.now() - pub_date).days
                            is_recent = days_ago <= 30
                            break
                        except (ValueError, TypeError):
                            pass
            
            # Score the candidate
            score = 0
            if zeroday_match:
                score += 0.6
            if no_cve:
                score += 0.3
            if is_recent:
                score += 0.2
            
            # Add high-scoring candidates
            if score >= 0.6:
                candidate = {
                    'index': i,
                    'item': item,
                    'score': score,
                    'zeroday_match': zeroday_match,
                    'no_cve': no_cve,
                    'is_recent': is_recent
                }
                candidates.append(candidate)
        
        print(f"Found {len(candidates)} potential zero-day candidates")
        return candidates
    
    def detect_mention_spikes(self, data, window_days=7, threshold=2.0):
        """
        Detect sudden increases in mentions of specific terms.
        Following methodology from Samtani et al. (2020).
        
        Args:
            data (list): List of threat data items
            window_days (int): Size of the time window to analyze
            threshold (float): Threshold for spike detection
            
        Returns:
            dict: Dictionary of term spike data
        """
        print(f"Detecting mention spikes over {window_days} day window...")
        
        # Extract publication dates
        dates = []
        for item in data:
            for date_field in ['published_date', 'date']:
                if date_field in item and item[date_field]:
                    try:
                        pub_date = datetime.fromisoformat(str(item[date_field]).replace('Z', '+00:00'))
                        dates.append(pub_date)
                        break
                    except (ValueError, TypeError):
                        try:
                            # Try alternative format
                            pub_date = datetime.strptime(str(item[date_field])[:10], '%Y-%m-%d')
                            dates.append(pub_date)
                            break
                        except (ValueError, TypeError):
                            pass
        
        if not dates:
            print("No valid dates found for mention spike analysis")
            return {}
        
        # Sort dates and find the date range
        dates.sort()
        start_date = dates[0]
        end_date = dates[-1]
        
        if (end_date - start_date).days < window_days * 2:
            print(f"Date range too small for reliable spike detection: {(end_date - start_date).days} days")
            return {}
        
        # Initialize term counter
        term_counter = {}
        recent_window_start = end_date - timedelta(days=window_days)
        
        # Count term occurrences
        for item in data:
            # Get content
            content = item.get('clean_text', '') or item.get('content', '') or item.get('description', '')
            
            if not content or not isinstance(content, str):
                continue
                
            # Get publication date
            pub_date = None
            for date_field in ['published_date', 'date']:
                if date_field in item and item[date_field]:
                    try:
                        pub_date = datetime.fromisoformat(str(item[date_field]).replace('Z', '+00:00'))
                        break
                    except (ValueError, TypeError):
                        try:
                            pub_date = datetime.strptime(str(item[date_field])[:10], '%Y-%m-%d')
                            break
                        except (ValueError, TypeError):
                            pass
            
            if not pub_date:
                continue
            
            # Determine if in recent window
            is_recent = pub_date >= recent_window_start
            
            # Extract important terms
            terms = set()
            
            # Look for capitalized terms that might be important
            term_pattern = r'\b([A-Z][a-zA-Z0-9\-_]{2,})\b'
            term_matches = re.finditer(term_pattern, content)
            for match in term_matches:
                term = match.group(1)
                if len(term) >= 3 and term.lower() not in ['cve', 'the', 'and', 'for', 'this', 'that']:
                    terms.add(term)
            
            # Look for terms in quotes
            quote_pattern = r'["\']([\w\s\-]{3,})["\']'
            quote_matches = re.finditer(quote_pattern, content)
            for match in quote_matches:
                term = match.group(1).strip()
                if len(term) >= 3:
                    terms.add(term)
                    
            # Add CVEs if available
            if 'cve_id' in item and item['cve_id']:
                terms.add(item['cve_id'])
            elif 'mentioned_cves' in item and item['mentioned_cves']:
                if isinstance(item['mentioned_cves'], list):
                    terms.update(item['mentioned_cves'])
                elif isinstance(item['mentioned_cves'], str) and len(item['mentioned_cves']) > 3:
                    terms.add(item['mentioned_cves'])
            
            # Update term counter
            for term in terms:
                if term not in term_counter:
                    term_counter[term] = {'recent': 0, 'historical': 0, 'items': []}
                
                if is_recent:
                    term_counter[term]['recent'] += 1
                else:
                    term_counter[term]['historical'] += 1
                
                term_counter[term]['items'].append(item)
        
        # Calculate spike ratio
        historical_days = (end_date - start_date).days - window_days
        if historical_days <= 0:
            historical_days = 1
        
        spikes = {}
        for term, counts in term_counter.items():
            # Only consider terms with enough mentions
            if counts['recent'] < 2:
                continue
                
            # Calculate per-day rates
            recent_rate = counts['recent'] / window_days
            historical_rate = counts['historical'] / historical_days if counts['historical'] > 0 else 0
            
            # Avoid division by zero
            if historical_rate == 0:
                if counts['recent'] >= 3:  # Require at least 3 mentions for a new term
                    ratio = float('inf')  # Infinite ratio (new term)
                else:
                    continue  # Skip terms with too few mentions
            else:
                ratio = recent_rate / historical_rate
            
            # Filter by threshold
            if ratio >= threshold:
                spikes[term] = {
                    'ratio': ratio,
                    'recent_count': counts['recent'],
                    'historical_count': counts['historical'],
                    'items': counts['items'][:5]  # Include first 5 items only
                }
        
        # Sort spikes by ratio (descending)
        spikes = {k: v for k, v in sorted(spikes.items(), key=lambda x: x[1]['ratio'], reverse=True)}
        
        print(f"Found {len(spikes)} terms with mention spikes")
        return spikes
    
    def statistical_anomaly_detection(self, data):
        """
        Use statistical methods to detect anomalies in the dataset.
        Following methodology from Pang et al. (2021).
        
        Args:
            data (DataFrame): Threat data
            
        Returns:
            list: Indices of anomalous data points
        """
        print("Performing statistical anomaly detection...")
        
        # Extract text content
        texts = []
        for _, item in data.iterrows():
            content = item.get('clean_text', '') or item.get('content', '') or item.get('description', '')
            texts.append(content)
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.7,
            ngram_range=(1, 2)
        )
        
        try:
            X = vectorizer.fit_transform(texts)
            
            # Initialize Isolation Forest
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=0.1,  # Expected percentage of anomalies
                random_state=42
            )
            
            # Fit and predict
            anomaly_predictions = self.isolation_forest.fit_predict(X.toarray())
            
            # -1 indicates anomaly, 1 indicates normal
            anomaly_indices = np.where(anomaly_predictions == -1)[0]
            
            print(f"Identified {len(anomaly_indices)} statistical anomalies")
            return anomaly_indices
            
        except Exception as e:
            print(f"Error during statistical anomaly detection: {str(e)}")
            return []
    
    def detect_emerging_clusters(self, data):
        """
        Use clustering to identify emerging patterns.
        Following methodology from Zhou et al. (2018).
        
        Args:
            data (DataFrame): Threat data
            
        Returns:
            dict: Clustering results
        """
        print("Detecting emerging threat clusters...")
        
        # Extract recent items (last 90 days)
        recent_items = []
        today = datetime.now()
        for i, (_, item) in enumerate(data.iterrows()):
            for date_field in ['published_date', 'date']:
                if date_field in item and item[date_field]:
                    try:
                        pub_date = datetime.fromisoformat(str(item[date_field]).replace('Z', '+00:00'))
                        days_ago = (today - pub_date).days
                        if days_ago <= 90:  # Within the last 90 days
                            recent_items.append((i, item))
                            break
                    except (ValueError, TypeError):
                        try:
                            pub_date = datetime.strptime(str(item[date_field])[:10], '%Y-%m-%d')
                            days_ago = (today - pub_date).days
                            if days_ago <= 90:
                                recent_items.append((i, item))
                                break
                        except (ValueError, TypeError):
                            pass
        
        if len(recent_items) < 5:
            print(f"Not enough recent items for clustering: {len(recent_items)}")
            return {}
        
        # Extract text content
        texts = []
        indices = []
        for idx, item in recent_items:
            content = item.get('clean_text', '') or item.get('content', '') or item.get('description', '')
            if content and isinstance(content, str):
                texts.append(content)
                indices.append(idx)
            
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=2,
            max_df=0.7,
            ngram_range=(1, 2)
        )
        
        try:
            X = vectorizer.fit_transform(texts)
            
            # Standardize features
            X_dense = X.toarray()
            X_scaled = StandardScaler().fit_transform(X_dense)
            
            # Perform clustering with DBSCAN
            self.dbscan = DBSCAN(eps=0.5, min_samples=3)
            labels = self.dbscan.fit_predict(X_scaled)
            
            # Count clusters (excluding noise points labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"Found {n_clusters} clusters and {n_noise} noise points")
            
            # Analyze clusters
            clusters = {}
            for i in range(n_clusters):
                # Get indices of items in this cluster
                cluster_indices = [idx for idx, label in enumerate(labels) if label == i]
                original_indices = [indices[idx] for idx in cluster_indices]
                
                # Get top terms for this cluster
                cluster_texts = [texts[idx] for idx in cluster_indices]
                cluster_vectorizer = TfidfVectorizer(max_features=20)
                cluster_tfidf = cluster_vectorizer.fit_transform(cluster_texts)
                feature_names = cluster_vectorizer.get_feature_names_out()
                
                # Calculate average TF-IDF scores per term
                cluster_avg = cluster_tfidf.toarray().mean(axis=0)
                top_term_indices = cluster_avg.argsort()[-10:][::-1]  # Top 10 terms
                top_terms = [feature_names[idx] for idx in top_term_indices]
                
                # Get dates to calculate recency
                dates = []
                for idx in cluster_indices:
                    _, item = recent_items[idx]
                    for date_field in ['published_date', 'date']:
                        if date_field in item and item[date_field]:
                            try:
                                pub_date = datetime.fromisoformat(str(item[date_field]).replace('Z', '+00:00'))
                                dates.append(pub_date)
                                break
                            except (ValueError, TypeError):
                                try:
                                    pub_date = datetime.strptime(str(item[date_field])[:10], '%Y-%m-%d')
                                    dates.append(pub_date)
                                    break
                                except (ValueError, TypeError):
                                    pass
                
                # Calculate recency score
                recency_score = 0.0
                if dates:
                    avg_days = sum((today - d).days for d in dates) / len(dates)
                    recency_score = 1.0 - min(1.0, avg_days / 90.0)  # Normalize to [0, 1]
                
                # Calculate overall cluster score (combines size and recency)
                size_score = min(1.0, len(cluster_indices) / 20.0)  # Normalize to [0, 1], capped at 20 items
                cluster_score = (size_score * 0.5) + (recency_score * 0.5)
                
                # Store cluster info
                clusters[i] = {
                    'size': len(cluster_indices),
                    'top_terms': top_terms,
                    'recency_score': recency_score,
                    'cluster_score': cluster_score,
                    'item_indices': original_indices
                }
            
            # Sort clusters by score
            clusters = {k: v for k, v in sorted(clusters.items(), key=lambda x: x[1]['cluster_score'], reverse=True)}
            
            return {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'clusters': clusters
            }
            
        except Exception as e:
            print(f"Error during clustering: {str(e)}")
            return {}
    
    def detect(self):
        """
        Detect emerging cyber threats using multiple techniques.
        """
        print("Starting emerging threat detection...")
        
        # Load the dataset (preferring more processed datasets if available)
        data_file = os.path.join(self.data_dir, 'urgency_assessed_dataset.parquet')
        if not os.path.exists(data_file):
            data_file = os.path.join(self.data_dir, 'categorized_dataset.parquet')
            if not os.path.exists(data_file):
                data_file = os.path.join(self.data_dir, 'master.parquet')
        
        if not os.path.exists(data_file):
            print("No dataset found for emerging threat detection")
            return False
        
        # Load data
        df = pd.read_parquet(data_file)
        print(f"Loaded {len(df)} records")
        
        # Run different detection methods
        
        # 1. Detect zero-day candidates
        data_list = df.to_dict('records')
        zeroday_candidates = self.detect_zeroday_candidates(data_list)
        
        # 2. Detect mention spikes
        mention_spikes = self.detect_mention_spikes(data_list)
        
        # 3. Statistical anomaly detection
        anomaly_indices = self.statistical_anomaly_detection(df)
        
        # 4. Detect emerging clusters
        cluster_results = self.detect_emerging_clusters(df)
        
        # Combine detection results
        print("Combining detection results...")
        
        # Initialize emerging threat flags
        df['is_emerging'] = False
        df['emerging_confidence'] = 0.0
        df['emerging_methods'] = df.apply(lambda x: [], axis=1)
        
        # Add zero-day candidates
        for candidate in zeroday_candidates:
            idx = candidate['index']
            df.at[idx, 'is_emerging'] = True
            df.at[idx, 'emerging_confidence'] = max(df.at[idx, 'emerging_confidence'], candidate['score'])
            methods = df.at[idx, 'emerging_methods'] + ['zeroday_candidate']
            df.at[idx, 'emerging_methods'] = methods
        
        # Add mention spikes
        for term, spike_data in mention_spikes.items():
            for item in spike_data['items']:
                item_idx = df.index[df['id'] == item.get('id', None) if 'id' in item else None]
                if len(item_idx) > 0:
                    idx = item_idx[0]
                    df.at[idx, 'is_emerging'] = True
                    spike_score = min(1.0, spike_data['ratio'] / 10.0)  # Normalize, cap at 10x ratio
                    df.at[idx, 'emerging_confidence'] = max(df.at[idx, 'emerging_confidence'], spike_score)
                    methods = df.at[idx, 'emerging_methods'] + ['term_spike']
                    df.at[idx, 'emerging_methods'] = methods
                    # Add term to spike terms
                    if 'spike_terms' not in df.at[idx]:
                        df.at[idx, 'spike_terms'] = []
                    df.at[idx, 'spike_terms'] = df.at[idx, 'spike_terms'] + [term]
        
        # Add statistical anomalies
        for idx in anomaly_indices:
            if idx < len(df):
                df.at[idx, 'is_emerging'] = True
                df.at[idx, 'emerging_confidence'] = max(df.at[idx, 'emerging_confidence'], 0.7)  # Standard confidence for IF
                methods = df.at[idx, 'emerging_methods'] + ['statistical_anomaly']
                df.at[idx, 'emerging_methods'] = methods
        
        # Add cluster members
        for cluster_id, cluster_data in cluster_results.get('clusters', {}).items():
            for idx in cluster_data['item_indices']:
                if idx < len(df):
                    df.at[idx, 'is_emerging'] = True
                    df.at[idx, 'emerging_confidence'] = max(df.at[idx, 'emerging_confidence'], cluster_data['cluster_score'])
                    methods = df.at[idx, 'emerging_methods'] + ['emerging_cluster']
                    df.at[idx, 'emerging_methods'] = methods
                    # Add cluster info
                    df.at[idx, 'cluster_id'] = cluster_id
                    df.at[idx, 'cluster_terms'] = cluster_data['top_terms']
        
        # Create emerging_threat dictionary
        df['emerging_threat'] = df.apply(
            lambda row: {
                'is_emerging': row['is_emerging'],
                'confidence': row['emerging_confidence'],
                'detection_methods': row['emerging_methods'],
                'spike_terms': row.get('spike_terms', []) if row.get('spike_terms') is not None else [],
                'cluster_terms': row.get('cluster_terms', []) if row.get('cluster_terms') is not None else []
            }, 
            axis=1
        )
        
        # Remove intermediate columns
        df = df.drop(['is_emerging', 'emerging_confidence', 'emerging_methods', 'spike_terms', 'cluster_terms'], axis=1, errors='ignore')
        
        # Save results
        output_file = os.path.join(self.output_dir, 'emerging_threats_dataset.parquet')
        df.to_parquet(output_file, index=False)
        
        # Create summary
        emerging_count = sum(1 for item in df['emerging_threat'] if item['is_emerging'])
        high_confidence = sum(1 for item in df['emerging_threat'] if item['is_emerging'] and item['confidence'] >= 0.7)
        
        print("\nEmerging threats summary:")
        print(f"Total threats: {len(df)}")
        print(f"Emerging threats: {emerging_count} ({emerging_count/len(df)*100:.1f}%)")
        print(f"High confidence: {high_confidence} ({high_confidence/emerging_count*100:.1f}% of emerging)")
        
        if mention_spikes:
            print(f"\nTop mention spikes:")
            for term, data in list(mention_spikes.items())[:5]:
                print(f"  {term}: {data['ratio']:.1f}x increase ({data['recent_count']} recent mentions)")
        
        if cluster_results.get('clusters', {}):
            print(f"\nTop emerging clusters:")
            for i, (cluster_id, data) in enumerate(list(cluster_results['clusters'].items())[:3]):
                print(f"  Cluster {i+1}: {', '.join(data['top_terms'][:5])} ({data['size']} threats)")
        
        print(f"\nResults saved to {output_file}")
        return True

if __name__ == "__main__":
    # Run emerging threat detection
    detector = EmergingThreatDetector()
    detector.detect()