"""
Anomaly detection module for identifying new or unknown attack patterns.
"""

import os
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils.helpers import load_from_json, save_to_json


class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        """
        Initialize autoencoder.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            latent_dim (int): Latent space dimension
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetector:
    """Class for detecting anomalies in cyber threat data."""
    
    def __init__(self):
        """Initialize the anomaly detector."""
        self.processed_dir = 'data/processed'
        self.models_dir = 'models'
        self.output_dir = 'data/processed'
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def detect(self):
        """Detect anomalies in the dataset."""
        self.logger.info("Starting anomaly detection")
        
        try:
            # Load risk-assessed dataset (preferred) or combined dataset
            data_file = os.path.join(self.processed_dir, 'risk_assessed_dataset.json')
            if not os.path.exists(data_file):
                self.logger.warning(f"Risk-assessed dataset not found, falling back to combined dataset")
                data_file = os.path.join(self.processed_dir, 'combined_dataset.json')
            
            if not os.path.exists(data_file):
                self.logger.error(f"No dataset found for anomaly detection")
                return False
                
            data = load_from_json(data_file)
            
            # Prepare data for anomaly detection
            features, document_map = self._prepare_data(data)
            
            if features is None or document_map is None:
                self.logger.error("Failed to prepare data for anomaly detection")
                return False
            
            # Run multiple anomaly detection methods
            isolation_forest_results = self._isolation_forest(features, document_map)
            lof_results = self._local_outlier_factor(features, document_map)
            kmeans_results = self._kmeans_clustering(features, document_map)
            
            # Combine results from different methods
            combined_results = self._combine_anomaly_results(
                [isolation_forest_results, lof_results, kmeans_results],
                document_map
            )
            
            # Run autoencoder if available
            if TORCH_AVAILABLE:
                autoencoder_results = self._autoencoder(features, document_map)
                combined_results = self._combine_anomaly_results(
                    [combined_results, autoencoder_results],
                    document_map
                )
            
            # Analyze temporal patterns
            temporal_anomalies = self._detect_temporal_anomalies(data)
            
            # Add anomaly detection results to dataset
            self._update_dataset_with_anomalies(data, combined_results, temporal_anomalies)
            
            # Save updated dataset
            anomaly_file = os.path.join(self.output_dir, 'anomaly_detected_dataset.json')
            save_to_json(data, anomaly_file)
            
            # Generate anomaly summary
            self._generate_anomaly_summary(data, combined_results, temporal_anomalies)
            
            self.logger.info("Anomaly detection completed")
            return True
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {str(e)}")
            return False
    
    def _prepare_data(self, data):
        """
        Prepare data for anomaly detection.
        
        Args:
            data (list): Dataset
            
        Returns:
            tuple: (features, document_map)
        """
        self.logger.info("Preparing data for anomaly detection")
        
        try:
            # Extract text content
            texts = []
            document_map = []
            
            for i, item in enumerate(data):
                # Get content
                content = item.get('content', '')
                
                if not content:
                    continue
                
                texts.append(content)
                document_map.append(i)
            
            if not texts:
                self.logger.warning("No text content found in dataset")
                return None, None
            
            # Try to load pre-computed TF-IDF matrix
            tfidf_matrix_file = os.path.join(self.processed_dir, 'tfidf_matrix.npz')
            if os.path.exists(tfidf_matrix_file):
                self.logger.info("Loading pre-computed TF-IDF matrix")
                try:
                    loader = np.load(tfidf_matrix_file)
                    features = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                          shape=loader['shape'])
                    
                    # Check if dimensions match
                    if features.shape[0] != len(document_map):
                        self.logger.warning("Pre-computed TF-IDF dimensions don't match current data")
                        features = self._compute_tfidf(texts)
                except Exception as e:
                    self.logger.warning(f"Error loading TF-IDF matrix: {str(e)}")
                    features = self._compute_tfidf(texts)
            else:
                # Compute TF-IDF
                features = self._compute_tfidf(texts)
            
            # Dimensionality reduction for large feature sets
            if features.shape[1] > 100:
                self.logger.info(f"Reducing feature dimensions from {features.shape[1]} to 100")
                svd = TruncatedSVD(n_components=min(100, features.shape[1] - 1))
                features = svd.fit_transform(features)
            
            return features, document_map
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return None, None
    
    def _compute_tfidf(self, texts):
        """
        Compute TF-IDF features for text data.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF features
        """
        self.logger.info("Computing TF-IDF features")
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2)
        )
        
        return vectorizer.fit_transform(texts)
    
    def _isolation_forest(self, features, document_map):
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            features (numpy.ndarray): Feature matrix
            document_map (list): Mapping from feature index to document index
            
        Returns:
            dict: Anomaly detection results
        """
        self.logger.info("Running Isolation Forest")
        
        # Initialize model
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # Fit and predict
        anomaly_scores = model.fit_predict(features)
        # Convert to anomaly score where higher values indicate more anomalous
        # -1 = anomaly, 1 = normal in Isolation Forest
        normalized_scores = -model.score_samples(features)
        
        # Identify anomalies
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score == -1:  # Anomaly
                anomalies.append({
                    'document_index': document_map[i],
                    'score': float(normalized_scores[i]),
                    'method': 'isolation_forest'
                })
        
        # Save model
        model_file = os.path.join(self.models_dir, 'isolation_forest_model.pkl')
        joblib.dump(model, model_file)
        
        return {
            'anomalies': anomalies,
            'scores': normalized_scores.tolist(),
            'document_map': document_map
        }
    
    def _local_outlier_factor(self, features, document_map):
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            features (numpy.ndarray): Feature matrix
            document_map (list): Mapping from feature index to document index
            
        Returns:
            dict: Anomaly detection results
        """
        self.logger.info("Running Local Outlier Factor")
        
        # Initialize model
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1
        )
        
        # Fit and predict
        anomaly_scores = model.fit_predict(features)
        # LOF doesn't provide a score_samples method directly, but negative_outlier_factor_ is available
        normalized_scores = -model.negative_outlier_factor_
        
        # Identify anomalies
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score == -1:  # Anomaly
                anomalies.append({
                    'document_index': document_map[i],
                    'score': float(normalized_scores[i]),
                    'method': 'local_outlier_factor'
                })
        
        return {
            'anomalies': anomalies,
            'scores': normalized_scores.tolist(),
            'document_map': document_map
        }
    
    def _kmeans_clustering(self, features, document_map):
        """
        Detect anomalies using K-means clustering.
        
        Args:
            features (numpy.ndarray): Feature matrix
            document_map (list): Mapping from feature index to document index
            
        Returns:
            dict: Anomaly detection results
        """
        self.logger.info("Running K-means clustering for anomaly detection")
        
        # Determine number of clusters
        k = min(5, features.shape[0] // 10) if features.shape[0] > 20 else 2
        k = max(k, 2)  # At least 2 clusters
        
        # Initialize model
        model = KMeans(
            n_clusters=k,
            random_state=42
        )
        
        # Fit and predict
        cluster_labels = model.fit_predict(features)
        
        # Calculate distance to cluster centers
        distances = np.zeros(features.shape[0])
        for i, label in enumerate(cluster_labels):
            distances[i] = np.linalg.norm(features[i] - model.cluster_centers_[label])
        
        # Normalize distances for scoring
        normalized_scores = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-10)
        
        # Set threshold for anomalies
        threshold = np.percentile(normalized_scores, 90)  # Top 10% as anomalies
        
        # Identify anomalies
        anomalies = []
        for i, score in enumerate(normalized_scores):
            if score > threshold:
                anomalies.append({
                    'document_index': document_map[i],
                    'score': float(score),
                    'method': 'kmeans_clustering'
                })
        
        # Save model
        model_file = os.path.join(self.models_dir, 'kmeans_model.pkl')
        joblib.dump(model, model_file)
        
        return {
            'anomalies': anomalies,
            'scores': normalized_scores.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'document_map': document_map
        }
    
    def _autoencoder(self, features, document_map):
        """
        Detect anomalies using Autoencoder.
        
        Args:
            features (numpy.ndarray): Feature matrix
            document_map (list): Mapping from feature index to document index
            
        Returns:
            dict: Anomaly detection results
        """
        self.logger.info("Running Autoencoder for anomaly detection")
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, skipping Autoencoder")
            return {
                'anomalies': [],
                'scores': [],
                'document_map': document_map
            }
        
        try:
            # Convert features to dense array if sparse
            if sp.issparse(features):
                features = features.toarray()
            
            # Convert to torch tensor
            features_tensor = torch.FloatTensor(features)
            
            # Create dataloader
            dataset = TensorDataset(features_tensor, features_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Initialize model
            input_dim = features.shape[1]
            model = Autoencoder(input_dim=input_dim)
            
            # Training parameters
            learning_rate = 1e-3
            num_epochs = 50
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train the model
            model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_features, _ in dataloader:
                    # Forward pass
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_features)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.6f}")
            
            # Compute reconstruction error for each sample
            model.eval()
            with torch.no_grad():
                outputs = model(features_tensor)
                mse = torch.mean((outputs - features_tensor) ** 2, dim=1)
                reconstruction_errors = mse.numpy()
            
            # Normalize errors for scoring
            normalized_scores = (reconstruction_errors - np.min(reconstruction_errors)) / (
                np.max(reconstruction_errors) - np.min(reconstruction_errors) + 1e-10
            )
            
            # Set threshold for anomalies
            threshold = np.percentile(normalized_scores, 90)  # Top 10% as anomalies
            
            # Identify anomalies
            anomalies = []
            for i, score in enumerate(normalized_scores):
                if score > threshold:
                    anomalies.append({
                        'document_index': document_map[i],
                        'score': float(score),
                        'method': 'autoencoder'
                    })
            
            # Save model
            model_file = os.path.join(self.models_dir, 'autoencoder_model.pt')
            torch.save(model.state_dict(), model_file)
            
            return {
                'anomalies': anomalies,
                'scores': normalized_scores.tolist(),
                'document_map': document_map
            }
            
        except Exception as e:
            self.logger.error(f"Error running Autoencoder: {str(e)}")
            return {
                'anomalies': [],
                'scores': [],
                'document_map': document_map
            }
    
    def _combine_anomaly_results(self, results_list, document_map):
        """
        Combine results from multiple anomaly detection methods.
        
        Args:
            results_list (list): List of results from different methods
            document_map (list): Mapping from feature index to document index
            
        Returns:
            dict: Combined anomaly detection results
        """
        self.logger.info("Combining anomaly detection results")
        
        # Initialize combined results
        combined_anomalies = []
        aggregated_scores = np.zeros(len(document_map))
        detection_counts = np.zeros(len(document_map))
        
        # Process results from each method
        for results in results_list:
            if not results or 'anomalies' not in results:
                continue
                
            # Add anomalies to combined list
            combined_anomalies.extend(results['anomalies'])
            
            # Aggregate scores
            if 'scores' in results and len(results['scores']) == len(document_map):
                aggregated_scores += np.array(results['scores'])
                
                # Count detections
                for anomaly in results['anomalies']:
                    idx = document_map.index(anomaly['document_index'])
                    detection_counts[idx] += 1
        
        # Normalize aggregated scores
        if np.max(aggregated_scores) > np.min(aggregated_scores):
            normalized_scores = (aggregated_scores - np.min(aggregated_scores)) / (
                np.max(aggregated_scores) - np.min(aggregated_scores)
            )
        else:
            normalized_scores = aggregated_scores
        
        # Identify consensus anomalies
        consensus_anomalies = []
        for i, count in enumerate(detection_counts):
            if count >= 2:  # Detected by at least 2 methods
                consensus_anomalies.append({
                    'document_index': document_map[i],
                    'score': float(normalized_scores[i]),
                    'detection_count': int(count),
                    'method': 'consensus'
                })
        
        return {
            'anomalies': combined_anomalies,
            'consensus_anomalies': consensus_anomalies,
            'normalized_scores': normalized_scores.tolist(),
            'detection_counts': detection_counts.tolist(),
            'document_map': document_map
        }
    
    def _detect_temporal_anomalies(self, data):
        """
        Detect anomalies in temporal patterns.
        
        Args:
            data (list): Dataset
            
        Returns:
            dict: Temporal anomaly detection results
        """
        self.logger.info("Detecting temporal anomalies")
        
        try:
            # Extract dates and create time series
            dates = []
            for item in data:
                date_str = item.get('date', '')
                
                if not date_str:
                    continue
                    
                try:
                    # Try various date formats
                    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                        try:
                            date = datetime.strptime(date_str[:10], fmt[:10])
                            dates.append(date)
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue
            
            if not dates:
                self.logger.warning("No valid dates found for temporal analysis")
                return {'spikes': [], 'clusters': []}
            
            # Sort dates
            dates.sort()
            
            # Create daily counts
            date_counts = {}
            for date in dates:
                day = date.date()
                if day not in date_counts:
                    date_counts[day] = 0
                date_counts[day] += 1
            
            # Convert to sorted list
            date_series = sorted(date_counts.items())
            days = [item[0] for item in date_series]
            counts = [item[1] for item in date_series]
            
            # Detect spikes
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            threshold = mean_count + 2 * std_count
            
            spikes = []
            for i, (day, count) in enumerate(date_series):
                if count > threshold:
                    spikes.append({
                        'date': day.isoformat(),
                        'count': count,
                        'z_score': (count - mean_count) / std_count
                    })
            
            # Detect clusters
            clusters = []
            if len(days) > 1:
                # Look for unusual clusters of activity
                consecutive_days = []
                current_cluster = []
                
                for i in range(len(days) - 1):
                    delta = (days[i+1] - days[i]).days
                    
                    if delta == 1:  # Consecutive days
                        if not current_cluster:
                            current_cluster.append((days[i], counts[i]))
                        current_cluster.append((days[i+1], counts[i+1]))
                    else:
                        if len(current_cluster) >= 3:  # At least 3 consecutive days
                            consecutive_days.append(current_cluster)
                        current_cluster = []
                
                # Add last cluster if it exists
                if len(current_cluster) >= 3:
                    consecutive_days.append(current_cluster)
                
                # Analyze each cluster
                for cluster in consecutive_days:
                    cluster_counts = [item[1] for item in cluster]
                    cluster_total = sum(cluster_counts)
                    cluster_avg = cluster_total / len(cluster)
                    
                    # Check if cluster average is significantly higher than overall average
                    if cluster_avg > mean_count * 1.5:
                        clusters.append({
                            'start_date': cluster[0][0].isoformat(),
                            'end_date': cluster[-1][0].isoformat(),
                            'days': len(cluster),
                            'total_count': cluster_total,
                            'average_count': cluster_avg
                        })
            
            return {
                'spikes': spikes,
                'clusters': clusters
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting temporal anomalies: {str(e)}")
            return {'spikes': [], 'clusters': []}
    
    def _update_dataset_with_anomalies(self, data, combined_results, temporal_anomalies):
        """
        Update dataset with anomaly detection results.
        
        Args:
            data (list): Dataset
            combined_results (dict): Combined anomaly detection results
            temporal_anomalies (dict): Temporal anomaly detection results
        """
        self.logger.info("Updating dataset with anomaly detection results")
        
        # Get anomaly information
        anomalies = combined_results.get('anomalies', [])
        consensus_anomalies = combined_results.get('consensus_anomalies', [])
        
        # Create lookup dictionaries
        anomaly_lookup = {}
        for anomaly in anomalies:
            idx = anomaly['document_index']
            if idx not in anomaly_lookup:
                anomaly_lookup[idx] = []
            anomaly_lookup[idx].append(anomaly)
        
        consensus_lookup = {}
        for anomaly in consensus_anomalies:
            idx = anomaly['document_index']
            consensus_lookup[idx] = anomaly
        
        # Update dataset items
        for i, item in enumerate(data):
            # Initialize anomaly info
            item['anomaly_detection'] = {
                'is_anomaly': False,
                'methods': [],
                'scores': {},
                'consensus': {
                    'is_consensus_anomaly': False,
                    'detection_count': 0,
                    'score': 0.0
                }
            }
            
            # Check if item is an anomaly
            if i in anomaly_lookup:
                item['anomaly_detection']['is_anomaly'] = True
                
                for anomaly in anomaly_lookup[i]:
                    method = anomaly['method']
                    score = anomaly['score']
                    
                    if method not in item['anomaly_detection']['methods']:
                        item['anomaly_detection']['methods'].append(method)
                    
                    item['anomaly_detection']['scores'][method] = score
            
            # Check if item is a consensus anomaly
            if i in consensus_lookup:
                consensus = consensus_lookup[i]
                item['anomaly_detection']['consensus'] = {
                    'is_consensus_anomaly': True,
                    'detection_count': consensus['detection_count'],
                    'score': consensus['score']
                }
        
        # Add temporal anomaly information
        temporal_info = {
            'spikes': temporal_anomalies.get('spikes', []),
            'clusters': temporal_anomalies.get('clusters', [])
        }
        
        # Check if item is part of a temporal anomaly
        for i, item in enumerate(data):
            date_str = item.get('date', '')
            
            if not date_str:
                continue
                
            try:
                item_date = None
                for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                    try:
                        item_date = datetime.strptime(date_str[:10], fmt[:10]).date()
                        break
                    except ValueError:
                        continue
                
                if item_date:
                    # Check if date is part of a spike
                    item['anomaly_detection']['temporal'] = {
                        'in_spike': False,
                        'in_cluster': False,
                        'spike_info': None,
                        'cluster_info': None
                    }
                    
                    for spike in temporal_info['spikes']:
                        spike_date = datetime.strptime(spike['date'], '%Y-%m-%d').date()
                        if item_date == spike_date:
                            item['anomaly_detection']['temporal']['in_spike'] = True
                            item['anomaly_detection']['temporal']['spike_info'] = spike
                            break
                    
                    # Check if date is part of a cluster
                    for cluster in temporal_info['clusters']:
                        start_date = datetime.strptime(cluster['start_date'], '%Y-%m-%d').date()
                        end_date = datetime.strptime(cluster['end_date'], '%Y-%m-%d').date()
                        
                        if start_date <= item_date <= end_date:
                            item['anomaly_detection']['temporal']['in_cluster'] = True
                            item['anomaly_detection']['temporal']['cluster_info'] = cluster
                            break
            except Exception as e:
                self.logger.error(f"Error processing temporal info for item {i}: {str(e)}")
    
    def _generate_anomaly_summary(self, data, combined_results, temporal_anomalies):
        """
        Generate summary of anomaly detection results.
        
        Args:
            data (list): Dataset
            combined_results (dict): Combined anomaly detection results
            temporal_anomalies (dict): Temporal anomaly detection results
        """
        self.logger.info("Generating anomaly detection summary")
        
        # Count anomalies
        anomaly_count = 0
        consensus_count = 0
        method_counts = {}
        
        for item in data:
            if 'anomaly_detection' in item:
                if item['anomaly_detection'].get('is_anomaly', False):
                    anomaly_count += 1
                
                if item['anomaly_detection'].get('consensus', {}).get('is_consensus_anomaly', False):
                    consensus_count += 1
                
                for method in item['anomaly_detection'].get('methods', []):
                    if method not in method_counts:
                        method_counts[method] = 0
                    method_counts[method] += 1
        
        # Temporal statistics
        spikes = temporal_anomalies.get('spikes', [])
        clusters = temporal_anomalies.get('clusters', [])
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_items': len(data),
            'anomaly_count': anomaly_count,
            'anomaly_percentage': (anomaly_count / len(data) * 100) if data else 0,
            'consensus_anomaly_count': consensus_count,
            'consensus_anomaly_percentage': (consensus_count / len(data) * 100) if data else 0,
            'method_counts': method_counts,
            'temporal_anomalies': {
                'spike_count': len(spikes),
                'cluster_count': len(clusters),
                'spikes': spikes,
                'clusters': clusters
            }
        }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'anomaly_summary.json')
        save_to_json(summary, summary_file)
        
        self.logger.info(f"Anomaly summary generated: {summary_file}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run anomaly detection
    detector = AnomalyDetector()
    detector.detect()