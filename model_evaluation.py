#!/usr/bin/env python3
"""
Model evaluation script for the Cyber Threat Intelligence system.
This script evaluates the performance of the classification, risk assessment, 
and anomaly detection models.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils.helpers import load_from_json, save_to_json

class ModelEvaluator:
    """Evaluates the performance of ML models for cyber threat intelligence."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.models_dir = 'models'
        self.data_dir = 'data/processed'
        self.output_dir = 'evaluations'
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_classifiers(self):
        """Evaluate the performance of threat classification models."""
        self.logger.info("Evaluating threat classification models")
        
        # Load results file if it exists
        results_file = os.path.join(self.models_dir, 'traditional_models_results.json')
        if not os.path.exists(results_file):
            self.logger.warning("Classification model results not found")
            return False
            
        results = load_from_json(results_file)
        
        # Load test data
        data_file = os.path.join(self.data_dir, 'combined_dataset.json')
        if not os.path.exists(data_file):
            self.logger.warning("Test data not found")
            return False
            
        data = load_from_json(data_file)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Evaluate each model
        model_names = results.keys()
        evaluation = {}
        
        for model_name in model_names:
            model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}")
                continue
                
            try:
                # Load model
                model = joblib.load(model_path)
                
                # Basic metrics from results
                accuracy = results[model_name]['accuracy']
                f1 = results[model_name]['f1']
                
                # Get cross-validation score if possible
                try:
                    if 'category' in df.columns and 'content' in df.columns:
                        X = df['content']
                        y = df['category']
                        cv_scores = cross_val_score(model, X, y, cv=5)
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    else:
                        cv_mean = None
                        cv_std = None
                except Exception as e:
                    self.logger.error(f"Error in cross-validation: {str(e)}")
                    cv_mean = None
                    cv_std = None
                
                # Store evaluation
                evaluation[model_name] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'cv_mean': float(cv_mean) if cv_mean is not None else None,
                    'cv_std': float(cv_std) if cv_std is not None else None
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {str(e)}")
        
        # Save evaluation
        eval_file = os.path.join(self.output_dir, 'classifier_evaluation.json')
        save_to_json(evaluation, eval_file)
        
        # Create visualization
        self._visualize_classifier_results(evaluation)
        
        return True
    
    def evaluate_risk_assessment(self):
        """Evaluate the performance of risk assessment models."""
        self.logger.info("Evaluating risk assessment models")
        
        # Load risk-assessed dataset
        risk_file = os.path.join(self.data_dir, 'risk_assessed_dataset.json')
        if not os.path.exists(risk_file):
            self.logger.warning("Risk-assessed dataset not found")
            return False
            
        data = load_from_json(risk_file)
        
        # Analyze risk assessment methods (VADER vs. RoBERTa)
        vader_results = []
        roberta_results = []
        combined_results = []
        
        for item in data:
            if 'risk_assessment' not in item:
                continue
                
            risk = item['risk_assessment']
            
            if 'vader' in risk and 'error' not in risk['vader']:
                vader_results.append({
                    'item_id': item.get('id', ''),
                    'risk_level': risk['vader']['risk_level'],
                    'risk_score': risk['vader']['risk_score']
                })
                
            if 'roberta' in risk and 'error' not in risk['roberta']:
                roberta_results.append({
                    'item_id': item.get('id', ''),
                    'risk_level': risk['roberta']['risk_level'],
                    'risk_score': risk['roberta']['risk_score']
                })
                
            if 'combined' in risk:
                combined_results.append({
                    'item_id': item.get('id', ''),
                    'risk_level': risk['combined']['risk_level'],
                    'risk_score': risk['combined']['risk_score'],
                    'confidence': risk['combined']['confidence']
                })
        
        # Convert to DataFrames
        vader_df = pd.DataFrame(vader_results) if vader_results else None
        roberta_df = pd.DataFrame(roberta_results) if roberta_results else None
        combined_df = pd.DataFrame(combined_results) if combined_results else None
        
        # Calculate agreement between methods
        agreement = self._calculate_risk_agreement(vader_df, roberta_df, combined_df)
        
        # Save evaluation
        evaluation = {
            'agreement': agreement,
            'vader_counts': vader_df['risk_level'].value_counts().to_dict() if vader_df is not None else {},
            'roberta_counts': roberta_df['risk_level'].value_counts().to_dict() if roberta_df is not None else {},
            'combined_counts': combined_df['risk_level'].value_counts().to_dict() if combined_df is not None else {},
            'confidence_counts': combined_df['confidence'].value_counts().to_dict() if combined_df is not None else {}
        }
        
        eval_file = os.path.join(self.output_dir, 'risk_assessment_evaluation.json')
        save_to_json(evaluation, eval_file)
        
        # Create visualization
        self._visualize_risk_assessment(vader_df, roberta_df, combined_df)
        
        return True
    
    def evaluate_anomaly_detection(self):
        """Evaluate the performance of anomaly detection models."""
        self.logger.info("Evaluating anomaly detection models")
        
        # Load anomaly-detected dataset
        anomaly_file = os.path.join(self.data_dir, 'anomaly_detected_dataset.json')
        if not os.path.exists(anomaly_file):
            self.logger.warning("Anomaly-detected dataset not found")
            return False
            
        data = load_from_json(anomaly_file)
        
        # Analyze anomaly detection methods
        anomaly_results = []
        consensus_results = []
        method_counts = {}
        
        for item in data:
            if 'anomaly_detection' not in item:
                continue
                
            anomaly = item['anomaly_detection']
            
            if anomaly.get('is_anomaly', False):
                methods = anomaly.get('methods', [])
                
                # Count methods
                for method in methods:
                    method_counts[method] = method_counts.get(method, 0) + 1
                
                # Collect anomaly details
                anomaly_results.append({
                    'item_id': item.get('id', ''),
                    'methods': methods,
                    'method_count': len(methods),
                    'scores': anomaly.get('scores', {})
                })
                
                # Check if consensus anomaly
                if anomaly.get('consensus', {}).get('is_consensus_anomaly', False):
                    consensus_results.append({
                        'item_id': item.get('id', ''),
                        'detection_count': anomaly['consensus'].get('detection_count', 0),
                        'score': anomaly['consensus'].get('score', 0.0)
                    })
        
        # Calculate evaluation metrics
        total_items = len(data)
        anomaly_count = len(anomaly_results)
        consensus_count = len(consensus_results)
        
        evaluation = {
            'total_items': total_items,
            'anomaly_count': anomaly_count,
            'anomaly_percentage': (anomaly_count / total_items * 100) if total_items > 0 else 0,
            'consensus_count': consensus_count,
            'consensus_percentage': (consensus_count / total_items * 100) if total_items > 0 else 0,
            'method_counts': method_counts
        }
        
        # Calculate method agreement
        if anomaly_results:
            df = pd.DataFrame(anomaly_results)
            method_agreement = self._calculate_method_agreement(df)
            evaluation['method_agreement'] = method_agreement
        
        # Save evaluation
        eval_file = os.path.join(self.output_dir, 'anomaly_detection_evaluation.json')
        save_to_json(evaluation, eval_file)
        
        # Create visualization
        self._visualize_anomaly_detection(evaluation, anomaly_results, consensus_results)
        
        return True
    
    def _visualize_classifier_results(self, evaluation):
        """Create visualizations for classifier evaluation."""
        # Bar chart of accuracy and F1 scores
        models = list(evaluation.keys())
        accuracy = [evaluation[m]['accuracy'] for m in models]
        f1 = [evaluation[m]['f1'] for m in models]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, accuracy, width, label='Accuracy')
        plt.bar(x + width/2, f1, width, label='F1 Score')
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Classifier Performance')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'classifier_performance.png'))
        plt.close()
    
    def _calculate_risk_agreement(self, vader_df, roberta_df, combined_df):
        """Calculate agreement between risk assessment methods."""
        agreement = {}
        
        if vader_df is not None and roberta_df is not None:
            # Merge dataframes
            merged = pd.merge(vader_df, roberta_df, on='item_id', suffixes=('_vader', '_roberta'))
            
            # Calculate agreement
            total = len(merged)
            match_count = (merged['risk_level_vader'] == merged['risk_level_roberta']).sum()
            
            agreement['vader_roberta'] = {
                'total': total,
                'match_count': int(match_count),
                'match_percentage': (match_count / total * 100) if total > 0 else 0
            }
            
            # Calculate correlation between scores
            corr = merged['risk_score_vader'].corr(merged['risk_score_roberta'])
            agreement['vader_roberta']['score_correlation'] = float(corr)
        
        return agreement
    
    def _visualize_risk_assessment(self, vader_df, roberta_df, combined_df):
        """Create visualizations for risk assessment evaluation."""
        # Plot risk level distributions
        plt.figure(figsize=(12, 8))
        
        # Create risk level distribution plots
        plt.subplot(2, 2, 1)
        if vader_df is not None:
            vader_counts = vader_df['risk_level'].value_counts()
            vader_counts.plot(kind='bar', title='VADER Risk Distribution')
        
        plt.subplot(2, 2, 2)
        if roberta_df is not None:
            roberta_counts = roberta_df['risk_level'].value_counts()
            roberta_counts.plot(kind='bar', title='RoBERTa Risk Distribution')
        
        plt.subplot(2, 2, 3)
        if combined_df is not None:
            combined_counts = combined_df['risk_level'].value_counts()
            combined_counts.plot(kind='bar', title='Combined Risk Distribution')
        
        plt.subplot(2, 2, 4)
        if combined_df is not None:
            confidence_counts = combined_df['confidence'].value_counts()
            confidence_counts.plot(kind='bar', title='Confidence Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'risk_assessment.png'))
        plt.close()
        
        # Create score comparison scatterplot
        if vader_df is not None and roberta_df is not None:
            plt.figure(figsize=(8, 8))
            
            merged = pd.merge(vader_df, roberta_df, on='item_id', suffixes=('_vader', '_roberta'))
            plt.scatter(merged['risk_score_vader'], merged['risk_score_roberta'], alpha=0.5)
            
            plt.xlabel('VADER Risk Score')
            plt.ylabel('RoBERTa Risk Score')
            plt.title('Risk Score Comparison')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'risk_score_comparison.png'))
            plt.close()
    
    def _calculate_method_agreement(self, df):
        """Calculate agreement between anomaly detection methods."""
        # Extract methods
        all_methods = set()
        for methods in df['methods']:
            all_methods.update(methods)
        
        all_methods = list(all_methods)
        
        # Calculate co-occurrence
        cooccurrence = {}
        
        for method1 in all_methods:
            cooccurrence[method1] = {}
            
            for method2 in all_methods:
                if method1 == method2:
                    continue
                    
                # Count co-occurrences
                count = 0
                for methods in df['methods']:
                    if method1 in methods and method2 in methods:
                        count += 1
                
                cooccurrence[method1][method2] = count
        
        return cooccurrence
    
    def _visualize_anomaly_detection(self, evaluation, anomaly_results, consensus_results):
        """Create visualizations for anomaly detection evaluation."""
        # Method counts bar chart
        plt.figure(figsize=(10, 6))
        
        method_counts = evaluation.get('method_counts', {})
        methods = list(method_counts.keys())
        counts = list(method_counts.values())
        
        if methods:
            plt.bar(methods, counts)
            plt.xlabel('Detection Method')
            plt.ylabel('Count')
            plt.title('Anomaly Detection Methods')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'anomaly_methods.png'))
        
        plt.close()
        
        # Method count distribution
        if anomaly_results:
            plt.figure(figsize=(8, 6))
            
            df = pd.DataFrame(anomaly_results)
            method_count_dist = df['method_count'].value_counts().sort_index()
            
            plt.bar(method_count_dist.index, method_count_dist.values)
            plt.xlabel('Number of Methods')
            plt.ylabel('Count')
            plt.title('Number of Detection Methods per Anomaly')
            plt.xticks(method_count_dist.index)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'anomaly_method_counts.png'))
            plt.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    evaluator = ModelEvaluator()
    evaluator.evaluate_classifiers()
    evaluator.evaluate_risk_assessment()
    evaluator.evaluate_anomaly_detection()