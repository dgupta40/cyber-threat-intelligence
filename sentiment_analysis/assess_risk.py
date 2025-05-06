"""
Risk assessment module using sentiment analysis to evaluate threat severity.
"""

import os
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
from nltk.sentiment import vader

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from utils.helpers import load_from_json, save_to_json

# Risk levels definition
RISK_LEVELS = ['Low', 'Medium', 'High']


class RiskAssessor:
    """Class for assessing the risk level of cyber threats."""
    
    def __init__(self):
        """Initialize the risk assessor."""
        self.processed_dir = 'data/processed'
        self.models_dir = 'models'
        self.output_dir = 'data/processed'
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize VADER sentiment analyzer
        try:
            self.vader_analyzer = vader.SentimentIntensityAnalyzer()
        except Exception as e:
            self.logger.error(f"Error initializing VADER: {str(e)}")
            self.vader_analyzer = None
        
        # Initialize RoBERTa sentiment analyzer if available
        self.roberta_model = None
        self.roberta_tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = "cardiffnlp/twitter-roberta-base-sentiment"
                self.roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.roberta_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            except Exception as e:
                self.logger.error(f"Error initializing RoBERTa: {str(e)}")
    
    def analyze(self):
        """Analyze risks for all threats in the combined dataset."""
        self.logger.info("Starting risk assessment")
        
        try:
            # Load combined dataset
            combined_file = os.path.join(self.processed_dir, 'combined_dataset.json')
            if not os.path.exists(combined_file):
                self.logger.warning(f"Combined dataset not found: {combined_file}")
                return False
                
            data = load_from_json(combined_file)
            
            # Analyze risk for each item
            for item in data:
                try:
                    # Get text content
                    text = item.get('content', '')
                    if not text:
                        continue
                    
                    # Analyze with both methods
                    vader_result = self._analyze_with_vader(text)
                    roberta_result = self._analyze_with_roberta(text)
                    
                    # Combine risk assessments
                    combined_risk = self._combine_risk_assessments(vader_result, roberta_result)
                    
                    # Add risk assessment to item
                    item['risk_assessment'] = {
                        'vader': vader_result,
                        'roberta': roberta_result,
                        'combined': combined_risk
                    }
                    
                    # Add CVSS score if available (for NVD data)
                    if 'metadata' in item and 'base_score' in item['metadata']:
                        base_score = item['metadata']['base_score']
                        if base_score:
                            # Adjust combined risk based on CVSS score
                            cvss_risk = self._cvss_to_risk_level(base_score)
                            item['risk_assessment']['cvss'] = {
                                'score': base_score,
                                'risk_level': cvss_risk
                            }
                            
                            # Update combined risk with CVSS information
                            item['risk_assessment']['combined'] = self._adjust_with_cvss(
                                combined_risk, cvss_risk
                            )
                    
                except Exception as e:
                    self.logger.error(f"Error assessing risk for item {item.get('id', '')}: {str(e)}")
            
            # Save updated dataset
            risk_file = os.path.join(self.output_dir, 'risk_assessed_dataset.json')
            save_to_json(data, risk_file)
            
            # Generate risk summary
            self._generate_risk_summary(data)
            
            self.logger.info(f"Risk assessment completed for {len(data)} items")
            return True
        except Exception as e:
            self.logger.error(f"Error during risk assessment: {str(e)}")
            return False
    
    def _analyze_with_vader(self, text):
        """
        Analyze text with VADER sentiment analyzer.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: VADER analysis results
        """
        if not self.vader_analyzer:
            return {'error': 'VADER analyzer not available'}
            
        try:
            # Get sentiment scores
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Analyze sentiment for risk assessment
            # For cyber threats, negative sentiment often correlates with higher risk
            compound = scores['compound']
            
            # Map compound score to risk level
            # VADER compound score ranges from -1 (most negative) to 1 (most positive)
            # For cyber threats, more negative means higher risk
            if compound <= -0.25:
                risk_level = 'High'
                risk_score = min(1.0, abs(compound) * 1.5)  # Scale to 0-1
            elif compound <= 0.25:
                risk_level = 'Medium'
                risk_score = 0.5
            else:
                risk_level = 'Low'
                risk_score = max(0.0, 1.0 - compound)  # Scale to 0-1
            
            return {
                'compound': compound,
                'neg': scores['neg'],
                'neu': scores['neu'],
                'pos': scores['pos'],
                'risk_level': risk_level,
                'risk_score': float(risk_score)
            }
        except Exception as e:
            self.logger.error(f"Error in VADER analysis: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_with_roberta(self, text):
        """
        Analyze text with RoBERTa sentiment model.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: RoBERTa analysis results
        """
        if not TRANSFORMERS_AVAILABLE or not self.roberta_model:
            return {'error': 'RoBERTa model not available'}
            
        try:
            # Prepare text for model
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
                
            # Tokenize input
            inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get model output
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
            
            # RoBERTa sentiment labels: 0 = negative, 1 = neutral, 2 = positive
            # Map negative to high risk, neutral to medium, positive to low
            label_mapping = {0: 'High', 1: 'Medium', 2: 'Low'}
            scores = probabilities[0].tolist()
            predicted_label = torch.argmax(probabilities, dim=1).item()
            risk_level = label_mapping[predicted_label]
            
            # Calculate risk score (more negative = higher risk)
            risk_score = scores[0]  # Negative sentiment probability as risk score
            
            return {
                'negative_score': scores[0],
                'neutral_score': scores[1],
                'positive_score': scores[2],
                'risk_level': risk_level,
                'risk_score': float(risk_score)
            }
        except Exception as e:
            self.logger.error(f"Error in RoBERTa analysis: {str(e)}")
            return {'error': str(e)}
    
    def _combine_risk_assessments(self, vader_result, roberta_result):
        """
        Combine risk assessments from different methods.
        
        Args:
            vader_result (dict): VADER analysis results
            roberta_result (dict): RoBERTa analysis results
            
        Returns:
            dict: Combined risk assessment
        """
        # Initialize with default values
        risk_level = 'Medium'
        risk_score = 0.5
        confidence = 'Low'
        
        # Check if both methods produced valid results
        vader_valid = 'error' not in vader_result
        roberta_valid = 'error' not in roberta_result
        
        if vader_valid and roberta_valid:
            # Get risk scores and levels from both methods
            vader_score = vader_result['risk_score']
            roberta_score = roberta_result['risk_score']
            vader_level = vader_result['risk_level']
            roberta_level = roberta_result['risk_level']
            
            # Calculate combined score
            combined_score = (vader_score + roberta_score) / 2
            
            # Determine combined risk level
            if combined_score >= 0.7:
                risk_level = 'High'
            elif combined_score >= 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Set confidence based on agreement between methods
            if vader_level == roberta_level:
                confidence = 'High'
            else:
                confidence = 'Medium'
                
            risk_score = combined_score
            
        elif vader_valid:
            # Use only VADER results
            risk_level = vader_result['risk_level']
            risk_score = vader_result['risk_score']
            confidence = 'Medium'
            
        elif roberta_valid:
            # Use only RoBERTa results
            risk_level = roberta_result['risk_level']
            risk_score = roberta_result['risk_score']
            confidence = 'Medium'
        
        return {
            'risk_level': risk_level,
            'risk_score': float(risk_score),
            'confidence': confidence
        }
    
    def _cvss_to_risk_level(self, cvss_score):
        """
        Convert CVSS score to risk level.
        
        Args:
            cvss_score (float): CVSS base score
            
        Returns:
            str: Risk level
        """
        try:
            score = float(cvss_score)
            
            # CVSS v3 score mapping
            if score >= 9.0:
                return 'Critical'
            elif score >= 7.0:
                return 'High'
            elif score >= 4.0:
                return 'Medium'
            elif score >= 0.1:
                return 'Low'
            else:
                return 'None'
        except (ValueError, TypeError):
            return 'Unknown'
    
    def _adjust_with_cvss(self, combined_risk, cvss_risk):
        """
        Adjust combined risk assessment with CVSS information.
        
        Args:
            combined_risk (dict): Combined risk assessment
            cvss_risk (str): CVSS-based risk level
            
        Returns:
            dict: Adjusted risk assessment
        """
        # Create copy of combined risk
        adjusted_risk = combined_risk.copy()
        
        # Map CVSS risk levels to numeric values
        cvss_risk_values = {
            'Critical': 1.0,
            'High': 0.8,
            'Medium': 0.5,
            'Low': 0.2,
            'None': 0.0,
            'Unknown': None
        }
        
        # Get numeric value for CVSS risk
        cvss_value = cvss_risk_values.get(cvss_risk)
        
        if cvss_value is not None:
            # Weighted average of combined risk and CVSS risk
            # Give more weight to CVSS as it's more standardized
            current_score = combined_risk['risk_score']
            adjusted_score = (current_score * 0.3) + (cvss_value * 0.7)
            
            # Update risk score
            adjusted_risk['risk_score'] = float(adjusted_score)
            
            # Update risk level based on adjusted score
            if adjusted_score >= 0.7:
                adjusted_risk['risk_level'] = 'High'
            elif adjusted_score >= 0.3:
                adjusted_risk['risk_level'] = 'Medium'
            else:
                adjusted_risk['risk_level'] = 'Low'
            
            # Set confidence to high when CVSS is available
            adjusted_risk['confidence'] = 'High'
        
        return adjusted_risk
    
    def _generate_risk_summary(self, data):
        """
        Generate risk summary statistics.
        
        Args:
            data (list): Risk-assessed dataset
        """
        # Count risk levels
        risk_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'Unknown': 0}
        source_risks = {}
        category_risks = {}
        
        for item in data:
            # Skip items without risk assessment
            if 'risk_assessment' not in item or 'combined' not in item['risk_assessment']:
                risk_counts['Unknown'] += 1
                continue
                
            # Get risk level
            risk_level = item['risk_assessment']['combined']['risk_level']
            risk_counts[risk_level] += 1
            
            # Track by source
            source = item.get('source', 'unknown')
            if source not in source_risks:
                source_risks[source] = {'High': 0, 'Medium': 0, 'Low': 0}
            source_risks[source][risk_level] += 1
            
            # Track by category if available
            if 'category' in item:
                category = item['category']
                if category not in category_risks:
                    category_risks[category] = {'High': 0, 'Medium': 0, 'Low': 0}
                category_risks[category][risk_level] += 1
        
        # Calculate percentages
        total_items = len(data)
        risk_percentages = {
            level: (count / total_items * 100) if total_items > 0 else 0
            for level, count in risk_counts.items()
        }
        
        # Calculate average risk score
        risk_scores = [
            item['risk_assessment']['combined']['risk_score']
            for item in data
            if 'risk_assessment' in item and 'combined' in item['risk_assessment']
            and 'risk_score' in item['risk_assessment']['combined']
        ]
        
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_items': total_items,
            'risk_counts': risk_counts,
            'risk_percentages': {k: float(v) for k, v in risk_percentages.items()},
            'average_risk_score': float(avg_risk_score),
            'by_source': source_risks,
            'by_category': category_risks
        }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'risk_summary.json')
        save_to_json(summary, summary_file)
        
        self.logger.info(f"Risk summary generated: {summary_file}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run risk assessment
    assessor = RiskAssessor()
    assessor.analyze()