"""
urgency_assessment.py - Assess the urgency of cyber threats

This module implements a factor-based urgency assessment model as described by
Javaid et al. (2020) and Li et al. (2021).
"""

import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class UrgencyAssessor:
    """
    Assess the urgency of cyber threats using a factor-based approach
    as recommended by Li et al. (2021).
    """
    
    def __init__(self, data_dir='data/processed', model_dir='models'):
        """Initialize the urgency assessor."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.output_dir = data_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Define urgency levels
        self.urgency_levels = ['Low', 'Medium', 'High']
        
        # Weights for different factors (based on Li et al., 2021)
        self.weights = {
            'severity': 0.35,            # Base severity weight
            'exploit_available': 0.25,   # Weight if exploits exist
            'patch_status': 0.15,        # Weight for patch availability
            'active_exploitation': 0.15, # Weight for active exploitation
            'recent_mentions': 0.10      # Weight for temporal aspects
        }
        
        # Initialize model components
        self.severity_map = {
            'critical': 1.0, 
            'high': 0.75, 
            'medium': 0.5, 
            'low': 0.25, 
            'unknown': 0.5
        }
    
    def check_exploit_availability(self, text):
        """
        Check if exploits are likely available based on text analysis.
        Following methodology from Shin et al. (2020).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Score between 0-1 indicating exploit availability
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        # Pattern for explicit exploit mentions (based on research)
        explicit_patterns = [
            r'exploit.*(available|exist|published|public|released)',
            r'(available|exist|published|public|released).*exploit',
            r'proof.?of.?concept',
            r'PoC',
            r'exploit.?code',
            r'working.?exploit'
        ]
        
        # Pattern for possible exploit discussions
        possible_patterns = [
            r'exploit.*(could|potential|possible)',
            r'exploit.*development',
            r'exploit.*in.*wild'
        ]
        
        score = 0.0
        
        # Check for explicit mentions (higher weight)
        for pattern in explicit_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.7
                break
        
        # Check for possible mentions (lower weight)
        for pattern in possible_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3
                break
        
        return min(1.0, score)
    
    def check_patch_status(self, text):
        """
        Check patch availability status based on text analysis.
        Following methodology from Li et al. (2021).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Score between 0-1 indicating patch status (1 = patched)
        """
        if not text or not isinstance(text, str):
            return 0.5  # Default to uncertain
        
        # Patterns for patched mentions
        patched_patterns = [
            r'patch.*(available|released|published)',
            r'(available|released|published).*patch',
            r'security.?update',
            r'has.been.patched',
            r'issue.*resolved'
        ]
        
        # Patterns for unpatched mentions
        unpatched_patterns = [
            r'no.?patch',
            r'patch.not.available',
            r'unpatched',
            r'zero.?day',
            r'0.?day'
        ]
        
        # Check for patched mentions
        for pattern in patched_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        
        # Check for unpatched mentions
        for pattern in unpatched_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 0.0
        
        # Default if no information
        return 0.5
    
    def check_active_exploitation(self, text):
        """
        Check if active exploitation is reported based on text analysis.
        Following methodology from Javaid et al. (2020).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Score between 0-1 indicating active exploitation
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        # Patterns for active exploitation mentions
        active_patterns = [
            r'active(ly)?.?(exploited|exploitation)',
            r'exploit.*in.*wild',
            r'exploited.*in.*wild',
            r'observed.*attack',
            r'attack.*observed',
            r'exploitation.*observed',
            r'currently.*exploit'
        ]
        
        # Check for active exploitation
        for pattern in active_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        
        return 0.0
    
    def check_recent_mentions(self, item):
        """
        Check if the threat has recent activity or mentions.
        Following methodology from Samtani et al. (2020).
        
        Args:
            item (dict): Threat item
            
        Returns:
            float: Score between 0-1 based on recency
        """
        # Default if no date information
        if 'published_date' not in item and 'date' not in item:
            return 0.5
        
        # Get the most recent date
        today = datetime.now()
        published_date = None
        
        if 'published_date' in item and item['published_date']:
            try:
                published_date = datetime.fromisoformat(item['published_date'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                try:
                    # Try alternative format
                    published_date = datetime.strptime(item['published_date'][:10], '%Y-%m-%d')
                except (ValueError, TypeError):
                    pass
        
        if published_date is None and 'date' in item and item['date']:
            try:
                published_date = datetime.fromisoformat(str(item['date']).replace('Z', '+00:00'))
            except (ValueError, TypeError):
                try:
                    published_date = datetime.strptime(str(item['date'])[:10], '%Y-%m-%d')
                except (ValueError, TypeError):
                    pass
            
        # Calculate days since publication
        if published_date:
            days_since = (today - published_date).days
            if days_since <= 7:  # Very recent (1 week)
                return 1.0
            elif days_since <= 14:  # Recent (2 weeks)
                return 0.8
            elif days_since <= 30:  # Somewhat recent (1 month)
                return 0.6
            elif days_since <= 90:  # Not so recent (3 months)
                return 0.3
            else:  # Old
                return 0.0
        
        return 0.5  # Default if dates couldn't be parsed
    
    def assess_urgency(self, item):
        """
        Assess urgency using factor-based approach.
        Following methodology from Li et al. (2021).
        
        Args:
            item (dict): Threat item to assess
            
        Returns:
            dict: Urgency assessment with scores and level
        """
        # Get text content for analysis
        content = item.get('clean_text', '') or item.get('content', '') or item.get('description', '')
        
        # Calculate base severity score
        severity_score = 0.0
        if 'severity_bin' in item:
            severity_score = self.severity_map.get(item.get('severity_bin', 'unknown'), 0.5)
        elif 'cvss_score' in item and item['cvss_score'] is not None:
            try:
                cvss = float(item['cvss_score'])
                if cvss >= 9.0:
                    severity_score = 1.0
                elif cvss >= 7.0:
                    severity_score = 0.75
                elif cvss >= 4.0:
                    severity_score = 0.5
                else:
                    severity_score = 0.25
            except (ValueError, TypeError):
                severity_score = 0.5
        
        # Calculate other factor scores
        exploit_score = self.check_exploit_availability(content)
        patch_score = self.check_patch_status(content)
        active_score = self.check_active_exploitation(content)
        recent_score = self.check_recent_mentions(item)
        
        # Calculate weighted urgency score
        urgency_score = (
            self.weights['severity'] * severity_score +
            self.weights['exploit_available'] * exploit_score +
            self.weights['patch_status'] * (1.0 - patch_score) +  # Invert: lower patch score = higher urgency
            self.weights['active_exploitation'] * active_score +
            self.weights['recent_mentions'] * recent_score
        )
        
        # Determine urgency level
        urgency_level = 'Low'
        if urgency_score >= 0.7:
            urgency_level = 'High'
        elif urgency_score >= 0.4:
            urgency_level = 'Medium'
        
        # Create urgency assessment
        assessment = {
            'urgency_score': round(urgency_score, 2),
            'urgency_level': urgency_level,
            'factors': {
                'severity': round(severity_score, 2),
                'exploit_available': round(exploit_score, 2),
                'patch_status': round(patch_score, 2),
                'active_exploitation': round(active_score, 2),
                'recent_mentions': round(recent_score, 2)
            }
        }
        
        return assessment
    
    def assess(self):
        """
        Assess the urgency of all threats in the dataset.
        """
        print("Starting urgency assessment...")
        
        # Load the dataset (preferring categorized dataset if available)
        data_file = os.path.join(self.data_dir, 'categorized_dataset.parquet')
        if not os.path.exists(data_file):
            print("Categorized dataset not found, falling back to master dataset")
            data_file = os.path.join(self.data_dir, 'master.parquet')
        
        if not os.path.exists(data_file):
            print("No dataset found for urgency assessment")
            return False
        
        df = pd.read_parquet(data_file)
        print(f"Loaded {len(df)} records")
        
        # Assess urgency for each item
        urgency_assessments = []
        for _, item in df.iterrows():
            assessment = self.assess_urgency(item)
            urgency_assessments.append(assessment)
        
        # Add assessments to DataFrame
        df['urgency_assessment'] = urgency_assessments
        df['urgency_score'] = df['urgency_assessment'].apply(lambda x: x['urgency_score'])
        df['urgency_level'] = df['urgency_assessment'].apply(lambda x: x['urgency_level'])
        
        # Save updated dataset
        output_file = os.path.join(self.output_dir, 'urgency_assessed_dataset.parquet')
        df.to_parquet(output_file, index=False)
        
        # Create summary statistics
        urgency_counts = df['urgency_level'].value_counts()
        print("\nUrgency distribution:")
        for level, count in urgency_counts.items():
            print(f"{level}: {count} ({count/len(df)*100:.1f}%)")
        
        # Calculate average factor scores
        avg_factors = {
            'severity': df['urgency_assessment'].apply(lambda x: x['factors']['severity']).mean(),
            'exploit_available': df['urgency_assessment'].apply(lambda x: x['factors']['exploit_available']).mean(),
            'patch_status': df['urgency_assessment'].apply(lambda x: x['factors']['patch_status']).mean(),
            'active_exploitation': df['urgency_assessment'].apply(lambda x: x['factors']['active_exploitation']).mean(),
            'recent_mentions': df['urgency_assessment'].apply(lambda x: x['factors']['recent_mentions']).mean()
        }
        
        print("\nAverage factor scores:")
        for factor, score in avg_factors.items():
            print(f"{factor}: {score:.2f}")
        
        print(f"\nUrgency assessment completed. Results saved to {output_file}")
        return True

if __name__ == "__main__":
    # Run urgency assessment
    assessor = UrgencyAssessor()
    assessor.assess()