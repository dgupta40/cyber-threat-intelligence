"""
Threat classification module for training models to categorize cyber threats.
"""

import os
import logging
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from transformers import DataCollatorWithPadding
    import torch
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from utils.helpers import load_from_json, save_to_json


# Threat categories definition
THREAT_CATEGORIES = [
    'Phishing',
    'Ransomware',
    'Malware',
    'SQLInjection',
    'XSS',
    'DDoS',
    'ZeroDay',
    'SupplyChain',
    'DataBreach',
    'APT',
    'Other'
]


class CustomDataset(Dataset):
    """Custom dataset for HuggingFace Transformers."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class ThreatClassifier:
    """Class for training and using threat classification models."""
    
    def __init__(self):
        """Initialize the threat classifier."""
        self.processed_dir = 'data/processed'
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Label encoder for categories
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(THREAT_CATEGORIES)
        
    def train(self):
        """Train threat classification models."""
        self.logger.info("Starting threat classification model training")
        
        try:
            # Load and prepare data
            data = self._load_and_prepare_data()
            if not data:
                self.logger.error("No data available for training")
                return False
                
            # Train traditional ML models
            self._train_traditional_models(data)
            
            # Train deep learning models if available
            if TENSORFLOW_AVAILABLE:
                self._train_deep_learning_model(data)
            else:
                self.logger.warning("TensorFlow not available, skipping deep learning model")
            
            # Train transformer model if available
            if TRANSFORMERS_AVAILABLE:
                self._train_transformer_model(data)
            else:
                self.logger.warning("Transformers not available, skipping BERT model")
            
            self.logger.info("Threat classification model training completed")
            return True
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return False
    
    def _load_and_prepare_data(self):
        """
        Load and prepare data for training.
    
        Returns:
        dict: Prepared data dictionary
        """
        self.logger.info("Loading and preparing data for training")
        
        # Try loading the Parquet file first
        master_file = os.path.join(self.processed_dir, 'master.parquet')
        if os.path.exists(master_file):
            self.logger.info(f"Loading data from Parquet file: {master_file}")
            try:
                import pandas as pd
                df = pd.read_parquet(master_file)
                
                # Extract relevant columns - we need content for training
                # First check which content columns we have
                content_cols = []
                if 'content_thn' in df.columns:
                    content_cols.append('content_thn')
                if 'content_nvd' in df.columns:
                    content_cols.append('content_nvd')
                
                # Create a combined content column if we have multiple sources
                if len(content_cols) > 1:
                    df['content'] = df[content_cols[0]].fillna('')
                    for col in content_cols[1:]:
                        df['content'] = df['content'] + ' ' + df[col].fillna('')
                    df['content'] = df['content'].str.strip()
                elif len(content_cols) == 1:
                    df['content'] = df[content_cols[0]]
                else:
                    self.logger.error("No content columns found in the data")
                    return None
                
                # Create a title column from available title columns
                title_cols = []
                if 'title_thn' in df.columns:
                    title_cols.append('title_thn')
                if 'title_nvd' in df.columns:
                    title_cols.append('title_nvd')
                
                if len(title_cols) > 0:
                    df['title'] = df[title_cols[0]].fillna('')
                else:
                    df['title'] = ''
                
                self.logger.info(f"Loaded {len(df)} records from Parquet file")
            except Exception as e:
                self.logger.error(f"Error loading Parquet file: {str(e)}")
                return None
        # Fall back to looking for JSON files
        else:
            # Try looking for individual processed files
            hackernews_file = os.path.join(self.processed_dir, 'hackernews_processed.json')
            nvd_file = os.path.join(self.processed_dir, 'nvd_processed.json')
            
            hackernews_data = []
            nvd_data = []
            
            if os.path.exists(hackernews_file):
                hackernews_data = load_from_json(hackernews_file)
                self.logger.info(f"Loaded {len(hackernews_data)} records from HackerNews")
            
            if os.path.exists(nvd_file):
                nvd_data = load_from_json(nvd_file)
                self.logger.info(f"Loaded {len(nvd_data)} records from NVD")
            
            if not hackernews_data and not nvd_data:
                self.logger.error("No data sources found")
                return None
                
            # Combine data sources
            combined_data = hackernews_data + nvd_data
            df = pd.DataFrame(combined_data)
        
        # Make sure we have the minimum required columns
        required_cols = ['content', 'title']
        for col in required_cols:
            if col not in df.columns:
                self.logger.error(f"Required column '{col}' not found in data")
                return None
        
        # Assign initial threat categories for training
        # This is a simplified approach - in a real system, we will have labeled data
        # Here we'll use keywords and metadata for initial labeling
        
        def assign_category(row):
            """Assign threat category based on content and metadata."""
            text = row['content'].lower()
            title = row['title'].lower() if row['title'] else ''
            
            # Keyword-based labeling (simplified)
            if 'phish' in text or 'phish' in title:
                return 'Phishing'
            elif 'ransomware' in text or 'ransom' in title:
                return 'Ransomware'
            elif 'malware' in text or 'trojan' in text or 'virus' in title:
                return 'Malware'
            elif 'sql' in text and ('inject' in text or 'attack' in text):
                return 'SQLInjection'
            elif 'xss' in text or 'cross site script' in text:
                return 'XSS'
            elif 'ddos' in text or 'denial of service' in text:
                return 'DDoS'
            elif 'zero day' in text or 'zero-day' in text or '0day' in text:
                return 'ZeroDay'
            elif 'supply chain' in text:
                return 'SupplyChain'
            elif 'data breach' in text or 'breach' in title or 'leak' in text:
                return 'DataBreach'
            elif 'apt' in text or 'advanced persistent threat' in text:
                return 'APT'
            else:
                return 'Other'
        
        # Assign categories
        df['category'] = df.apply(assign_category, axis=1)
        
        # Drop rows with empty content
        df = df.dropna(subset=['content'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['content'], 
            df['category'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Encode labels
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Return prepared data
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_train_encoded': y_train_encoded,
            'y_test': y_test,
            'y_test_encoded': y_test_encoded,
            'df': df
        }
    
    def _train_traditional_models(self, data):
        """
        Train traditional ML models.
        
        Args:
            data (dict): Prepared data dictionary
        """
        self.logger.info("Training traditional ML models")
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train_encoded']
        y_test = data['y_test_encoded']
        
        # Define models to train
        models = {
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', LogisticRegression(max_iter=1000, C=1.0))
            ]),
            'random_forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', RandomForestClassifier(n_estimators=100))
            ]),
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', MultinomialNB(alpha=0.1))
            ]),
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('clf', LinearSVC(C=1.0))
            ])
        }
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            try:
                self.logger.info(f"Training {name} model")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                self.logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
                # Save model
                model_file = os.path.join(self.models_dir, f'{name}_model.pkl')
                joblib.dump(model, model_file)
                
                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'f1': f1
                }
            except Exception as e:
                self.logger.error(f"Error training {name} model: {str(e)}")
        
        # Save results
        results_file = os.path.join(self.models_dir, 'traditional_models_results.json')
        save_to_json(results, results_file)
        
        # Determine best model
        best_model = max(results, key=lambda x: results[x]['f1'])
        self.logger.info(f"Best traditional model: {best_model} (F1: {results[best_model]['f1']:.4f})")
        
        # Mark best model
        with open(os.path.join(self.models_dir, 'best_traditional_model.txt'), 'w') as f:
            f.write(best_model)
    
    def _train_deep_learning_model(self, data):
        """
        Train deep learning model using TensorFlow.
        
        Args:
            data (dict): Prepared data dictionary
        """
        self.logger.info("Training deep learning model")
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train_encoded']
        y_test = data['y_test_encoded']
        
        try:
            # Tokenize text
            tokenizer = Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(X_train)
            
            # Convert text to sequences
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            
            # Pad sequences
            max_len = 200
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
            
            # Convert to one-hot encoding
            num_classes = len(THREAT_CATEGORIES)
            y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
            y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
            
            # Build model
            model = Sequential()
            model.add(Embedding(10000, 128, input_length=max_len))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Bidirectional(LSTM(32)))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train_pad, y_train_cat,
                epochs=10,
                batch_size=32,
                validation_data=(X_test_pad, y_test_cat),
                verbose=1
            )
            
            # Evaluate model
            loss, accuracy = model.evaluate(X_test_pad, y_test_cat)
            self.logger.info(f"Deep learning model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save model
            model_file = os.path.join(self.models_dir, 'deep_learning_model')
            model.save(model_file)
            
            # Save tokenizer
            tokenizer_file = os.path.join(self.models_dir, 'tokenizer.pkl')
            with open(tokenizer_file, 'wb') as f:
                pickle.dump(tokenizer, f)
            
            # Save training history
            history_file = os.path.join(self.models_dir, 'deep_learning_history.json')
            history_dict = {
                'accuracy': [float(acc) for acc in history.history['accuracy']],
                'val_accuracy': [float(acc) for acc in history.history['val_accuracy']],
                'loss': [float(loss) for loss in history.history['loss']],
                'val_loss': [float(loss) for loss in history.history['val_loss']]
            }
            save_to_json(history_dict, history_file)
            
        except Exception as e:
            self.logger.error(f"Error training deep learning model: {str(e)}")
    
    def _train_transformer_model(self, data):
        """
        Train BERT-based transformer model using HuggingFace.
        
        Args:
            data (dict): Prepared data dictionary
        """
        self.logger.info("Training transformer model")
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train_encoded']
        y_test = data['y_test_encoded']
        
        try:
            # Load pre-trained tokenizer and model
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(THREAT_CATEGORIES)
            )
            
            # Tokenize data
            train_encodings = tokenizer(
                X_train.tolist(), 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            test_encodings = tokenizer(
                X_test.tolist(), 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            
            # Create datasets
            train_dataset = CustomDataset(train_encodings, y_train)
            test_dataset = CustomDataset(test_encodings, y_test)
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.models_dir, 'transformer_results'),
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                weight_decay=0.01,
                logging_dir=os.path.join(self.models_dir, 'logs'),
                logging_steps=10,
                evaluation_strategy="epoch"
            )
            
            # Define data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Define trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator
            )
            
            # Train model
            trainer.train()
            
            # Evaluate model
            eval_result = trainer.evaluate()
            self.logger.info(f"Transformer evaluation results: {eval_result}")
            
            # Save model and tokenizer
            model_dir = os.path.join(self.models_dir, 'transformer_model')
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            # Save evaluation results
            eval_file = os.path.join(self.models_dir, 'transformer_eval_results.json')
            save_to_json(eval_result, eval_file)
            
        except Exception as e:
            self.logger.error(f"Error training transformer model: {str(e)}")
    
    def predict(self, text, model_type='best'):
        """
        Predict threat category for a given text.
        
        Args:
            text (str): Text to classify
            model_type (str): Type of model to use ('best', 'logistic_regression', etc.)
            
        Returns:
            dict: Prediction results
        """
        try:
            # Determine which model to use
            if model_type == 'best':
                # Read best model name
                with open(os.path.join(self.models_dir, 'best_traditional_model.txt'), 'r') as f:
                    model_type = f.read().strip()
            
            # Load model
            model_file = os.path.join(self.models_dir, f'{model_type}_model.pkl')
            model = joblib.load(model_file)
            
            # Make prediction
            prediction = model.predict([text])[0]
            prediction_proba = None
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba([text])[0]
            
            # Get category name
            category = self.label_encoder.inverse_transform([prediction])[0]
            
            # Format result
            result = {
                'category': category,
                'model_used': model_type,
                'probability': None
            }
            
            if prediction_proba is not None:
                result['probability'] = float(max(prediction_proba))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train models
    classifier = ThreatClassifier()
    classifier.train()