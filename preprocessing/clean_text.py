import os
import re
import logging
import json
gzip = None
from datetime import datetime
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

from utils.helpers import get_all_files, save_to_json, load_from_json

# Ensure NLTK resources are available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass

class TextPreprocessor:
    """Cleans and preprocesses text for HackerNews and NVD data sources."""
    def __init__(self):
        self.raw_dir = 'data/raw'
        self.processed_dir = 'data/processed'
        os.makedirs(self.processed_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def process_all_sources(self):
        self.logger.info("Starting preprocessing for HackerNews and NVD")
        self._process_hackernews()
        self._process_nvd()
        self._create_combined_dataset()
        self._generate_embeddings()
        self.logger.info("Preprocessing complete")

    def _process_hackernews(self):
        self.logger.info("Processing HackerNews data")
        src = os.path.join(self.raw_dir, 'hackernews')
        files = get_all_files(src, '.json')
        all_docs = []
        for path in files:
            data = load_from_json(path)
            articles = data if isinstance(data, list) else data.get('articles', data if isinstance(data, dict) else [])
            for art in articles:
                title = art.get('title','')
                content = art.get('content','')
                if not content: continue
                clean = self._clean_text(content)
                tokens = self._tokenize(clean)
                all_docs.append({
                    'source':'hackernews',
                    'id': art.get('url',''),
                    'title': title,
                    'content': clean,
                    'tokens': tokens,
                    'date': art.get('date',''),
                    'metadata': {'tags': art.get('tags',[]), 'url': art.get('url',''), 'cves': art.get('cves', [])}
                })
        if all_docs:
            out = os.path.join(self.processed_dir, 'hackernews_processed.json')
            save_to_json(all_docs, out)
            self.logger.info(f"Saved {len(all_docs)} HackerNews docs")

    def _process_nvd(self):
        self.logger.info("Processing NVD data")
        src = os.path.join(self.raw_dir, 'nvd')
        files = get_all_files(src, '.json')
        docs = []
        for path in files:
            data = load_from_json(path)
            if 'vulnerabilities' in data:
                for item in data['vulnerabilities']:
                    cve = item.get('cve',{})
                    descs = cve.get('descriptions',[]) or []
                    text = next((d['value'] for d in descs if d.get('lang')=='en'), '')
                    cve_id = cve.get('id','')
                    metrics = item.get('metrics',{})
                    cvss3 = metrics.get('cvssMetricV31', [{}])[0].get('cvssData',{})
                    cvss2 = metrics.get('cvssMetricV2',  [{}])[0].get('cvssData',{})
                    score = cvss3.get('baseScore', cvss2.get('baseScore',0))
                    clean = self._clean_text(text)
                    tokens = self._tokenize(clean)
                    docs.append({
                        'source':'nvd',
                        'id': cve_id,
                        'title': cve_id,
                        'content': clean,
                        'tokens': tokens,
                        'date': item.get('publishedDate',''),
                        'metadata': {'base_score': score, 'last_modified': item.get('lastModified','')}
                    })
            elif 'CVE_Items' in data:
                for item in data['CVE_Items']:
                    cve = item.get('cve',{})
                    cve_id = cve.get('CVE_data_meta',{}).get('ID','')
                    descs = cve.get('description',{}).get('description_data',[])
                    text = next((d['value'] for d in descs if d.get('lang')=='en'), '')
                    impact = item.get('impact',{})
                    cvss3 = impact.get('baseMetricV3',{}).get('cvssV3',{})
                    cvss2 = impact.get('baseMetricV2',{}).get('cvssV2',{})
                    score = cvss3.get('baseScore', cvss2.get('baseScore',0))
                    clean = self._clean_text(text)
                    tokens = self._tokenize(clean)
                    docs.append({
                        'source':'nvd',
                        'id': cve_id,
                        'title': cve_id,
                        'content': clean,
                        'tokens': tokens,
                        'date': item.get('publishedDate', ''),
                        'metadata': {'base_score': score, 'last_modified': item.get('lastModifiedDate','')}
                    })
        if docs:
            out = os.path.join(self.processed_dir, 'nvd_processed.json')
            save_to_json(docs, out)
            self.logger.info(f"Saved {len(docs)} NVD entries")

    def _create_combined_dataset(self):
        self.logger.info("Combining datasets")
        combined = []
        for fname in ['hackernews_processed.json', 'nvd_processed.json']:
            path = os.path.join(self.processed_dir, fname)
            if not os.path.exists(path): continue
            data = load_from_json(path)
            combined.extend(data)
        if combined:
            out = os.path.join(self.processed_dir, 'combined_dataset.json')
            save_to_json(combined, out)
            self.logger.info(f"Combined dataset size: {len(combined)}")

    def _generate_embeddings(self):
        self.logger.info("Generating embeddings")
        combined_path = os.path.join(self.processed_dir, 'combined_dataset.json')
        if not os.path.exists(combined_path): return
        data = load_from_json(combined_path)
        docs = [d['content'] for d in data]
        vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85,
                                     stop_words='english', ngram_range=(1,2))
        tfidf = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out().tolist()
        tf_meta = {'features': feature_names, 'timestamp': datetime.utcnow().isoformat()}
        save_to_json(tf_meta, os.path.join(self.processed_dir, 'tfidf_metadata.json'))
        import numpy as np
        np.savez_compressed(os.path.join(self.processed_dir,'tfidf_matrix.npz'),
                            data=tfidf.data, indices=tfidf.indices,
                            indptr=tfidf.indptr, shape=tfidf.shape)
        token_lists = [d['tokens'] for d in data if d['tokens']]
        if token_lists:
            model = Word2Vec(sentences=token_lists, vector_size=300, window=5,
                             min_count=1, workers=4)
            model.save(os.path.join(self.processed_dir, 'word2vec_model.bin'))

    def _clean_text(self, text):
        text = BeautifulSoup(text, 'html.parser').get_text() if text else ''
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        # preserve hyphens, periods, digits
        text = re.sub(r'[^\w\s\.-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _tokenize(self, text):
        if not text: return []
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words and len(t)>1]
        return [self.lemmatizer.lemmatize(t) for t in tokens]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    tp = TextPreprocessor()
    tp.process_all_sources()
