"""
NewsBot Intelligence System 2.0 - Text Classification Module
Classifies news articles into categories and topics
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np


class NewsClassifier:
    """
    News article classification using machine learning
    """
    
    def __init__(self, method='naive_bayes', max_features=5000):
        """
        Initialize classifier
        
        Args:
            method: Classification method ('naive_bayes', 'logistic')
            max_features: Maximum number of features for vectorization
        """
        self.method = method
        self.max_features = max_features
        self.pipeline = None
        self.categories = []
        
    def build_pipeline(self):
        """
        Build scikit-learn pipeline for classification
        
        Returns:
            Sklearn pipeline
        """
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        if self.method == 'naive_bayes':
            classifier = MultinomialNB()
        elif self.method == 'logistic':
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
    
    def train(self, texts, labels):
        """
        Train the classifier
        
        Args:
            texts: List of article texts
            labels: List of category labels
            
        Returns:
            Training accuracy
        """
        self.categories = list(set(labels))
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(texts, labels)
        
        # Calculate training accuracy
        predictions = self.pipeline.predict(texts)
        accuracy = np.mean(predictions == np.array(labels))
        
        return accuracy
    
    def predict(self, text):
        """
        Predict category for a single article
        
        Args:
            text: Article text
            
        Returns:
            Predicted category
        """
        if not self.pipeline:
            raise ValueError("Model not trained. Call train() first.")
        
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        
        return {
            'category': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {
                cat: float(prob) 
                for cat, prob in zip(self.categories, probabilities)
            }
        }
    
    def predict_batch(self, texts):
        """
        Predict categories for multiple articles
        
        Args:
            texts: List of article texts
            
        Returns:
            List of predictions
        """
        return [self.predict(text) for text in texts]


class TopicModeler:
    """
    Topic modeling for news articles using LDA
    """
    
    def __init__(self, n_topics=10, method='lda'):
        """
        Initialize topic modeler
        
        Args:
            n_topics: Number of topics to extract
            method: Method ('lda' or 'nmf')
        """
        self.n_topics = n_topics
        self.method = method
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        
    def fit(self, texts, max_features=1000):
        """
        Fit topic model to texts
        
        Args:
