"""NewsBot Intelligence System 2.0 - Source Code Package"""

__version__ = "2.0.0"
__author__ = "Leroy Brown"

from .preprocessing import TextPreprocessor, preprocess_article
from .sentiment import SentimentAnalyzer, analyze_article_sentiment

__all__ = [
    'TextPreprocessor',
    'preprocess_article',
    'SentimentAnalyzer',
    'analyze_article_sentiment',
]
