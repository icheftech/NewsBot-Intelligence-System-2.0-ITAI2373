"""NewsBot Intelligence System 2.0 - Sentiment Analysis Module"""
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

class SentimentAnalyzer:
    def __init__(self, method='vader'):
        self.method = method
        self.vader = SentimentIntensityAnalyzer()
    def analyze_vader(self, text):
        scores = self.vader.polarity_scores(text)
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return {'method': 'vader', 'sentiment': sentiment, 'scores': scores, 'confidence': abs(scores['compound'])}
    def analyze_textblob(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return {'method': 'textblob', 'sentiment': sentiment, 'scores': {'polarity': polarity, 'subjectivity': subjectivity}, 'confidence': abs(polarity)}
    def analyze_ensemble(self, text):
        vader_result = self.analyze_vader(text)
        textblob_result = self.analyze_textblob(text)
        vader_score = vader_result['scores']['compound']
        textblob_score = textblob_result['scores']['polarity']
        ensemble_score = (vader_score + textblob_score) / 2
        if ensemble_score >= 0.05:
            sentiment = 'positive'
        elif ensemble_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return {'method': 'ensemble', 'sentiment': sentiment, 'scores': {'vader': vader_score, 'textblob': textblob_score, 'ensemble': ensemble_score}, 'confidence': abs(ensemble_score), 'agreement': vader_result['sentiment'] == textblob_result['sentiment']}
    def analyze(self, text):
        if not text or not isinstance(text, str):
            return {'sentiment': 'neutral', 'scores': {}, 'confidence': 0.0, 'error': 'Invalid input'}
        if self.method == 'vader':
            return self.analyze_vader(text)
        elif self.method == 'textblob':
            return self.analyze_textblob(text)
        elif self.method == 'ensemble':
            return self.analyze_ensemble(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    def analyze_batch(self, texts):
        return [self.analyze(text) for text in texts]
    def get_sentiment_distribution(self, texts):
        results = self.analyze_batch(texts)
        sentiments = [r['sentiment'] for r in results]
        total = len(sentiments)
        distribution = {'positive': sentiments.count('positive'), 'negative': sentiments.count('negative'), 'neutral': sentiments.count('neutral')}
        percentages = {k: round((v / total) * 100, 2) if total > 0 else 0 for k, v in distribution.items()}
        return {'counts': distribution, 'percentages': percentages, 'total': total}
def analyze_article_sentiment(article_text, method='ensemble'):
    analyzer = SentimentAnalyzer(method=method)
    return analyzer.analyze(article_text)
