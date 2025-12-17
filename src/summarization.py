"""NewsBot Intelligence System 2.0 - Text Summarization Module"""
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

class TextSummarizer:
    def __init__(self, method='extractive'):
        self.method = method
        self.abstractive_model = None
    def extractive_summary(self, text, num_sentences=3):
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-num_sentences:][::-1]]
        return ' '.join(ranked_sentences)
    def abstractive_summary(self, text, max_length=150, min_length=50):
        if self.abstractive_model is None:
            self.abstractive_model = pipeline('summarization', model='facebook/bart-large-cnn')
        result = self.abstractive_model(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    def summarize(self, text, num_sentences=3, max_length=150):
        if not text or len(text.strip()) < 100:
            return text
        if self.method == 'extractive':
            return self.extractive_summary(text, num_sentences)
        elif self.method == 'abstractive':
            return self.abstractive_summary(text, max_length)
        else:
            return self.extractive_summary(text, num_sentences)
def summarize_article(article_text, method='extractive', num_sentences=3):
    summarizer = TextSummarizer(method=method)
    return summarizer.summarize(article_text, num_sentences=num_sentences)
