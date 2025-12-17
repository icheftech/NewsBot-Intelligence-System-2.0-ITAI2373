"""NewsBot Intelligence System 2.0 - Text Preprocessing Module"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import unicodedata

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, language='english', remove_stopwords=True, lemmatize=True):
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize_flag = lemmatize
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer()
    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    def tokenize(self, text):
        return word_tokenize(text.lower())
    def remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]
    def lemmatize(self, tokens):
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    def preprocess(self, text, return_tokens=False):
        text = self.clean_text(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        if self.lemmatize_flag:
            tokens = self.lemmatize(tokens)
        tokens = [token for token in tokens if len(token) > 2]
        if return_tokens:
            return tokens
        return ' '.join(tokens)
    def extract_sentences(self, text):
        return sent_tokenize(text)
    def get_word_count(self, text):
        tokens = self.tokenize(text)
        return len(tokens)
def preprocess_article(article_text, config=None):
    config = config or {}
    preprocessor = TextPreprocessor(**config)
    return preprocessor.preprocess(article_text)
