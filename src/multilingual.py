"""NewsBot Intelligence System 2.0 - Multilingual Module"""
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

DetectorFactory.seed = 0

class MultilingualProcessor:
    def __init__(self):
        self.translator = None
    def detect_language(self, text):
        try:
            lang = detect(text)
            return {'language': lang, 'confidence': 0.9}
        except:
            return {'language': 'unknown', 'confidence': 0.0}
    def translate_text(self, text, target_lang='en', source_lang='auto'):
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            return {'original': text, 'translated': translated, 'source_lang': source_lang, 'target_lang': target_lang}
        except Exception as e:
            return {'error': str(e), 'original': text}
    def process_multilingual_article(self, text, target_lang='en'):
        detection = self.detect_language(text)
        if detection['language'] == target_lang:
            return {'text': text, 'language': target_lang, 'translated': False}
        translation = self.translate_text(text, target_lang=target_lang, source_lang=detection['language'])
        return {'text': translation.get('translated', text), 'original_language': detection['language'], 'target_language': target_lang, 'translated': True}
def detect_article_language(article_text):
    processor = MultilingualProcessor()
    return processor.detect_language(article_text)
def translate_article(article_text, target_lang='en'):
    processor = MultilingualProcessor()
    return processor.translate_text(article_text, target_lang=target_lang)
