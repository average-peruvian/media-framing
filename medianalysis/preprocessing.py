"""
Preprocesamiento de texto.
"""

import re, unidecode
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class TextPreprocessor:
    def __init__(self):
        self.sw = stopwords.words('spanish')
        self.snow = SnowballStemmer('spanish')

    def clean_text(self, text: str):
        if not text: return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def tokenize(self, text: str):
        tokens = text.split()
        return [self.snow.stem(unidecode.unidecode(t)) for t in tokens if t not in self.sw and len(t) > 2]
    
    def process_df(self, df: pd.DataFrame, text_column: str = 'body'):
        df = df.copy()
        df['texto_limpio'] = df[text_column].apply(self.clean_text)
        df['tokens'] = df['texto_limpio'].apply(self.tokenize)
        df['texto_procesado'] = df['tokens'].apply(lambda x: ' '.join(x))
        return df