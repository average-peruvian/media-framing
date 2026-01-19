"""
Preprocesamiento de texto.
"""

import re
import pandas as pd

class TextPreprocessor:
    def __init__(self, custom_stopwords: list):
        self.stopwords = self.__load_stopwords()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
    def __load_stopwords(self) -> set:
        return set([
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'una', 'por', 'con',
            'para', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'fue', 'este',
            'ha', 'sí', 'porque', 'esta', 'son', 'entre', 'cuando', 'muy', 'sin', 'sobre',
            'tiene', 'también', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado',
            'desde', 'todo', 'nos', 'durante', 'estados', 'todos', 'uno', 'les', 'ni', 'contra',
            'otros', 'fueron', 'ese', 'eso', 'había', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes',
            'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos',
            'mucho', 'quienes', 'nada', 'muchos', 'cual', 'sea', 'poco', 'ella', 'estar', 'haber'
        ])
        
    def clean_text(self, text: str) -> str:
        if not text: return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> list[str]:
        tokens = text.split()
        return [t for t in tokens if t not in self.stopwords and len(t) > 2]
    
    def process_df(self, df: pd.DataFrame, text_column: str = 'body') -> pd.DataFrame:
        df = df.copy()
        df['texto_limpio'] = df[text_column].apply(self.clean_text)
        df['tokens'] = df['texto_limpio'].apply(self.tokenize)
        df['texto_procesado'] = df['tokens'].apply(lambda x: ' '.join(x))
        return df


        